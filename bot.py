import os, re, glob, json, time, asyncio, logging, threading, tempfile, shutil, subprocess
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
from telethon import TelegramClient, events
from docx import Document
from fpdf import FPDF

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bot")

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]
BOT_TOKEN = os.environ["BOT_TOKEN"]
GROQ_KEY = os.environ["GROQ_API_KEY"]
MODEL = os.getenv("GROQ_MODEL", "whisper-large-v3")   # eng aniqi
FALLBACK_MODEL = "whisper-large-v3-turbo"
CHUNK_SEC = 600
DATA = Path(os.getenv("DATA_DIR", "/tmp/data"))
DATA.mkdir(parents=True, exist_ok=True)
OWNER_FILE = DATA / "owner.txt"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
URL = "https://api.groq.com/openai/v1/audio/transcriptions"

def get_owner():
    if os.getenv("OWNER_ID"):
        return int(os.environ["OWNER_ID"])
    if OWNER_FILE.exists():
        return int(OWNER_FILE.read_text().strip())
    return None

client = TelegramClient(str(DATA / "session"), API_ID, API_HASH)
lock = asyncio.Lock()

# ---------- health server + self-ping (bepul hostlar uxlab qolmasligi uchun) ----------
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *a): pass

threading.Thread(target=lambda: HTTPServer(("0.0.0.0", int(os.getenv("PORT", "7860"))), H).serve_forever(), daemon=True).start()

def keepalive():
    host = os.getenv("SPACE_HOST")
    if not host:
        return
    while True:
        time.sleep(600)
        try:
            requests.get(f"https://{host}", timeout=20)
        except Exception:
            pass
threading.Thread(target=keepalive, daemon=True).start()

# ---------- Groq transcription ----------
def call_groq(path, model, prompt):
    for attempt in range(8):
        with open(path, "rb") as fh:
            r = requests.post(
                URL, headers={"Authorization": f"Bearer {GROQ_KEY}"},
                files={"file": (os.path.basename(path), fh)},
                data={"model": model, "language": "en", "temperature": "0",
                      "response_format": "verbose_json", **({"prompt": prompt} if prompt else {})},
                timeout=600)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            body = r.text
            if "per day" in body or "ASPD" in body or "RPD" in body:
                raise RuntimeError("Groq kunlik limiti tugadi. Ertaga qayta urinib ko'ring.")
            wait = float(r.headers.get("retry-after", 30))
            if wait > 3700:
                raise RuntimeError("Groq limiti tugadi, juda uzoq kutish kerak.")
            log.info("429, %s soniya kutiladi", wait)
            yield_wait(wait)
            continue
        if r.status_code >= 500:
            time.sleep(5 * (attempt + 1)); continue
        raise RuntimeError(f"Groq xatosi {r.status_code}: {r.text[:200]}")
    raise RuntimeError("Groq javob bermadi.")

_wait_cb = {"f": None}
def yield_wait(sec):
    if _wait_cb["f"]:
        _wait_cb["f"](f"Groq soatlik limiti: {int(sec)} soniya kutilmoqda...")
    time.sleep(sec + 1)

def probe(src):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=format_name,duration",
         "-of", "json", src], capture_output=True, text=True).stdout
    try:
        d = json.loads(out)["format"]
        return d.get("format_name", ""), float(d.get("duration", 0))
    except Exception:
        return "", 0.0

def transcribe(src, tmp, progress):
    _wait_cb["f"] = progress
    fmt_name, dur = probe(src)
    seg_args = ["-f", "segment", "-segment_time", str(CHUNK_SEC), "-reset_timestamps", "1"]
    # mp3 bo'lsa va bo'lak 25 MB dan kichik bo'lsa - qayta kodlamasdan kesamiz (juda tez)
    if "mp3" in fmt_name and dur > 0 and os.path.getsize(src) / dur * CHUNK_SEC < 20e6:
        cmd = ["ffmpeg", "-y", "-i", src, "-vn", "-c:a", "copy"] + seg_args + [os.path.join(tmp, "c_%03d.mp3")]
    else:
        cmd = ["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac"] + seg_args + [os.path.join(tmp, "c_%03d.flac")]
    subprocess.run(cmd, check=True, capture_output=True)
    chunks = sorted(glob.glob(os.path.join(tmp, "c_*.*")))
    if not chunks:
        raise RuntimeError("Audio o'qib bo'lmadi.")
    segs, tail, model = [], "", MODEL
    for i, c in enumerate(chunks):
        progress(f"Transkripsiya: {i + 1}/{len(chunks)} qism")
        try:
            res = call_groq(c, model, tail)
        except RuntimeError as e:
            if model == MODEL and "kunlik" in str(e):
                model = FALLBACK_MODEL
                res = call_groq(c, model, tail)
            else:
                raise
        text_parts = []
        for s in res.get("segments", []):
            t = s["text"].strip()
            if not t:
                continue
            if s.get("no_speech_prob", 0) > 0.7 and s.get("avg_logprob", 0) < -1.0:
                continue
            segs.append((s["start"] + i * CHUNK_SEC, s["end"] + i * CHUNK_SEC, t))
            text_parts.append(t)
        tail = " ".join(text_parts)[-400:]
    return segs

# ---------- output ----------
def paragraphs(segs):
    paras, cur, cur_len, prev_end = [], [], 0, 0.0
    for st, en, t in segs:
        if cur and ((st - prev_end > 1.5 and cur_len > 300) or cur_len > 700):
            paras.append(" ".join(cur)); cur, cur_len = [], 0
        cur.append(t); cur_len += len(t); prev_end = en
    if cur:
        paras.append(" ".join(cur))
    return paras

def save_txt(paras, p):
    Path(p).write_text("\n\n".join(paras), encoding="utf-8")

def save_docx(paras, p, title):
    d = Document(); d.add_heading(title, level=1)
    for x in paras:
        d.add_paragraph(x)
    d.save(p)

def save_pdf(paras, p, title):
    pdf = FPDF(); pdf.set_auto_page_break(True, 15); pdf.add_page()
    fam = "Helvetica"
    if os.path.exists(FONT):
        pdf.add_font("DJ", "", FONT); fam = "DJ"
    pdf.set_font(fam, size=14)
    pdf.multi_cell(0, 9, title, new_x="LMARGIN", new_y="NEXT"); pdf.ln(3)
    pdf.set_font(fam, size=11)
    for x in paras:
        if fam == "Helvetica":
            x = x.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6.5, x, new_x="LMARGIN", new_y="NEXT"); pdf.ln(2)
    pdf.output(p)

def fmt(sec):
    sec = int(sec); m, s = divmod(sec, 60); h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

# ---------- Telegram ----------
async def safe_edit(msg, text):
    try:
        await msg.edit(text)
    except Exception:
        pass

@client.on(events.NewMessage(incoming=True))
async def on_msg(ev):
    uid = ev.sender_id
    text = (ev.raw_text or "").strip()
    owner = get_owner()
    if text.startswith("/start"):
        if owner is None:
            OWNER_FILE.write_text(str(uid)); owner = uid
        if uid == owner:
            await ev.reply("Listening audio yuboring — TXT, Word va PDF transkripsiya qaytaraman.\n"
                           f"Sizning ID: {uid}")
        return
    if uid != owner or text.startswith("/"):
        return
    f = ev.message.file
    if f is None or ev.message.photo:
        await ev.reply("Audio fayl yuboring.")
        return

    status = await ev.reply("Navbatda..." if lock.locked() else "Qabul qilindi...")
    async with lock:
        tmp = tempfile.mkdtemp(dir=DATA)
        loop = asyncio.get_running_loop()
        try:
            await safe_edit(status, "Yuklab olinmoqda...")
            src = await client.download_media(ev.message, file=os.path.join(tmp, "in"))

            def progress(t):
                asyncio.run_coroutine_threadsafe(safe_edit(status, t), loop)

            t0 = time.time()
            segs = await loop.run_in_executor(None, transcribe, src, tmp, progress)
            if not segs:
                await safe_edit(status, "Nutq topilmadi.")
                return
            base = Path(f.name).stem if f.name else "transcript"
            title = f"Transcript - {base}"
            paras = paragraphs(segs)
            files = []
            for ext, fn in (("txt", lambda p: save_txt(paras, p)),
                            ("docx", lambda p: save_docx(paras, p, title)),
                            ("pdf", lambda p: save_pdf(paras, p, title))):
                p = os.path.join(tmp, f"{base}.{ext}")
                fn(p); files.append(p)
            await client.send_file(ev.chat_id, files, force_document=True,
                                   caption=f"Davomiyligi: {fmt(segs[-1][1])} | Ish vaqti: {fmt(time.time() - t0)}")
            await safe_edit(status, "Tayyor ✅")
        except Exception as e:
            log.exception("xato")
            await safe_edit(status, f"Xatolik: {e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

async def main():
    await client.start(bot_token=BOT_TOKEN)
    log.info("Bot ishga tushdi")
    await client.run_until_disconnected()

asyncio.run(main())
