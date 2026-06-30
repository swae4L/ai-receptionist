import os, json, base64, asyncio, re
from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
from dotenv import load_dotenv
import websockets, httpx
from openai import AsyncOpenAI

load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
CALCOM_API_KEY = os.getenv("CALCOM_API_KEY")
CALCOM_EVENT_TYPE_ID = os.getenv("CALCOM_EVENT_TYPE_ID")
openai_client = AsyncOpenAI()
app = FastAPI()


def system_prompt():
    now = datetime.now(ZoneInfo("America/Toronto"))
    return (
        f"You are a friendly phone receptionist for a dental clinic. "
        f"The current date and time is {now.strftime('%A, %B %d, %Y, %I:%M %p')}. "
        f"Use this to resolve relative dates like 'Thursday' or 'tomorrow' into actual calendar dates. "
        f"Keep replies short and natural. When a caller wants to book, collect their name, "
        f"preferred date, time, and email address. Callers often say emails out loud, like "
        f"'george dot smith at gmail dot com' — convert that into standard format "
        f"('george.smith@gmail.com'). "
        f"Email accuracy is critical, so always confirm it before booking: read the full email "
        f"back clearly, letter group by letter group, and ask 'Is that correct?'. "
        f"If the caller says any part is wrong, ask them to spell just that part one letter at a "
        f"time. They may use the phonetic alphabet — 'B as in Bravo' means the letter B, so take "
        f"only the letter. Rebuild the full email, read it back again, and keep repeating this "
        f"confirm-and-correct loop until the caller explicitly agrees it is right. "
        f"Only after the caller confirms the email, call the book_appointment tool with the date "
        f"in YYYY-MM-DD format. Confirm the booking back in one sentence."
    )


TOOLS = [{
    "type": "function",
    "function": {
        "name": "book_appointment",
        "description": "Book an appointment once name, date, and time are known.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Caller's full name"},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "time": {"type": "string", "description": "Time, e.g. '3:00 PM'"},
                "email": {"type": "string", "description": "Caller's email in standard format, e.g. 'george.smith@gmail.com'"}
            },
            "required": ["name", "date", "time", "email"]
        }
    }
}]


def strip_phonetics(text):
    """Collapse phonetic-alphabet spellings down to their single letter.
    'b as in bravo' -> 'b', 'm for mike' -> 'm'. When a caller corrects a
    misheard letter they often spell it this way, so we keep just the
    letter and drop the example word.

    The regex pieces:
      \\b([a-z])      a single standalone letter, captured as group 1
      \\s+            one or more spaces
      (?:as in|...)   the connector phrase ('as in', 'as', 'for', 'like')
      \\s+[a-z]+      the example word we want to throw away
    The replacement r'\\1' puts back only the captured letter."""
    return re.sub(r"\b([a-z])\s+(?:as in|as|for|like)\s+[a-z]+\b", r"\1", text)


def normalize_email(raw):
    """Turn a spoken email like 'george dot smith at gmail dot com'
    into 'george.smith@gmail.com'. Safe to run on an already-clean
    email too — it just lowercases and returns it unchanged."""
    email = raw.strip().lower()
    # Resolve any 'B as in Bravo' spellings to plain letters first, so the
    # example words don't survive into the address.
    email = strip_phonetics(email)
    # Replace spoken words with their symbols. \b is a "word boundary",
    # so \bat\b matches the standalone word "at" but NOT the "at" inside
    # "nathan". The \s* around it eats any spaces on either side.
    spoken = {
        r"\s*\bat\b\s*": "@",
        r"\s*\bdot\b\s*": ".",
        r"\s*\bunderscore\b\s*": "_",
        r"\s*\b(?:dash|hyphen)\b\s*": "-",
        r"\s*\bplus\b\s*": "+",
    }
    for pattern, symbol in spoken.items():
        email = re.sub(pattern, symbol, email)
    # Drop any leftover spaces, e.g. "g mail" -> "gmail".
    return email.replace(" ", "")


async def book_appointment(name, date, time, email):
    try:
        dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %I:%M %p")
    except ValueError:
        dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
    dt = dt.replace(tzinfo=ZoneInfo("America/Toronto"))
    start_iso = dt.isoformat()

    email = normalize_email(email)

    url = "https://api.cal.com/v2/bookings"
    headers = {
        "Authorization": f"Bearer {CALCOM_API_KEY}",
        "cal-api-version": "2024-08-13",
        "Content-Type": "application/json",
    }
    payload = {
        "start": start_iso,
        "eventTypeId": int(CALCOM_EVENT_TYPE_ID),
        "attendee": {
            "name": name,
            "email": email,
            "timeZone": "America/Toronto",
        },
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=payload)

    if resp.status_code in (200, 201):
        print(f">>> BOOKED on Cal.com: {name} at {start_iso}")
        return {"status": "confirmed", "name": name, "date": date, "time": time}
    else:
        print(f">>> Cal.com booking FAILED ({resp.status_code}): {resp.text}")
        return {"status": "failed", "error": resp.text}


@app.post("/twilio/inbound")
async def handle_inbound_call(request: Request):
    host = request.headers.get("host")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Hi! Thanks for calling. How can I help you today?</Say>
    <Connect><Stream url="wss://{host}/twilio/stream" /></Connect>
</Response>"""
    return Response(content=twiml, media_type="text/xml")


@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    await websocket.accept()
    print(">>> Twilio connected")

    stream_sid = None
    history = [{"role": "system", "content": system_prompt()}]
    speaking = asyncio.Event()
    interrupt = asyncio.Event()
    speak_task = None

    dg_url = ("wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000"
              "&channels=1&model=nova-3&punctuate=true&interim_results=true&endpointing=600")
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

    async with websockets.connect(dg_url, additional_headers=headers) as dg:
        print(">>> Connected to Deepgram")

        async def speak(text):
            speaking.set()
            interrupt.clear()
            try:
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream?output_format=ulaw_8000"
                h = {"xi-api-key": ELEVENLABS_API_KEY}
                body = {"text": text, "model_id": "eleven_turbo_v2"}
                async with httpx.AsyncClient() as client:
                    async with client.stream("POST", url, headers=h, json=body) as resp:
                        async for chunk in resp.aiter_bytes():
                            if interrupt.is_set():
                                break
                            if chunk:
                                await websocket.send_text(json.dumps({
                                    "event": "media", "streamSid": stream_sid,
                                    "media": {"payload": base64.b64encode(chunk).decode()}}))
                                await asyncio.sleep(0.02)
            except Exception as e:
                print(f"speak error: {e}")
            finally:
                speaking.clear()

        async def respond(text):
            nonlocal speak_task
            history.append({"role": "user", "content": text})
            r = await openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=history, tools=TOOLS)
            msg = r.choices[0].message

            if msg.tool_calls:
                history.append(msg)
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result = await book_appointment(**args)
                    history.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": json.dumps(result)})
                r2 = await openai_client.chat.completions.create(
                    model="gpt-4o-mini", messages=history)
                reply = r2.choices[0].message.content
                history.append({"role": "assistant", "content": reply})
            else:
                reply = msg.content
                history.append({"role": "assistant", "content": reply})

            print(f"AI: {reply}")
            speak_task = asyncio.create_task(speak(reply))

        async def barge_in():
            interrupt.set()
            await websocket.send_text(json.dumps({
                "event": "clear", "streamSid": stream_sid}))
            print(">>> interrupted")

        async def twilio_to_dg():
            nonlocal stream_sid
            try:
                while True:
                    data = json.loads(await websocket.receive_text())
                    ev = data.get("event")
                    if ev == "start":
                        stream_sid = data["start"]["streamSid"]
                    elif ev == "media":
                        await dg.send(base64.b64decode(data["media"]["payload"]))
                    elif ev == "stop":
                        await dg.send(json.dumps({"type": "CloseStream"}))
                        break
            except Exception as e:
                print(f"twilio_to_dg error: {e}")

        async def dg_to_ai():
            try:
                async for msg in dg:
                    r = json.loads(msg)
                    if r.get("type") != "Results":
                        continue
                    alt = r["channel"]["alternatives"][0]
                    text = alt["transcript"]
                    is_final = r.get("is_final", False)
                    if speaking.is_set() and len(text.strip()) >= 6 and not interrupt.is_set():
                        await barge_in()
                    if is_final and text.strip():
                        print(f"CALLER: {text}")
                        await respond(text)
            except Exception as e:
                import traceback
                print(f"dg_to_ai error: {e}")
                traceback.print_exc()

        await asyncio.gather(twilio_to_dg(), dg_to_ai())