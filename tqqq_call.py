"""
Ask an OpenAI model to make a discretionary BUY/SELL call on TQQQ,
using live web search, log every call to a CSV (with SPY as a benchmark),
and email you the call.

Designed to run unattended on GitHub Actions, but works locally too.

Environment variables:
    OPENAI_API_KEY   required
    SMTP_USER        optional - email account that sends the alert (e.g. a Gmail address)
    SMTP_PASSWORD    optional - app password for that account; no email if unset
    EMAIL_TO         optional - where to send it (defaults to SMTP_USER)
    SMTP_HOST        optional - defaults to smtp.gmail.com (port 465, SSL)
    POSITION         optional - force the current position (IN/OUT); otherwise it is
                     inferred from the last call in the log (BUY -> IN, SELL -> OUT)
    START_POSITION   optional - position to assume on the very first run (default OUT)
"""

import csv
import json
import os
import smtplib
from email.message import EmailMessage
from datetime import datetime, timezone

import yfinance as yf
from openai import OpenAI

MODEL = "gpt-6-astra"          # swap for whichever current model you want
LOG_FILE = "tqqq_calls.csv"
BENCHMARK = "SPY"        # S&P 500 ETF; use "^GSPC" for the raw index level

SYSTEM_PROMPT = """You are a discretionary trader managing a position in TQQQ \
(3x leveraged Nasdaq-100 ETF). Your job is to make one call: BUY or SELL.

Before deciding, use web search to check current conditions as of today, including:
- Recent price action and trend in QQQ/TQQQ
- Volatility (VIX level and direction)
- Upcoming macro events in the next 1-2 weeks (Fed, CPI, jobs data, major earnings)
- Any significant market-moving news

Weigh these however you judge best. TQQQ suffers volatility decay, so sideways or \
choppy markets are bad for it, not neutral. Make a decisive call.

Timing: this decision is made just after the US market close, using today's \
closing data. The order will be placed from Australia during US after-hours or \
queued for the next US open, so judge for the next session, not intraday moves."""

SCHEMA = {
    "type": "object",
    "properties": {
        "call": {"type": "string", "enum": ["BUY", "SELL"]},
        "confidence": {"type": "integer", "minimum": 1, "maximum": 10},
        "time_horizon": {"type": "string"},
        "reasoning": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "what_would_change_my_mind": {"type": "string"},
    },
    "required": ["call", "confidence", "time_horizon", "reasoning",
                 "what_would_change_my_mind"],
    "additionalProperties": False,
}

HEADER = ["timestamp", "position_before", "call", "action", "confidence",
          "time_horizon", "tqqq_price", "spy_price", "reasoning", "change_mind"]


def read_log():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE) as f:
        return list(csv.DictReader(f))


def current_position(rows):
    forced = os.environ.get("POSITION", "").strip().upper()
    if forced in ("IN", "OUT"):
        return forced
    if rows:
        return "IN" if rows[-1]["call"] == "BUY" else "OUT"
    return os.environ.get("START_POSITION", "OUT").strip().upper() or "OUT"


def recent_calls(rows, n=5):
    if not rows:
        return "No previous calls."
    return "\n".join(
        f"{r['timestamp']}: {r['call']} (conf {r['confidence']}) @ ${r['tqqq_price']}"
        for r in rows[-n:]
    )


def latest_price(ticker):
    return round(float(yf.Ticker(ticker).history(period="5d")["Close"].iloc[-1]), 2)


def make_call(position, rows):
    client = OpenAI()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    user_msg = (
        f"Today is {today}.\n"
        f"My current position: {position}\n\n"
        f"Your previous calls:\n{recent_calls(rows)}"
    )
    resp = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=user_msg,
        tools=[{"type": "web_search"}],
        text={"format": {"type": "json_schema", "name": "tqqq_call",
                         "schema": SCHEMA, "strict": True}},
    )
    return json.loads(resp.output_text)


def action_for(call, position):
    """Translate the call into what you actually need to do in Webull."""
    if call == "BUY":
        return "BUY NOW" if position == "OUT" else "HOLD (already in)"
    return "SELL NOW" if position == "IN" else "STAY OUT (already out)"


def log_call(result, position, action, price, spy):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        w.writerow([
            datetime.now(timezone.utc).isoformat(timespec="minutes"), position,
            result["call"], action, result["confidence"], result["time_horizon"],
            price, spy, " | ".join(result["reasoning"]),
            result["what_would_change_my_mind"],
        ])


def notify(title, body, urgent):
    """Email the call. Skips quietly if email secrets aren't set."""
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    to = os.environ.get("EMAIL_TO") or user
    if not (user and password):
        return
    msg = EmailMessage()
    msg["Subject"] = ("🔔 " if urgent else "") + title
    msg["From"] = user
    msg["To"] = to
    msg.set_content(body)
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    with smtplib.SMTP_SSL(host, 465, timeout=30) as s:
        s.login(user, password)
        s.send_message(msg)


if __name__ == "__main__":
    rows = read_log()
    position = current_position(rows)

    price = latest_price("TQQQ")
    spy = latest_price(BENCHMARK)
    result = make_call(position, rows)
    action = action_for(result["call"], position)
    log_call(result, position, action, price, spy)

    body = (
        f"TQQQ ${price} | SPY ${spy}\n"
        f"Confidence {result['confidence']}/10, horizon {result['time_horizon']}\n"
        + "\n".join(f"- {r}" for r in result["reasoning"])
        + f"\nWould change mind if: {result['what_would_change_my_mind']}"
    )
    print(f"{action}\n{body}")
    notify(f"TQQQ: {action}", body, urgent=action in ("BUY NOW", "SELL NOW"))
