"""The `send_message` capability — forward a visitor's message to the site owner
by email, via the portfolio's existing **Formspree** endpoint.

Kept in its own stdlib-only module (like `rag.search_site_text` for search) so
both transports return byte-identical results: the real FastMCP contact server
(mcp_server/contact_server.py) and the in-process contact client
(mcp_client.InProcessContactMCPClient) import this one function. No third-party
deps — a contact form shouldn't pull in the whole RAG stack.

Formspree emails the owner whatever we POST; its free tier is ~50 messages/mo,
so the guardrails here (real-looking email + non-empty message required) matter:
the agent must only reach for this when a visitor genuinely wants to get in touch.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

# Deliberately loose: reject obvious garbage/empties, don't police valid-but-odd
# addresses. The point is "did the visitor actually give an email", not RFC 5322.
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def send_contact_message(message: str, from_email: str, name: str = "",
                         timeout: float = 10.0) -> str:
    """POST a visitor's message to FORMSPREE_ENDPOINT so Formspree emails the
    owner. Returns a short human-readable status the agent can relay verbatim.

    Never raises: every failure path returns a "couldn't send" string, because
    this runs inside the agent loop and a raised exception would abort the turn."""
    endpoint = os.environ.get("FORMSPREE_ENDPOINT", "").strip()
    if not endpoint:
        return ("Sorry — the contact channel isn't configured, so I can't send a "
                "message right now.")

    message = (message or "").strip()
    from_email = (from_email or "").strip()
    if not message:
        return "I need a message to send. What would you like to tell Aditya?"
    if not _EMAIL.match(from_email):
        return ("I need a valid email address to send this so Aditya can reply. "
                "What's the best email to reach you at?")

    payload = json.dumps({
        "email": from_email,
        "name": (name or "").strip(),
        "message": message,
        "_subject": f"Portfolio chatbot: message from {name.strip() or from_email}",
    }).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=payload, method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if 200 <= resp.status < 300:
                return (f"Sent — Aditya will get your message and can reply to "
                        f"{from_email}. Thanks for reaching out!")
            return (f"Sorry, the message didn't go through (status {resp.status}). "
                    "Please try emailing Aditya directly.")
    except urllib.error.HTTPError as e:
        return (f"Sorry, the message didn't go through (status {e.code}). "
                "Please try emailing Aditya directly.")
    except Exception:
        return ("Sorry, I couldn't reach the contact service just now. "
                "Please try emailing Aditya directly.")
