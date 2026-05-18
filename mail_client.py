#!/usr/bin/env python3
"""
mail_client.py — Minimal Maildir-based IMAP mail client for CLI / AI agent use.

Designed to work with isync/mbsync Maildir and any SMTP server.
Built to be called from scripts and AI agents as a structured mail interface.

Usage:
    python mail_client.py new                          # list unread
    python mail_client.py list [N]                     # list last N messages (default 20)
    python mail_client.py read <UID>                   # read message by UID
    python mail_client.py search <keyword>             # search subject/from/body
    python mail_client.py send <to> <subject> <file>   # send email (body from file)
    python mail_client.py reply <UID> <file>           # reply to message
    python mail_client.py sync                         # run mbsync to fetch new mail

Configuration (environment variables):
    MAIL_FROM       sender email address
    MAIL_SMTP_HOST  SMTP server hostname
    MAIL_SMTP_PORT  SMTP port (default: 587)
    MAIL_SMTP_PASS  SMTP password
    MAIL_MAILDIR    path to Maildir INBOX folder
    MAIL_MBSYNC     mbsync profile name (default: mailuk-main)
"""

import sys
import os
import email
import smtplib
import ssl
import subprocess
from email import policy
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime

# ── Config (from environment variables) ───────────────────────────────────────
MAILDIR   = Path(os.environ.get("MAIL_MAILDIR", str(Path.home() / "Mail/INBOX")))
FROM_ADDR = os.environ.get("MAIL_FROM", "")
SMTP_HOST = os.environ.get("MAIL_SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("MAIL_SMTP_PORT", "587"))
SMTP_PASS = os.environ.get("MAIL_SMTP_PASS", "")
MBSYNC_PROFILE = os.environ.get("MAIL_MBSYNC", "main")
DISPLAY   = 80


def load_messages(folder="cur"):
    """Load all messages from a Maildir folder, sorted newest-first."""
    d = MAILDIR / folder
    if not d.exists():
        return []
    msgs = []
    for f in sorted(d.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if f.is_file():
            try:
                raw = f.read_bytes()
                msg = email.message_from_bytes(raw, policy=policy.default)
                uid = f.name.split(",U=")[1].split(":")[0] if ",U=" in f.name else f.name
                msgs.append({"uid": uid, "path": f, "msg": msg, "raw": raw})
            except Exception:
                pass
    return msgs


def fmt_date(d):
    if not d:
        return "?"
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(d)
        return dt.strftime("%d %b %Y %H:%M")
    except Exception:
        return str(d)[:16]


def get_body(msg):
    body = ""
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            try:
                body = part.get_content()
                break
            except Exception:
                pass
    return body.strip()


def print_header():
    print(f"{'UID':<6} {'DATE':<18} {'FROM':<30} {'SUBJECT'}")
    print("-" * DISPLAY)


def print_msg_line(m):
    msg  = m["msg"]
    uid  = m["uid"][:5]
    date = fmt_date(msg.get("Date", ""))
    frm  = str(msg.get("From", ""))[:28]
    subj = str(msg.get("Subject", "(no subject)"))[:35]
    print(f"{uid:<6} {date:<18} {frm:<30} {subj}")


def cmd_new():
    msgs = load_messages("new")
    if not msgs:
        print("No new messages.")
        return
    print(f"{len(msgs)} new message(s):\n")
    print_header()
    for m in msgs:
        print_msg_line(m)


def cmd_list(n=20):
    msgs = load_messages("cur") + load_messages("new")
    msgs = msgs[:int(n)]
    if not msgs:
        print("Mailbox empty.")
        return
    print(f"Last {len(msgs)} messages:\n")
    print_header()
    for m in msgs:
        print_msg_line(m)


def cmd_read(uid):
    all_msgs = load_messages("cur") + load_messages("new")
    found = [m for m in all_msgs if m["uid"] == str(uid)]
    if not found:
        print(f"Message U={uid} not found.")
        return
    m   = found[0]
    msg = m["msg"]
    print(f"\n{'='*DISPLAY}")
    print(f"From:    {msg.get('From','')}")
    print(f"To:      {msg.get('To','')}")
    print(f"Date:    {fmt_date(msg.get('Date',''))}")
    print(f"Subject: {msg.get('Subject','')}")
    print(f"{'='*DISPLAY}\n")
    body = get_body(msg)
    print(body if body else "(no plain text body)")
    print(f"\n{'='*DISPLAY}")
    print(f"[Reply: python mail_client.py reply {uid} body.txt]")


def cmd_search(keyword):
    kw = keyword.lower()
    all_msgs = load_messages("cur") + load_messages("new")
    hits = []
    for m in all_msgs:
        msg  = m["msg"]
        subj = str(msg.get("Subject", "")).lower()
        frm  = str(msg.get("From", "")).lower()
        body = get_body(msg).lower()
        if kw in subj or kw in frm or kw in body:
            hits.append(m)
    if not hits:
        print(f"No messages matching '{keyword}'.")
        return
    print(f"{len(hits)} match(es) for '{keyword}':\n")
    print_header()
    for m in hits:
        print_msg_line(m)


def cmd_send(to, subject, body_file):
    body = Path(body_file).read_text(encoding="utf-8") if body_file != "-" else sys.stdin.read()
    _send_email(to, subject, body)
    print(f"Sent to {to}")


def cmd_reply(uid, body_file):
    all_msgs = load_messages("cur") + load_messages("new")
    found = [m for m in all_msgs if m["uid"] == str(uid)]
    if not found:
        print(f"Message U={uid} not found.")
        return
    orig    = found[0]["msg"]
    to      = str(orig.get("Reply-To") or orig.get("From", ""))
    subject = str(orig.get("Subject", ""))
    if not subject.lower().startswith("re:"):
        subject = "Re: " + subject
    msg_id  = orig.get("Message-ID", "")

    body = Path(body_file).read_text(encoding="utf-8") if body_file != "-" else sys.stdin.read()

    orig_body = get_body(orig)
    quoted    = "\n".join(f"> {l}" for l in orig_body.splitlines())
    full_body = body + f"\n\n---\n{quoted}"

    _send_email(to, subject, full_body, in_reply_to=msg_id)
    print(f"Reply sent to {to}")


def cmd_sync():
    print(f"Syncing ({MBSYNC_PROFILE})...")
    result = subprocess.run(["mbsync", MBSYNC_PROFILE], capture_output=True, text=True)
    if result.returncode == 0:
        print("Sync complete.")
    else:
        print(f"mbsync returned {result.returncode}")
        if result.stderr:
            print(result.stderr[:500])


def _send_email(to, subject, body, in_reply_to=None):
    if not FROM_ADDR or not SMTP_HOST or not SMTP_PASS:
        raise RuntimeError(
            "Set MAIL_FROM, MAIL_SMTP_HOST, and MAIL_SMTP_PASS environment variables."
        )
    mime = MIMEMultipart("alternative")
    mime["From"]    = FROM_ADDR
    mime["To"]      = to
    mime["Subject"] = subject
    mime["Date"]    = email.utils.formatdate(localtime=True)
    if in_reply_to:
        mime["In-Reply-To"] = in_reply_to
        mime["References"]  = in_reply_to
    mime.attach(MIMEText(body, "plain", "utf-8"))

    ctx = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as s:
        s.ehlo()
        s.starttls(context=ctx)
        s.login(FROM_ADDR, SMTP_PASS)
        s.sendmail(FROM_ADDR, [to], mime.as_string())


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__)
        sys.exit(0)

    cmd = args[0]
    try:
        if cmd == "new":
            cmd_new()
        elif cmd == "list":
            cmd_list(args[1] if len(args) > 1 else 20)
        elif cmd == "read":
            cmd_read(args[1])
        elif cmd == "search":
            cmd_search(" ".join(args[1:]))
        elif cmd == "send":
            cmd_send(args[1], args[2], args[3] if len(args) > 3 else "-")
        elif cmd == "reply":
            cmd_reply(args[1], args[2] if len(args) > 2 else "-")
        elif cmd == "sync":
            cmd_sync()
        else:
            print(f"Unknown command: {cmd}\nRun with --help for usage.")
    except KeyboardInterrupt:
        print("\nAborted.")
    except Exception as e:
        print(f"Error: {e}")
        raise
