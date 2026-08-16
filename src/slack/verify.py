"""Slack request signature verification (HMAC SHA-256).

Spec: https://api.slack.com/authentication/verifying-requests-from-slack
"""
import hashlib
import hmac
import time

from src.config import settings


def verify(body: bytes, signature: str, timestamp: str) -> bool:
    """Return True if the signature matches; False otherwise.

    Also rejects requests older than 5 minutes (replay attack mitigation).
    """
    if not signature or not timestamp:
        return False

    try:
        ts = int(timestamp)
    except (ValueError, TypeError):
        return False

    if abs(time.time() - ts) > 60 * 5:
        return False

    basestring = f"v0:{timestamp}:{body.decode()}".encode()
    expected = "v0=" + hmac.new(
        settings.slack_signing_secret.encode(),
        basestring,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, signature)
