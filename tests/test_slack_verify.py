"""Test HMAC verification of Slack signatures."""
import hashlib
import hmac
import time

import pytest

from src.config import settings
from src.slack import verify as verify_mod


@pytest.fixture(autouse=True)
def _set_signing_secret(monkeypatch):
    monkeypatch.setattr(settings, "slack_signing_secret", "test-secret-do-not-use")
    yield


def _sign(body: bytes, ts: str, secret: str = "test-secret-do-not-use") -> str:
    basestring = f"v0:{ts}:{body.decode()}".encode()
    return "v0=" + hmac.new(secret.encode(), basestring, hashlib.sha256).hexdigest()


def test_valid_signature_passes():
    body = b"payload=hello"
    ts = str(int(time.time()))
    sig = _sign(body, ts)
    assert verify_mod.verify(body, sig, ts) is True


def test_bad_signature_rejected():
    body = b"payload=hello"
    ts = str(int(time.time()))
    assert verify_mod.verify(body, "v0=deadbeef", ts) is False


def test_wrong_secret_rejected():
    body = b"payload=hello"
    ts = str(int(time.time()))
    bad = _sign(body, ts, secret="wrong-secret")
    assert verify_mod.verify(body, bad, ts) is False


def test_old_timestamp_rejected():
    body = b"payload=hello"
    old_ts = str(int(time.time()) - 60 * 10)
    sig = _sign(body, old_ts)
    assert verify_mod.verify(body, sig, old_ts) is False


def test_missing_signature_rejected():
    assert verify_mod.verify(b"x", "", "12345") is False
    assert verify_mod.verify(b"x", "v0=abc", "") is False
