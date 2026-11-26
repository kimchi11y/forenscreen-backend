"""Verify Neon Auth JWT verification and actor attribution.

Checks that the API trusts a signature rather than a claim: an anonymous call
stays anonymous, a valid token is accepted and attributed, and a tampered token
is rejected outright.

Needs a test account. Create one by signing up in the app, then pass it here:

    python scripts/verify_auth.py <email> <password>

or set FORENSCREEN_TEST_EMAIL / FORENSCREEN_TEST_PASSWORD.
"""

import io
import os
import random
import sys
import time

import httpx
from PIL import Image, ImageDraw
from dotenv import dotenv_values

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = os.environ.get("FORENSCREEN_API", "http://127.0.0.1:8000")

failures = []


def check(label, condition, detail=""):
    failures.append(condition)
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


def get_token(auth_base, email, password):
    """Sign in and exchange the session for a JWT."""
    origin = "http://localhost:5173"
    headers = {"Origin": origin, "Content-Type": "application/json"}
    with httpx.Client(timeout=60) as client:
        response = client.post(
            f"{auth_base}/sign-in/email",
            json={"email": email, "password": password},
            headers=headers,
        )
        response.raise_for_status()
        pair = response.headers.get("set-cookie", "").split(";")[0]
        token_response = client.get(
            f"{auth_base}/token", headers={"Origin": origin, "Cookie": pair}
        )
        token_response.raise_for_status()
        return token_response.json()["token"]


def make_image():
    base = Image.open(
        sorted(
            __import__("glob").glob(os.path.join(BACKEND_DIR, "uploads", "*.jpeg"))
        )[0]
    ).convert("RGB")
    width, height = base.size
    canvas = base.copy()
    draw = ImageDraw.Draw(canvas)
    random.seed()
    for _ in range(5):
        x = random.randint(0, max(1, width - 90))
        y = random.randint(0, max(1, height - 90))
        draw.rectangle(
            [x, y, x + random.randint(25, 80), y + random.randint(25, 80)],
            fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
        )
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def main():
    env = dotenv_values(os.path.join(BACKEND_DIR, ".env"))
    auth_base = env.get("NEON_AUTH_BASE_URL")
    email = os.environ.get("FORENSCREEN_TEST_EMAIL") or (sys.argv[1] if len(sys.argv) > 1 else None)
    password = os.environ.get("FORENSCREEN_TEST_PASSWORD") or (sys.argv[2] if len(sys.argv) > 2 else None)

    if not auth_base or not email or not password:
        print("Need NEON_AUTH_BASE_URL plus a test email and password.")
        print("  python scripts/verify_auth.py <email> <password>")
        return 2

    print(f"auth base: {auth_base}")

    print("\n1. Anonymous call stays anonymous")
    response = httpx.get(f"{API}/me", timeout=60)
    body = response.json()
    check("HTTP 200", response.status_code == 200, str(response.status_code))
    check("reports unauthenticated", body.get("authenticated") is False, str(body))
    check("reports auth is configured server-side", body.get("auth_configured") is True)

    print("\n2. A valid token is accepted")
    token = get_token(auth_base, email, password)
    response = httpx.get(f"{API}/me", headers={"Authorization": f"Bearer {token}"}, timeout=60)
    body = response.json()
    check("HTTP 200", response.status_code == 200, f"HTTP {response.status_code} {response.text[:120]}")
    check("reports authenticated", body.get("authenticated") is True, str(body)[:160])
    check("returns the right email", body.get("email") == email, str(body.get("email")))
    user_id = body.get("id")
    check("returns a user id", bool(user_id), str(user_id))
    print(f"     actor={body.get('actor')}  sub={user_id}")

    print("\n3. A tampered token is rejected, not downgraded to anonymous")
    parts = token.split(".")
    forged = f"{parts[0]}.{parts[1]}.{'A' * len(parts[2])}"
    response = httpx.get(f"{API}/me", headers={"Authorization": f"Bearer {forged}"}, timeout=60)
    check("tampered signature -> 401", response.status_code == 401, f"HTTP {response.status_code}")
    response = httpx.get(f"{API}/me", headers={"Authorization": "Bearer not-a-jwt"}, timeout=60)
    check("garbage token -> 401", response.status_code == 401, f"HTTP {response.status_code}")

    print("\n4. Registration with a token attributes the real actor")
    response = httpx.post(
        f"{API}/upload-asset/",
        files={"file": ("auth-check.png", make_image(), "image/png")},
        headers={"Authorization": f"Bearer {token}"},
        timeout=180,
    )
    response.raise_for_status()
    asset_id = response.json()["id"]
    check("registered", bool(asset_id), asset_id[:8] if asset_id else "")

    entries = httpx.get(f"{API}/custody/asset/{asset_id}", timeout=120).json()["entries"]
    registered = next((e for e in entries if e["event_type"] == "ASSET_REGISTERED"), None)
    check("custody entry exists", registered is not None)
    if registered:
        check(
            "actor is the verified user, not 'unauthenticated'",
            registered["actor"] == f"user:{user_id}",
            registered["actor"],
        )
        check(
            "payload records the email",
            registered["payload"].get("actor_email") == email,
            str(registered["payload"].get("actor_email")),
        )

    print("\n5. Anonymous registration is still allowed, and says so")
    response = httpx.post(
        f"{API}/upload-asset/",
        files={"file": ("anon-check.png", make_image(), "image/png")},
        timeout=180,
    )
    response.raise_for_status()
    anon_id = response.json()["id"]
    entries = httpx.get(f"{API}/custody/asset/{anon_id}", timeout=120).json()["entries"]
    anon = next((e for e in entries if e["event_type"] == "ASSET_REGISTERED"), None)
    check("anonymous actor recorded honestly", anon and anon["actor"] == "unauthenticated",
          anon["actor"] if anon else "no entry")

    print("\n6. Anchor events are attributed to the system")
    # Anchoring runs as a background task after the response, and a Sepolia
    # block takes ~12s, so the entry legitimately is not there yet.
    anchor_entries = []
    for _ in range(20):
        entries = httpx.get(f"{API}/custody/asset/{anon_id}", timeout=120).json()["entries"]
        anchor_entries = [e for e in entries if e["event_type"].startswith("ANCHOR")]
        if anchor_entries:
            break
        time.sleep(6)
    check(
        "anchor events exist and are attributed to 'system'",
        bool(anchor_entries) and all(e["actor"] == "system" for e in anchor_entries),
        str([(e["event_type"], e["actor"]) for e in anchor_entries]),
    )

    print("\n7. The chain still verifies after all of that")
    report = httpx.get(f"{API}/custody/asset/{asset_id}/verify", timeout=60).json()
    check("chain intact", report["intact"] is True, str(report.get("problems")))

    print("\n" + ("ALL PASS" if all(failures) else f"{failures.count(False)} CHECK(S) FAILED"))
    return 0 if all(failures) else 1


if __name__ == "__main__":
    raise SystemExit(main())
