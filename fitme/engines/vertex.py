"""Vertex AI Virtual Try-On engine."""

from __future__ import annotations

import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import google.auth.transport.requests
from google.oauth2 import service_account

from fitme.config import (
    OUTPUT_VERTEX,
    VERTEX_CREDENTIALS,
    VERTEX_LOCATION,
    VERTEX_MODEL,
    VERTEX_PROJECT,
)


def _endpoint() -> str:
    return (
        f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}/"
        f"publishers/google/models/{VERTEX_MODEL}:predict"
    )


def _access_token() -> str:
    credentials = service_account.Credentials.from_service_account_file(
        str(VERTEX_CREDENTIALS),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


def require_vertex_config() -> None:
    if not VERTEX_PROJECT:
        print("Error: Set GOOGLE_CLOUD_PROJECT in .env")
        sys.exit(1)
    if not VERTEX_CREDENTIALS.exists():
        print(f"Error: Credentials not found: {VERTEX_CREDENTIALS}")
        sys.exit(1)


def test_connection() -> None:
    """Verify service account and project settings."""
    require_vertex_config()
    token = _access_token()
    print(f"Access token obtained: {token[:20]}...")
    print(f"Project: {VERTEX_PROJECT}")
    print(f"Location: {VERTEX_LOCATION}")
    print(f"Endpoint: {_endpoint()}")
    print("Connection OK!")


def run_tryon(
    person_path: str | Path,
    clothing_path: str | Path,
    sample_count: int = 1,
) -> list[Path]:
    """Run Vertex virtual try-on and save PNG outputs."""
    require_vertex_config()
    person_path = Path(person_path)
    clothing_path = Path(clothing_path)

    person_b64 = base64.b64encode(person_path.read_bytes()).decode("utf-8")
    clothing_b64 = base64.b64encode(clothing_path.read_bytes()).decode("utf-8")

    print(f"Person image: {person_path} ({person_path.stat().st_size:,} bytes)")
    print(f"Clothing image: {clothing_path} ({clothing_path.stat().st_size:,} bytes)")

    request_body = {
        "instances": [
            {
                "personImage": {"image": {"bytesBase64Encoded": person_b64}},
                "productImages": [{"image": {"bytesBase64Encoded": clothing_b64}}],
            }
        ],
        "parameters": {"sampleCount": sample_count},
    }

    data = json.dumps(request_body).encode("utf-8")
    req = urllib.request.Request(
        _endpoint(),
        data=data,
        headers={
            "Authorization": f"Bearer {_access_token()}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )

    print(f"Sending request to {VERTEX_MODEL} (sampleCount={sample_count})...")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code} Error: {exc.read().decode('utf-8')}")
        sys.exit(1)

    OUTPUT_VERTEX.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for i, pred in enumerate(result.get("predictions", [])):
        img_b64 = pred.get("bytesBase64Encoded", "")
        if not img_b64:
            continue
        out = OUTPUT_VERTEX / f"{person_path.stem}_{clothing_path.stem}_{i}.png"
        out.write_bytes(base64.b64decode(img_b64))
        print(f"Saved: {out} ({out.stat().st_size:,} bytes)")
        saved.append(out)
    return saved
