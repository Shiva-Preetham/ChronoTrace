"""Web demo: preloaded cases load with no API key, and uploads fail cleanly without one."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from chronotrace.web.app import create_app  # noqa: E402
from chronotrace.web.demo import DEMO_DIR  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    return TestClient(create_app(DEMO_DIR, tmp_path / "web", tmp_path / "cache"))


def test_cases_load_without_api(client):
    cases = client.get("/api/cases").json()
    assert [c["id"] for c in cases] == ["genuine", "swap", "fault"]
    expected = {"genuine": "GENUINE_RETURN", "swap": "SWAP_DETECTED", "fault": "PACKING_NOT_PROVEN"}
    for c in cases:
        data = client.get(f"/api/cases/{c['id']}").json()
        assert data["report"]["status"] == expected[c["id"]]
        for key in ("packing_video", "return_video", "evidence_card", "report"):
            r = client.get(data["assets"][key])
            assert r.status_code == 200 and len(r.content) > 100, key
        assert data["report"]["videos"]["packing"]["seconds"] > 0


def test_index_and_unknown_case(client):
    assert "ChronoTrace" in client.get("/").text
    assert client.get("/api/cases/nope").status_code == 404
    assert client.get("/api/cases/genuine.json").status_code == 404


def test_verify_without_key_is_503(client):
    files = {"packing": ("a.mp4", b"x"), "returned": ("b.mp4", b"x")}
    r = client.post("/api/verify", files=files, data={"order_id": "FK-1"})
    assert r.status_code == 503 and "MODEL_API_KEY" in r.json()["detail"]
