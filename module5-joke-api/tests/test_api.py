import pytest
from fastapi.testclient import TestClient
from src.api import app


# ─────────────────────────────────────────────
# FIXTURE — starts the app properly
# scope="module" means it runs ONCE for all
# tests in this file — not once per test.
# This is important because building the
# LangGraph on every test would be very slow.
# ─────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager
    # triggers FastAPI's lifespan events:
    # → startup: builds the LangGraph
    # → yields the client for tests to use
    # → shutdown: cleans up
    with TestClient(app) as c:
        yield c


# ─────────────────────────────────────────────
# HEALTH CHECK TESTS
# ─────────────────────────────────────────────
def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_correct_fields(client):
    response = client.get("/health")
    body = response.json()
    assert body["status"] == "healthy"
    assert body["agent"] == "joke-bot"
    assert body["version"] == "1.0.0"


def test_root_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


# ─────────────────────────────────────────────
# JOKE ENDPOINT TESTS
# ─────────────────────────────────────────────
def test_joke_returns_200(client):
    response = client.post("/joke", json={"topic": "testing"})
    assert response.status_code == 200


def test_joke_returns_correct_fields(client):
    response = client.post("/joke", json={"topic": "testing"})
    body = response.json()
    assert "joke" in body
    assert "rating" in body
    assert "score" in body
    assert "attempts" in body


def test_joke_is_not_empty(client):
    response = client.post("/joke", json={"topic": "testing"})
    body = response.json()
    assert len(body["joke"]) > 0
    assert len(body["rating"]) > 0


def test_joke_score_is_valid_range(client):
    response = client.post("/joke", json={"topic": "testing"})
    body = response.json()
    assert 1 <= body["score"] <= 10


def test_joke_attempts_is_valid(client):
    response = client.post("/joke", json={"topic": "testing"})
    body = response.json()
    assert 1 <= body["attempts"] <= 3


def test_joke_different_topics(client):
    topics = ["cats", "python", "AWS"]
    for topic in topics:
        response = client.post("/joke", json={"topic": topic})
        assert response.status_code == 200
        assert len(response.json()["joke"]) > 0


# ─────────────────────────────────────────────
# VALIDATION TESTS
# ─────────────────────────────────────────────
def test_empty_topic_rejected(client):
    response = client.post("/joke", json={"topic": ""})
    assert response.status_code == 422


def test_missing_topic_rejected(client):
    response = client.post("/joke", json={})
    assert response.status_code == 422


def test_topic_too_long_rejected(client):
    response = client.post("/joke", json={"topic": "x" * 101})
    assert response.status_code == 422


def test_wrong_content_type_rejected(client):
    response = client.post(
        "/joke",
        data="testing",
        headers={"Content-Type": "text/plain"}
    )
    assert response.status_code in [422, 415]