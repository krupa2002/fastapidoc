from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_upload_pdf():
    with open("sample.pdf", "rb") as file:
        response = client.post("/upload/", files={"file": ("sample.pdf", file, "application/pdf")})
    assert response.status_code == 200
    assert "filename" in response.json()
    assert "content" in response.json()

def test_query_pdf():
    response = client.post("/query/", json={"user_query": "What is the capital of France?"})
    assert response.status_code == 200
    assert "response" in response.json()
