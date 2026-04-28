import pytest
import requests
import time

API_BASE = "http://127.0.0.1:8000"

def test_health_endpoint():
    """Test that the health endpoint returns correct response"""
    response = requests.get(f"{API_BASE}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["chunks_indexed"] > 0
    print(f"✅ Health check passed! Chunks: {data['chunks_indexed']}")

def test_root_endpoint():
    """Test that the root endpoint returns correct response"""
    response = requests.get(f"{API_BASE}/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    print(f"✅ Root endpoint passed!")

def test_query_endpoint():
    """Test that the query endpoint returns correct response"""
    response = requests.post(
        f"{API_BASE}/query",
        json={
            "question": "What are the liability rules for business entities?",
            "top_k": 3
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "sources" in data
    assert len(data["sources"]) == 3
    assert "response_time" in data
    print(f"✅ Query endpoint passed! Response time: {data['response_time']}s")

def test_query_response_time():
    """Test that response time is under 10 seconds"""
    start = time.time()
    response = requests.post(
        f"{API_BASE}/query",
        json={"question": "What is gross negligence?", "top_k": 3}
    )
    elapsed = time.time() - start
    assert elapsed < 10, f"Response too slow: {elapsed:.2f}s"
    print(f"✅ Response time test passed! {elapsed:.2f}s")

def test_empty_question():
    """Test that empty question returns error"""
    response = requests.post(
        f"{API_BASE}/query",
        json={"question": "", "top_k": 3}
    )
    assert response.status_code == 400
    print(f"✅ Empty question validation passed!")

if __name__ == "__main__":
    print("🧪 Running API tests...")
    test_health_endpoint()
    test_root_endpoint()
    test_query_endpoint()
    test_query_response_time()
    test_empty_question()
    print("\n🎉 All tests passed!")