from ui.app import normalize_api_url


def test_normalize_api_url_leaves_full_url_unchanged():
    assert normalize_api_url("http://localhost:8000") == "http://localhost:8000"
    assert normalize_api_url("https://api.example.com") == "https://api.example.com"


def test_normalize_api_url_adds_https_to_bare_hostname():
    assert normalize_api_url("rag-api.onrender.com") == "https://rag-api.onrender.com"
