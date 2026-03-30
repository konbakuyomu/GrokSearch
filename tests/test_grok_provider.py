import pytest

from smart_search.providers.grok import GrokSearchProvider


class DummyResponse:
    """模拟 httpx.Response 用于测试 completion 解析"""

    def __init__(self, text="", json_data=None, json_error=None):
        self.text = text
        self._json_data = json_data
        self._json_error = json_error

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._json_data


@pytest.mark.asyncio
async def test_search_uses_non_stream_completion_and_headers(monkeypatch):
    """验证 search() 使用非流式 completion + 自定义 headers"""
    provider = GrokSearchProvider("https://api.example.com", "test-key", "test-model")
    captured = {}

    async def fake_execute(headers, payload, ctx):
        captured["headers"] = headers
        captured["payload"] = payload
        return "ok"

    monkeypatch.setattr(provider, "_execute_completion_with_retry", fake_execute)

    result = await provider.search("What is Scrape.do?")

    assert result == "ok"
    assert "User-Agent" in captured["headers"]
    assert captured["headers"]["Accept"] == "application/json, text/event-stream"
    assert captured["payload"]["stream"] is False


@pytest.mark.asyncio
async def test_fetch_uses_non_stream(monkeypatch):
    """验证 fetch() 使用非流式 completion"""
    provider = GrokSearchProvider("https://api.example.com", "test-key", "test-model")
    captured = {}

    async def fake_execute(headers, payload, ctx):
        captured["payload"] = payload
        return "fetched content"

    monkeypatch.setattr(provider, "_execute_completion_with_retry", fake_execute)

    result = await provider.fetch("https://example.com")

    assert result == "fetched content"
    assert captured["payload"]["stream"] is False


@pytest.mark.asyncio
async def test_describe_url_uses_non_stream(monkeypatch):
    """验证 describe_url() 使用非流式 completion"""
    provider = GrokSearchProvider("https://api.example.com", "test-key", "test-model")
    captured = {}

    async def fake_execute(headers, payload, ctx):
        captured["payload"] = payload
        return "Title: Example\nExtracts: Some text"

    monkeypatch.setattr(provider, "_execute_completion_with_retry", fake_execute)

    result = await provider.describe_url("https://example.com")

    assert result["title"] == "Example"
    assert captured["payload"]["stream"] is False


@pytest.mark.asyncio
async def test_rank_sources_uses_non_stream(monkeypatch):
    """验证 rank_sources() 使用非流式 completion"""
    provider = GrokSearchProvider("https://api.example.com", "test-key", "test-model")
    captured = {}

    async def fake_execute(headers, payload, ctx):
        captured["payload"] = payload
        return "2 1 3"

    monkeypatch.setattr(provider, "_execute_completion_with_retry", fake_execute)

    result = await provider.rank_sources("test query", "sources...", 3)

    assert result == [2, 1, 3]
    assert captured["payload"]["stream"] is False


@pytest.mark.asyncio
async def test_parse_completion_response_reads_json():
    """验证 JSON completion 响应正常解析"""
    provider = GrokSearchProvider("https://api.example.com", "test-key", "test-model")
    response = DummyResponse(
        text='{"choices":[{"message":{"content":"hello world"}}]}',
        json_data={"choices": [{"message": {"content": "hello world"}}]},
    )

    result = await provider._parse_completion_response(response)

    assert result == "hello world"


@pytest.mark.asyncio
async def test_parse_completion_response_falls_back_to_sse():
    """验证 JSON 解析失败时 fallback 到 SSE 文本解析"""
    provider = GrokSearchProvider("https://api.example.com", "test-key", "test-model")
    response = DummyResponse(
        text=(
            'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
            'data: [DONE]\n'
        ),
        json_error=ValueError("not json"),
    )

    result = await provider._parse_completion_response(response)

    assert result == "hello world"


@pytest.mark.asyncio
async def test_parse_completion_response_empty_choices():
    """验证空 choices 返回空字符串"""
    provider = GrokSearchProvider("https://api.example.com", "test-key", "test-model")
    response = DummyResponse(
        text='{"choices":[]}',
        json_data={"choices": []},
    )

    result = await provider._parse_completion_response(response)

    assert result == ""


@pytest.mark.asyncio
async def test_parse_completion_response_null_content():
    """验证 content 为 null 时返回空字符串"""
    provider = GrokSearchProvider("https://api.example.com", "test-key", "test-model")
    response = DummyResponse(
        text='{"choices":[{"message":{"content":null}}]}',
        json_data={"choices": [{"message": {"content": None}}]},
    )

    result = await provider._parse_completion_response(response)

    assert result == ""


@pytest.mark.asyncio
async def test_build_api_headers():
    """验证 headers 包含 Accept 和 User-Agent"""
    provider = GrokSearchProvider("https://api.example.com", "test-key", "test-model")
    headers = provider._build_api_headers()

    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"
    assert "text/event-stream" in headers["Accept"]
    assert headers["User-Agent"].startswith("smart-search-mcp/")
