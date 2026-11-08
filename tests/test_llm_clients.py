import pytest
from helpers import (
    FakeAnthropicSDK,
    FakeOpenAISDK,
    a_response,
    a_text,
    a_tool,
    o_call,
    o_model,
    o_response,
)

from fixpilot.llm import (
    AssistantMsg,
    LLMConfig,
    LLMError,
    ToolCall,
    ToolResultItem,
    ToolResultsMsg,
    ToolSupportError,
    UserMsg,
    check_tool_support,
)
from fixpilot.llm.anthropic_client import AnthropicClient
from fixpilot.llm.fake import FakeLLM, FakeTurn, scripted_policy
from fixpilot.llm.openai_client import OpenAICompatClient

TOOLS = [{"name": "read_file", "description": "Read a file.",
          "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}]
KEY = "sk-test-0123456789abcdefghij"
ANTH = {"FIXPILOT_LLM_PROVIDER": "anthropic", "FIXPILOT_MODEL": "claude-test", "FIXPILOT_LLM_API_KEY": KEY}
OAI = {"FIXPILOT_LLM_PROVIDER": "openai_compat", "FIXPILOT_MODEL": "vendor/m",
       "FIXPILOT_LLM_BASE_URL": "https://example.test/v1", "FIXPILOT_LLM_API_KEY": KEY}


def conversation():
    call = ToolCall("c1", "read_file", {"path": "a.py"})
    return [UserMsg("fix it"), AssistantMsg("looking", [call], None, "x"),
            ToolResultsMsg([ToolResultItem("c1", "read_file", "contents", False)])]


# ------------------------------------------------------------------ Anthropic


def test_anthropic_request_shape_and_key_never_in_it():
    sdk = FakeAnthropicSDK(a_response([a_text("hi")]))
    client = AnthropicClient(LLMConfig.from_env(ANTH), sdk)
    client.generate("SYSTEM", conversation(), TOOLS)
    req = sdk.requests[0]
    assert req["model"] == "claude-test" and req["system"] == "SYSTEM" and req["tools"] == TOOLS
    assert req["messages"][0] == {"role": "user", "content": "fix it"}
    assert req["messages"][1]["content"] == [{"type": "text", "text": "looking"},
                                             {"type": "tool_use", "id": "c1", "name": "read_file",
                                              "input": {"path": "a.py"}}]
    assert req["messages"][2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "c1", "content": "contents"}]}
    assert "cache_control" not in req and "output_config" not in req
    assert KEY not in repr(req)


def test_anthropic_cache_and_effort_only_when_asked():
    sdk = FakeAnthropicSDK(a_response([a_text("hi")]))
    cfg = LLMConfig.from_env({**ANTH, "FIXPILOT_PROMPT_CACHE": "1", "FIXPILOT_EFFORT": "low"})
    AnthropicClient(cfg, sdk).generate("S", [UserMsg("x")], TOOLS)
    assert sdk.requests[0]["cache_control"] == {"type": "ephemeral"}
    assert sdk.requests[0]["output_config"] == {"effort": "low"}


def test_anthropic_replays_the_providers_own_blocks_unchanged():
    sdk = FakeAnthropicSDK(a_response([a_text("hi")]))
    raw = [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": "x"}]
    msgs = [UserMsg("q"), AssistantMsg("x", [], raw, "anthropic")]
    AnthropicClient(LLMConfig.from_env(ANTH), sdk).generate("S", msgs, TOOLS)
    assert sdk.requests[0]["messages"][1]["content"] == raw  # thinking block kept as received


def test_anthropic_response_parsing_and_usage():
    resp = a_response([a_text("plan"), a_tool("t1", "read_file", {"path": "x.py"})], "tool_use",
                      {"in": 100, "out": 20, "read": 50, "write": 10})
    out = AnthropicClient(LLMConfig.from_env(ANTH), FakeAnthropicSDK(resp)).generate("S", [UserMsg("q")], TOOLS)
    assert out.stop == "tool_use" and out.message.text == "plan"
    assert out.message.tool_calls == [ToolCall("t1", "read_file", {"path": "x.py"})]
    u = out.usage
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens) == (100, 20, 50, 10)
    assert [b["type"] for b in out.message.raw] == ["text", "tool_use"]


@pytest.mark.parametrize("stop,expected", [("end_turn", "end_turn"), ("max_tokens", "max_tokens"),
                                           ("refusal", "refusal"), ("pause_turn", "other")])
def test_anthropic_stop_reasons(stop, expected):
    from types import SimpleNamespace as NS

    resp = a_response([a_text("x")], stop, stop_details=NS(category="cyber"))
    out = AnthropicClient(LLMConfig.from_env(ANTH), FakeAnthropicSDK(resp)).generate("S", [UserMsg("q")], TOOLS)
    assert out.stop == expected
    assert (out.refusal_detail == "cyber") == (expected == "refusal")


def test_anthropic_errors_become_one_safe_error():
    class Boom(Exception):
        status_code = 429

    client = AnthropicClient(LLMConfig.from_env(ANTH), FakeAnthropicSDK(Boom("rate limited")))
    with pytest.raises(LLMError, match="HTTP 429"):
        client.generate("S", [UserMsg("q")], TOOLS)


# ------------------------------------------------------------------ OpenAI-compatible


def test_openai_request_shape():
    sdk = FakeOpenAISDK(o_response("hi"))
    cfg = LLMConfig.from_env({**OAI, "FIXPILOT_TEMPERATURE": "0.2"})
    OpenAICompatClient(cfg, sdk).generate("SYSTEM", conversation(), TOOLS)
    req = sdk.requests[0]
    assert req["messages"][0] == {"role": "system", "content": "SYSTEM"}
    assert req["messages"][2]["tool_calls"][0] == {
        "id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
    assert req["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": "contents"}
    assert req["tools"] == [{"type": "function", "function": {
        "name": "read_file", "description": "Read a file.", "parameters": TOOLS[0]["input_schema"]}}]
    assert req["max_tokens"] == 4096 and req["temperature"] == 0.2
    assert KEY not in repr(req) and "cache_control" not in req


def test_openai_tool_results_are_one_message_per_call_and_temperature_is_off_by_default():
    sdk = FakeOpenAISDK(o_response("hi"))
    msgs = [UserMsg("q"), AssistantMsg("", [ToolCall("a", "read_file", {}), ToolCall("b", "read_file", {})]),
            ToolResultsMsg([ToolResultItem("a", "read_file", "A"), ToolResultItem("b", "read_file", "B", True)])]
    OpenAICompatClient(LLMConfig.from_env(OAI), sdk).generate("S", msgs, TOOLS)
    req = sdk.requests[0]
    assert [m["role"] for m in req["messages"]] == ["system", "user", "assistant", "tool", "tool"]
    assert "temperature" not in req


def test_openai_max_tokens_parameter_name_is_configurable():
    sdk = FakeOpenAISDK(o_response("hi"))
    cfg = LLMConfig.from_env({**OAI, "FIXPILOT_OPENAI_MAX_TOKENS_PARAM": "max_completion_tokens"})
    OpenAICompatClient(cfg, sdk).generate("S", [UserMsg("q")], TOOLS)
    assert "max_completion_tokens" in sdk.requests[0] and "max_tokens" not in sdk.requests[0]


def test_openai_response_parsing():
    resp = o_response(None, [o_call("k1", "read_file", '{"path": "x.py"}')], "tool_calls", (100, 20, 40))
    out = OpenAICompatClient(LLMConfig.from_env(OAI), FakeOpenAISDK(resp)).generate("S", [UserMsg("q")], TOOLS)
    assert out.stop == "tool_use"
    assert out.message.tool_calls == [ToolCall("k1", "read_file", {"path": "x.py"})]
    u = out.usage  # cached tokens are split out so the fields never overlap
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens) == (60, 20, 40)


def test_openai_malformed_arguments_are_kept_as_an_error_not_a_crash():
    resp = o_response(None, [o_call("k1", "read_file", "{not json"), o_call("k2", "read_file", "[1]")],
                      "tool_calls")
    calls = OpenAICompatClient(LLMConfig.from_env(OAI), FakeOpenAISDK(resp)).generate(
        "S", [UserMsg("q")], TOOLS).message.tool_calls
    assert calls[0].args is None and "not valid JSON" in calls[0].arg_error
    assert calls[1].args is None and "JSON object" in calls[1].arg_error


@pytest.mark.parametrize("finish,calls,expected", [
    ("stop", None, "end_turn"), ("length", None, "max_tokens"), ("content_filter", None, "refusal"),
    ("stop", [o_call("k", "read_file", "{}")], "tool_use")])
def test_openai_finish_reasons(finish, calls, expected):
    client = OpenAICompatClient(LLMConfig.from_env(OAI), FakeOpenAISDK(o_response("x", calls, finish)))
    assert client.generate("S", [UserMsg("q")], TOOLS).stop == expected


def test_openai_explicit_refusal_field():
    out = OpenAICompatClient(LLMConfig.from_env(OAI), FakeOpenAISDK(o_response(None, refusal="no"))).generate(
        "S", [UserMsg("q")], TOOLS)
    assert out.stop == "refusal" and out.refusal_detail == "no"


def test_openai_missing_usage_is_estimated_and_flagged():
    out = OpenAICompatClient(LLMConfig.from_env(OAI), FakeOpenAISDK(o_response("x", usage=None))).generate(
        "S", [UserMsg("q")], TOOLS)
    assert out.usage.estimated and out.usage.input_tokens > 0


def test_openai_no_choices_and_sdk_errors():
    with pytest.raises(LLMError, match="no choices"):
        OpenAICompatClient(LLMConfig.from_env(OAI), FakeOpenAISDK(o_response(choices=False))).generate(
            "S", [UserMsg("q")], TOOLS)
    with pytest.raises(LLMError, match="example.test"):
        OpenAICompatClient(LLMConfig.from_env(OAI), FakeOpenAISDK(RuntimeError("down"))).generate(
            "S", [UserMsg("q")], TOOLS)


# ------------------------------------------------------------------ startup check: tool calling


def _oai(models=None, error=None, respond=None):
    return OpenAICompatClient(LLMConfig.from_env(OAI), FakeOpenAISDK(respond, models, error))


def test_metadata_says_tools_supported():
    assert _oai([o_model("vendor/m", ["tools", "max_tokens"])]).metadata_tool_support().status == "supported"


def test_metadata_says_tools_not_supported():
    assert _oai([o_model("vendor/m", ["max_tokens"])]).metadata_tool_support().status == "unsupported"


@pytest.mark.parametrize("models,error", [
    ([o_model("vendor/m")], None), ([o_model("other")], None), ([], None), (None, RuntimeError("x"))])
def test_metadata_unknown_cases(models, error):
    assert _oai(models, error).metadata_tool_support().status == "unknown"


def test_unsupported_model_fails_with_a_clear_message():
    with pytest.raises(ToolSupportError) as err:
        check_tool_support(_oai([o_model("vendor/m", ["max_tokens"])]))
    text = str(err.value)
    assert "vendor/m" in text and "openai_compat" in text and "tool calling" in text
    assert "FIXPILOT_MODEL" in text


def test_unknown_metadata_without_probe_does_not_spend_anything():
    sdk = FakeOpenAISDK(o_response("x"), models=[])
    client = OpenAICompatClient(LLMConfig.from_env(OAI), sdk)
    assert check_tool_support(client, probe=False).status == "unknown"
    assert sdk.requests == []  # no chat request was made


def test_probe_succeeds_when_the_model_calls_the_test_tool():
    resp = o_response(None, [o_call("p", "ping", '{"word": "ok"}')], "tool_calls")
    assert check_tool_support(_oai([], respond=resp), probe=True).status == "supported"


def test_probe_fails_when_the_model_answers_in_text():
    with pytest.raises(ToolSupportError, match="answered in text"):
        check_tool_support(_oai([], respond=o_response("I will not use tools")), probe=True)


def test_probe_fails_when_the_provider_rejects_tools():
    err = RuntimeError("this model does not support tools")
    with pytest.raises(ToolSupportError, match="rejected the tool request"):
        check_tool_support(_oai([], respond=err), probe=True)


def test_probe_reraises_unrelated_errors():
    with pytest.raises(LLMError, match="down"):
        check_tool_support(_oai([], respond=RuntimeError("connection down")), probe=True)


def test_anthropic_claude_models_are_known_to_support_tools_other_names_are_unknown():
    assert AnthropicClient(LLMConfig.from_env(ANTH), FakeAnthropicSDK()).metadata_tool_support().status == "supported"
    odd = LLMConfig.from_env({**ANTH, "FIXPILOT_MODEL": "mystery"})
    assert AnthropicClient(odd, FakeAnthropicSDK()).metadata_tool_support().status == "unknown"


def test_fake_model_without_tools_fails_the_check():
    llm = FakeLLM(scripted_policy([FakeTurn(text="x")]), tools_supported=False)
    with pytest.raises(ToolSupportError):
        check_tool_support(llm)
