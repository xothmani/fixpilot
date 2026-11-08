"""Shared test helpers."""

import hashlib

import numpy as np

from fixpilot.retrieval.bm25 import tokenize


class FakeEmbedder:
    """Deterministic fake: hashes words into a small vector. Records every query it sees."""

    def __init__(self):
        self.queries: list[str] = []

    def _vec(self, text):
        v = np.zeros(32, dtype=np.float32)
        for w in tokenize(text):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 32] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts):
        return np.vstack([self._vec(t) for t in texts])

    def embed_query(self, text):
        self.queries.append(text)
        return self._vec(text)


# ---------------------------------------------------------------- fake SDK objects
from types import SimpleNamespace as NS


class FakeAnthropicSDK:
    """Stands in for anthropic.Anthropic(). Records every request. Never touches the network."""

    def __init__(self, respond=None):
        self.requests: list[dict] = []
        self._respond = respond
        self.messages = NS(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if isinstance(self._respond, Exception):
            raise self._respond
        return self._respond(kwargs) if callable(self._respond) else self._respond


def a_text(text):
    return NS(type="text", text=text)


def a_tool(id, name, input):
    return NS(type="tool_use", id=id, name=name, input=input)


def a_response(content, stop="end_turn", usage=None, stop_details=None):
    u = usage or {}
    return NS(content=content, stop_reason=stop, stop_details=stop_details, usage=NS(
        input_tokens=u.get("in", 10), output_tokens=u.get("out", 5),
        cache_read_input_tokens=u.get("read", 0), cache_creation_input_tokens=u.get("write", 0)))


class FakeOpenAISDK:
    def __init__(self, respond=None, models=None, models_error=None):
        self.requests: list[dict] = []
        self._respond, self._models, self._models_error = respond, models or [], models_error
        self.chat = NS(completions=NS(create=self._create))
        self.models = NS(list=self._list)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if isinstance(self._respond, Exception):
            raise self._respond
        return self._respond(kwargs) if callable(self._respond) else self._respond

    def _list(self):
        if self._models_error:
            raise self._models_error
        return self._models


def o_call(id, name, arguments):
    return NS(id=id, type="function", function=NS(name=name, arguments=arguments))


def o_response(content=None, calls=None, finish="stop", usage=(10, 5, 0), refusal=None, choices=True):
    msg = NS(content=content, tool_calls=calls, refusal=refusal)
    u = None if usage is None else NS(prompt_tokens=usage[0], completion_tokens=usage[1],
                                      prompt_tokens_details=NS(cached_tokens=usage[2]))
    return NS(choices=[NS(message=msg, finish_reason=finish)] if choices else [], usage=u)


def o_model(id, params="absent"):
    extra = {} if params == "absent" else {"supported_parameters": params}
    return NS(id=id, model_extra=extra)
