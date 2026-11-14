#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two client APIs, ten free-tier providers.

The server reads the trial brief (the file named by MAKEFUN_PROMPT_FILE): the
provider directory, the sealed quality table and the workload manifest. It
probes every provider's catalog route in that provider's own wire, builds one
catalog document, listens on a port it picks, and registers the port and the
catalog with the env at MAKEFUN_ENV_URL. Each client request (OpenAI Chat
Completions or Anthropic Messages, JSON or server-sent events) is routed to
the highest-rated free model that meets the request's requirements and can
carry the request's wire features; when a provider refuses (402, 429 or 5xx)
the request fails over to the next candidate. The request is translated into
the provider's wire without losing anything the provider echoes back (system
prompt, user text, tool names, max_tokens, stop sequences, images, JSON mode),
and the provider's answer is translated back into the client's wire with its
text whole, its usage counts exact and its stop reason mapped across.

`--list` prints the catalog document and exits 0, byte for byte the document
the register body carries. With no brief it prints {"providers":[]}.

Everything is a function of the brief, the requests and the providers'
answers: no wall clock and no randomness reach a decision or a response, so a
replayed trial leaves the same state export. Standard library only.
"""

import http.client
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, urlsplit

PROTOCOL_VERSION = 1
PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"
ANTHROPIC_VERSION = "2023-06-01"
FIXED_CREATED = 1767225600  # a fixed epoch for model listings: never the clock
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5
UPSTREAM_TIMEOUT_S = 15
REGISTER_TIMEOUT_S = 5
MAX_BODY_BYTES = 32 * 1024 * 1024
DEAD_PROVIDER_REFUSALS = 3  # refusals after which a provider is tried last
SSE_PIECE_CHARS = 13  # streamed text piece size, matching the providers' own

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def _is_record(value):
    return isinstance(value, dict)


def _str_of(value):
    return value if isinstance(value, str) else ""


def _num_or_none(value):
    if isinstance(value, bool):
        return None
    return value if isinstance(value, (int, float)) and value == value else None


def _bool_or_none(value):
    return value if isinstance(value, bool) else None


# --------------------------------------------------------------------------
# The brief: three JSON documents inside prose, found by their schema field.
# --------------------------------------------------------------------------

def _object_end(text, start):
    """Index of the `}` closing the object that opens at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    """Every JSON object in `text` carrying a string `schema`, in order."""
    docs = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if _is_record(value) and isinstance(value.get("schema"), str):
                docs.append(value)
        start = text.find("{", start + 1)
    return docs


def _provider_of(value):
    if not _is_record(value):
        return None
    pid = _str_of(value.get("id"))
    wire = value.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = value.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if _str_of(value.get(key)) != "":
            base = _str_of(value.get(key))
            break
    return {
        "id": pid,
        "name": _str_of(value.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": _str_of(value.get("key_env")),
        "key": _str_of(value.get("key")),
    }


def _rating_of(value):
    if not _is_record(value):
        return None
    model = _str_of(value.get("model"))
    provider = _str_of(value.get("provider"))
    if model == "" or provider == "":
        return None
    return {
        "model": model,
        "provider": provider,
        "rating_milli": _num_or_none(value.get("rating_milli")) or 0,
        "context_length": _num_or_none(value.get("context_length")),
        "tools": value.get("tools") is True,
        "vision": value.get("vision") is True,
        "json": value.get("json") is True,
        "free": value.get("free") is True,
    }


def _workload_request_of(value):
    if not _is_record(value):
        return None
    rid = _str_of(value.get("id"))
    if rid == "":
        return None
    requires = value.get("requires") if _is_record(value.get("requires")) else {}
    parsed = {
        "tools": requires.get("tools") is True,
        "vision": requires.get("vision") is True,
        "json": requires.get("json") is True,
    }
    min_context = _num_or_none(requires.get("min_context"))
    if min_context is not None:
        parsed["min_context"] = min_context
    return {
        "id": rid,
        "api": "anthropic" if value.get("api") == "anthropic" else "openai",
        "stream": value.get("stream") is True,
        "weight": 2 if value.get("weight") == 2 else 1,
        "requires": parsed,
    }


class Brief:
    """The trial brief: directory, quality table, workload manifest."""

    def __init__(self):
        self.providers = []          # directory entries, in the brief's order
        self.ratings = {}            # (provider, model) -> quality table row
        self.manifest = []           # workload requests, in sending order
        self.error = None

    def rating_row(self, provider_id, model_id):
        return self.ratings.get((provider_id, model_id))


def read_brief(path):
    brief = Brief()
    if path == "":
        brief.error = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        brief.error = "cannot read %s: %s" % (path, err)
        return brief
    docs = _documents_in(text)
    directory = None
    for doc in docs:
        if doc.get("schema") == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list):
            directory = doc
            break
    if directory is None:
        brief.error = "the brief holds no %s document" % PROVIDERS_SCHEMA
        return brief
    for raw in directory["providers"]:
        provider = _provider_of(raw)
        if provider is not None:
            brief.providers.append(provider)
    for doc in docs:
        if doc.get("schema") == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list):
            for raw in doc["ratings"]:
                row = _rating_of(raw)
                if row is not None:
                    brief.ratings[(row["provider"], row["model"])] = row
            break
    for doc in docs:
        if doc.get("schema") == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list):
            for raw in doc["manifest"]:
                request = _workload_request_of(raw)
                if request is not None:
                    brief.manifest.append(request)
            break
    return brief


# --------------------------------------------------------------------------
# HTTP transport (standard library only), the catalog probe and the catalog.
# --------------------------------------------------------------------------

def join_base(env_url, base):
    """A provider's base path joined to the env's URL; an absolute base stands."""
    b = (base or "").strip()
    if b.startswith("http://") or b.startswith("https://"):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    """The header a provider wants its key in; no key, no header."""
    if not key:
        return {}
    if auth == "bearer":
        return {"Authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def http_request(base_url, method, path, headers=None, body=None, timeout=UPSTREAM_TIMEOUT_S):
    """One HTTP request; returns (status, headers, body bytes).

    Raises OSError or http.client.HTTPException on a transport failure.
    """
    parts = urlsplit(base_url)
    host = parts.hostname or "127.0.0.1"
    port = parts.port or (443 if parts.scheme == "https" else 80)
    prefix = parts.path.rstrip("/")
    conn_cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(host, port, timeout=timeout)
    try:
        conn.request(method, prefix + path, body=body, headers=headers or {})
        response = conn.getresponse()
        data = response.read()
        lowered = {}
        for name, value in response.getheaders():
            lowered[name.lower()] = value
        return response.status, lowered, data
    finally:
        conn.close()


def _zero_priced(item):
    """True when a listing's pricing names only zeros, None when it is silent."""
    pricing = item.get("pricing")
    if not _is_record(pricing):
        return None
    values = []
    for raw in pricing.values():
        if isinstance(raw, bool):
            continue
        if isinstance(raw, (int, float)):
            values.append(float(raw))
        elif isinstance(raw, str) and raw.strip() != "":
            try:
                values.append(float(raw))
            except ValueError:
                pass
    if not values:
        return None
    return all(v == 0 for v in values)


def _probed_model(wire, item):
    if not _is_record(item):
        return None
    if wire == "gemini":
        mid = _str_of(item.get("name"))
        if mid.startswith("models/"):
            mid = mid[len("models/"):]
    else:
        mid = _str_of(item.get("id"))
    if mid == "":
        return None
    free = _bool_or_none(item.get("free"))
    if free is None:
        free = _zero_priced(item)
    context = _num_or_none(item.get("context_length"))
    if context is None:
        context = _num_or_none(item.get("context_window"))
    if context is None:
        context = _num_or_none(item.get("inputTokenLimit"))
    display = _str_of(item.get("display_name")) or _str_of(item.get("displayName")) or mid
    return {
        "id": mid,
        "display_name": display,
        "free": free,
        "context_length": context,
        "tools": _bool_or_none(item.get("tools")),
        "vision": _bool_or_none(item.get("vision")),
        "json": _bool_or_none(item.get("json")),
    }


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT_S):
    """One provider's catalog listing, in its own wire; None when it fails."""
    base = join_base(env_url, provider["base"])
    path = "/v1beta/models" if provider["wire"] == "gemini" else "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    try:
        status, _headers, data = http_request(base, "GET", path, headers, None, timeout)
    except (OSError, http.client.HTTPException):
        return None
    if status < 200 or status >= 300:
        return None
    try:
        body = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not _is_record(body):
        return None
    items = body.get("models") if provider["wire"] == "gemini" else body.get("data")
    if not isinstance(items, list):
        return None
    models = []
    for item in items:
        model = _probed_model(provider["wire"], item)
        if model is not None:
            models.append(model)
    return models


def model_free(model, row):
    """Free per the quality table where it has a row, else the listing's word."""
    if row is not None:
        return row["free"]
    return model["free"] if model["free"] is not None else False


def build_catalog(brief, probes):
    """The one catalog document: what --list prints and /register carries."""
    providers = []
    for provider in brief.providers:
        models = probes.get(provider["id"])
        if models is None:
            continue
        seen = set()
        listed = []
        for model in sorted(models, key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = brief.rating_row(provider["id"], model["id"])
            context = model["context_length"]
            if row is not None and row["context_length"] is not None:
                context = row["context_length"]
            listed.append({
                "id": model["id"],
                "free": model_free(model, row),
                "context_length": context,
            })
        providers.append({
            "id": provider["id"],
            "base_url": provider["base"],
            "key_env": provider["key_env"],
            "wire": provider["wire"],
            "models": listed,
        })
    return {"providers": providers}


# --------------------------------------------------------------------------
# Client requests as wire-neutral features.
#
# A provider answers with an echo of what it parsed: the system prompt, the
# last user message's text, the tool names, max_tokens, the stop sequences,
# an image hash and whether JSON mode was set. A translation is correct when
# the provider's parse of the translated request equals the client's parse of
# the original, so the features below are extracted with exactly the parsers'
# semantics, and each wire builder further down reproduces them.
# --------------------------------------------------------------------------

def _blocks_text(content):
    """A string itself; a block array's text blocks joined with newlines."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if _is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = _blocks_text(content)
        return text if text != "" else json.dumps(content)
    if content is None:
        return ""
    return json.dumps(content)


def _image_from_url(url):
    """An image part from an OpenAI image_url: data URLs split into bytes."""
    if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
        media, data = url[len("data:"):].split(";base64,", 1)
        return {"kind": "image", "media_type": media or "image/png", "data": data, "url": None}
    return {"kind": "image", "media_type": None, "data": None, "url": url if isinstance(url, str) else ""}


def _image_from_source(source):
    """An image part from an Anthropic image block's source."""
    if _is_record(source):
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source.get("media_type")
            return {"kind": "image", "media_type": media if isinstance(media, str) else "image/png",
                    "data": source["data"], "url": None}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"kind": "image", "media_type": None, "data": None, "url": source["url"]}
    return {"kind": "image", "media_type": None, "data": None, "url": ""}


def _arguments_input(arguments):
    """A tool call's arguments string as an object; {} when it is not one."""
    if isinstance(arguments, str) and arguments.strip() != "":
        try:
            parsed = json.loads(arguments)
        except ValueError:
            return {}
        if _is_record(parsed):
            return parsed
    return {}


def _empty_features():
    return {
        "system_text": "",
        "messages": [],
        "tools": [],
        "tool_choice": None,
        "max_tokens": None,
        "temperature": None,
        "top_p": None,
        "stops": [],
        "stream": False,
        "json_mode": False,
        "response_format": None,
    }


def _string_list(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def parse_openai_client(body):
    """An OpenAI Chat Completions request as wire-neutral features."""
    features = _empty_features()
    system_parts = []
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not _is_record(message):
                continue
            role = message.get("role")
            content = message.get("content")
            if role == "system":
                text = _blocks_text(content)
                if text != "":
                    system_parts.append(text)
                continue
            if role == "tool":
                features["messages"].append({
                    "role": "user",
                    "parts": [{
                        "kind": "tool_result",
                        "tool_use_id": _str_of(message.get("tool_call_id")),
                        "content": _tool_result_text(content),
                        "is_error": False,
                    }],
                })
                continue
            if role not in ("user", "assistant"):
                continue
            parts = []
            if isinstance(content, str):
                if content != "":
                    parts.append({"kind": "text", "text": content})
            elif isinstance(content, list):
                for block in content:
                    if not _is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append({"kind": "text", "text": block["text"]})
                    elif block.get("type") == "image_url" and _is_record(block.get("image_url")):
                        parts.append(_image_from_url(block["image_url"].get("url")))
            if role == "assistant" and isinstance(message.get("tool_calls"), list):
                for call in message["tool_calls"]:
                    if not _is_record(call) or not _is_record(call.get("function")):
                        continue
                    function = call["function"]
                    arguments = function.get("arguments")
                    parts.append({
                        "kind": "tool_call",
                        "id": _str_of(call.get("id")),
                        "name": _str_of(function.get("name")),
                        "arguments": arguments if isinstance(arguments, str) else "",
                        "input": _arguments_input(arguments),
                    })
            features["messages"].append({"role": role, "parts": parts})
    features["system_text"] = "\n".join(system_parts)
    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if _is_record(tool) and _is_record(tool.get("function")) and isinstance(tool["function"].get("name"), str):
                function = tool["function"]
                schema = function.get("parameters")
                features["tools"].append({
                    "name": function["name"],
                    "description": _str_of(function.get("description")),
                    "schema": schema if _is_record(schema) else {"type": "object", "properties": {}},
                })
    choice = body.get("tool_choice")
    if isinstance(choice, str) and choice in ("auto", "required", "none"):
        features["tool_choice"] = {"mode": choice, "name": None}
    elif _is_record(choice) and _is_record(choice.get("function")) and isinstance(choice["function"].get("name"), str):
        features["tool_choice"] = {"mode": "tool", "name": choice["function"]["name"]}
    max_tokens = _num_or_none(body.get("max_tokens"))
    if max_tokens is None:
        max_tokens = _num_or_none(body.get("max_completion_tokens"))
    features["max_tokens"] = max_tokens
    features["temperature"] = _num_or_none(body.get("temperature"))
    features["top_p"] = _num_or_none(body.get("top_p"))
    features["stops"] = _string_list(body.get("stop"))
    features["stream"] = body.get("stream") is True
    response_format = body.get("response_format")
    if _is_record(response_format):
        features["response_format"] = response_format
        features["json_mode"] = response_format.get("type") in ("json_object", "json_schema")
    return features


def parse_anthropic_client(body):
    """An Anthropic Messages request as wire-neutral features."""
    features = _empty_features()
    features["system_text"] = _blocks_text(body.get("system"))
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not _is_record(message):
                continue
            role = message.get("role")
            if role not in ("user", "assistant"):
                continue
            content = message.get("content")
            parts = []
            if isinstance(content, str):
                if content != "":
                    parts.append({"kind": "text", "text": content})
            elif isinstance(content, list):
                for block in content:
                    if not _is_record(block):
                        continue
                    kind = block.get("type")
                    if kind == "text" and isinstance(block.get("text"), str):
                        parts.append({"kind": "text", "text": block["text"]})
                    elif kind == "image":
                        parts.append(_image_from_source(block.get("source")))
                    elif kind == "tool_use":
                        tool_input = block.get("input")
                        tool_input = tool_input if _is_record(tool_input) else {}
                        parts.append({
                            "kind": "tool_call",
                            "id": _str_of(block.get("id")),
                            "name": _str_of(block.get("name")),
                            "arguments": json.dumps(tool_input),
                            "input": tool_input,
                        })
                    elif kind == "tool_result":
                        parts.append({
                            "kind": "tool_result",
                            "tool_use_id": _str_of(block.get("tool_use_id")),
                            "content": _tool_result_text(block.get("content")),
                            "is_error": block.get("is_error") is True,
                        })
            features["messages"].append({"role": role, "parts": parts})
    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if _is_record(tool) and isinstance(tool.get("name"), str):
                schema = tool.get("input_schema")
                features["tools"].append({
                    "name": tool["name"],
                    "description": _str_of(tool.get("description")),
                    "schema": schema if _is_record(schema) else {"type": "object", "properties": {}},
                })
    choice = body.get("tool_choice")
    if _is_record(choice):
        kind = choice.get("type")
        if kind == "auto":
            features["tool_choice"] = {"mode": "auto", "name": None}
        elif kind == "any":
            features["tool_choice"] = {"mode": "required", "name": None}
        elif kind == "none":
            features["tool_choice"] = {"mode": "none", "name": None}
        elif kind == "tool" and isinstance(choice.get("name"), str):
            features["tool_choice"] = {"mode": "tool", "name": choice["name"]}
    features["max_tokens"] = _num_or_none(body.get("max_tokens"))
    features["temperature"] = _num_or_none(body.get("temperature"))
    features["top_p"] = _num_or_none(body.get("top_p"))
    features["stops"] = _string_list(body.get("stop_sequences"))
    features["stream"] = body.get("stream") is True
    return features


# --------------------------------------------------------------------------
# Wire builders: wire-neutral features as one provider's request body.
#
# Each builder reproduces the client's parsed features in its own wire (see
# the echo note above). A message that holds only tool results is folded into
# the previous user message on the Anthropic and Gemini builds: on those
# wires a tool result lives inside a user message, and a trailing one would
# otherwise become the last user message and read as empty text.
# --------------------------------------------------------------------------

def _fold_tool_results(messages):
    folded = []
    for message in messages:
        parts = message["parts"]
        only_results = message["role"] == "user" and len(parts) > 0 and all(p["kind"] == "tool_result" for p in parts)
        if only_results and folded and folded[-1]["role"] == "user":
            folded[-1] = {"role": "user", "parts": folded[-1]["parts"] + parts}
        else:
            folded.append({"role": message["role"], "parts": list(parts)})
    return folded


def _tool_names_by_id(messages):
    names = {}
    for message in messages:
        for part in message["parts"]:
            if part["kind"] == "tool_call" and part["id"] != "":
                names[part["id"]] = part["name"]
    return names


def _image_url_of(part):
    if part["data"] is not None:
        return "data:%s;base64,%s" % (part["media_type"] or "image/png", part["data"])
    return part["url"] or ""


def build_openai_body(features, model):
    messages = []
    if features["system_text"] != "":
        messages.append({"role": "system", "content": features["system_text"]})
    for message in features["messages"]:
        parts = message["parts"]
        if message["role"] == "assistant":
            text = "\n".join(p["text"] for p in parts if p["kind"] == "text")
            calls = [{
                "id": p["id"],
                "type": "function",
                "function": {"name": p["name"], "arguments": p["arguments"]},
            } for p in parts if p["kind"] == "tool_call"]
            built = {"role": "assistant", "content": text if text != "" else None}
            if calls:
                built["tool_calls"] = calls
            messages.append(built)
            continue
        rest = []
        for part in parts:
            if part["kind"] == "tool_result":
                messages.append({"role": "tool", "tool_call_id": part["tool_use_id"], "content": part["content"]})
            elif part["kind"] == "text":
                rest.append({"type": "text", "text": part["text"]})
            elif part["kind"] == "image":
                rest.append({"type": "image_url", "image_url": {"url": _image_url_of(part)}})
        if len(rest) == 1 and rest[0]["type"] == "text":
            messages.append({"role": "user", "content": rest[0]["text"]})
        elif rest:
            messages.append({"role": "user", "content": rest})
    body = {"model": model, "messages": messages}
    if features["tools"]:
        body["tools"] = [{
            "type": "function",
            "function": {"name": t["name"], "description": t["description"], "parameters": t["schema"]},
        } for t in features["tools"]]
    choice = features["tool_choice"]
    if choice is not None:
        if choice["mode"] == "tool":
            body["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
        else:
            body["tool_choice"] = choice["mode"]
    if features["max_tokens"] is not None:
        body["max_tokens"] = features["max_tokens"]
    if features["temperature"] is not None:
        body["temperature"] = features["temperature"]
    if features["top_p"] is not None:
        body["top_p"] = features["top_p"]
    if features["stops"]:
        body["stop"] = list(features["stops"])
    if features["response_format"] is not None:
        body["response_format"] = features["response_format"]
    if features["stream"]:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def _anthropic_block(part):
    if part["kind"] == "text":
        return {"type": "text", "text": part["text"]}
    if part["kind"] == "image":
        if part["data"] is not None:
            source = {"type": "base64", "media_type": part["media_type"] or "image/png", "data": part["data"]}
        else:
            source = {"type": "url", "url": part["url"] or ""}
        return {"type": "image", "source": source}
    if part["kind"] == "tool_call":
        return {"type": "tool_use", "id": part["id"], "name": part["name"], "input": part["input"]}
    block = {"type": "tool_result", "tool_use_id": part["tool_use_id"], "content": part["content"]}
    if part["is_error"]:
        block["is_error"] = True
    return block


def build_anthropic_body(features, model):
    messages = []
    for message in _fold_tool_results(features["messages"]):
        blocks = [_anthropic_block(p) for p in message["parts"]]
        if len(blocks) == 1 and blocks[0]["type"] == "text":
            messages.append({"role": message["role"], "content": blocks[0]["text"]})
        elif blocks:
            messages.append({"role": message["role"], "content": blocks})
        else:
            messages.append({"role": message["role"], "content": ""})
    body = {"model": model, "messages": messages}
    if features["system_text"] != "":
        body["system"] = features["system_text"]
    if features["tools"]:
        body["tools"] = [{
            "name": t["name"], "description": t["description"], "input_schema": t["schema"],
        } for t in features["tools"]]
    choice = features["tool_choice"]
    if choice is not None:
        if choice["mode"] == "tool":
            body["tool_choice"] = {"type": "tool", "name": choice["name"]}
        elif choice["mode"] == "required":
            body["tool_choice"] = {"type": "any"}
        else:
            body["tool_choice"] = {"type": choice["mode"]}
    if features["max_tokens"] is not None:
        body["max_tokens"] = features["max_tokens"]
    if features["temperature"] is not None:
        body["temperature"] = features["temperature"]
    if features["top_p"] is not None:
        body["top_p"] = features["top_p"]
    if features["stops"]:
        body["stop_sequences"] = list(features["stops"])
    if features["stream"]:
        body["stream"] = True
    return body


def _gemini_part(part, tool_names):
    if part["kind"] == "text":
        return {"text": part["text"]}
    if part["kind"] == "image":
        if part["data"] is not None:
            return {"inlineData": {"mimeType": part["media_type"] or "image/png", "data": part["data"]}}
        return {"fileData": {"fileUri": part["url"] or ""}}
    if part["kind"] == "tool_call":
        return {"functionCall": {"name": part["name"], "args": part["input"]}}
    return {"functionResponse": {
        "name": tool_names.get(part["tool_use_id"], ""),
        "response": {"result": part["content"]},
    }}


def build_gemini_body(features, model):
    tool_names = _tool_names_by_id(features["messages"])
    contents = []
    for message in _fold_tool_results(features["messages"]):
        role = "model" if message["role"] == "assistant" else "user"
        parts = [_gemini_part(p, tool_names) for p in message["parts"]]
        contents.append({"role": role, "parts": parts if parts else [{"text": ""}]})
    body = {"contents": contents}
    if features["system_text"] != "":
        body["systemInstruction"] = {"parts": [{"text": features["system_text"]}]}
    config = {}
    if features["max_tokens"] is not None:
        config["maxOutputTokens"] = features["max_tokens"]
    if features["temperature"] is not None:
        config["temperature"] = features["temperature"]
    if features["top_p"] is not None:
        config["topP"] = features["top_p"]
    if features["stops"]:
        config["stopSequences"] = list(features["stops"])
    if features["json_mode"]:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    if features["tools"]:
        body["tools"] = [{"functionDeclarations": [{
            "name": t["name"], "description": t["description"], "parameters": t["schema"],
        } for t in features["tools"]]}]
    choice = features["tool_choice"]
    if choice is not None:
        modes = {"auto": "AUTO", "required": "ANY", "none": "NONE", "tool": "ANY"}
        calling = {"mode": modes[choice["mode"]]}
        if choice["mode"] == "tool":
            calling["allowedFunctionNames"] = [choice["name"]]
        body["toolConfig"] = {"functionCallingConfig": calling}
    return body


# --------------------------------------------------------------------------
# Provider calls and answer extraction.
#
# A provider's answer is reduced to its text (whole), its usage counts and its
# stop reason, in whichever wire it spoke. Streams are read to the end and
# reassembled: the client asked for one answer, and the trial's client reads
# a complete response either way.
# --------------------------------------------------------------------------

def _sse_frames(text):
    """(event name or None, data string) per server-sent event frame."""
    frames = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                value = line[len("data:"):]
                if value.startswith(" "):
                    value = value[1:]
                data.append(value)
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def _answer_text(content):
    """A message content as its text: a string, or text blocks joined tight."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block["text"] for block in content
            if _is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str)
        )
    return ""


def _openai_json_answer(text):
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if not _is_record(value) or not isinstance(value.get("choices"), list) or not value["choices"]:
        return None
    choice = value["choices"][0]
    if not _is_record(choice) or not _is_record(choice.get("message")):
        return None
    usage = value.get("usage") if _is_record(value.get("usage")) else {}
    return {
        "text": _answer_text(choice["message"].get("content")),
        "input": _num_or_none(usage.get("prompt_tokens")),
        "output": _num_or_none(usage.get("completion_tokens")),
        "stop": choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None,
    }


def _openai_stream_answer(text):
    answer = {"text": [], "input": None, "output": None, "stop": None}
    for _event, data in _sse_frames(text):
        if data.strip() == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if not _is_record(chunk):
            continue
        if _is_record(chunk.get("usage")):
            answer["input"] = _num_or_none(chunk["usage"].get("prompt_tokens"))
            answer["output"] = _num_or_none(chunk["usage"].get("completion_tokens"))
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices or not _is_record(choices[0]):
            continue
        choice = choices[0]
        delta = choice.get("delta") if _is_record(choice.get("delta")) else {}
        if isinstance(delta.get("content"), str):
            answer["text"].append(delta["content"])
        if isinstance(choice.get("finish_reason"), str):
            answer["stop"] = choice["finish_reason"]
    answer["text"] = "".join(answer["text"])
    return answer


def _anthropic_json_answer(text):
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if not _is_record(value) or not isinstance(value.get("content"), list):
        return None
    usage = value.get("usage") if _is_record(value.get("usage")) else {}
    return {
        "text": _answer_text(value["content"]),
        "input": _num_or_none(usage.get("input_tokens")),
        "output": _num_or_none(usage.get("output_tokens")),
        "stop": value.get("stop_reason") if isinstance(value.get("stop_reason"), str) else None,
    }


def _anthropic_stream_answer(text):
    answer = {"text": [], "input": None, "output": None, "stop": None}
    for event, data in _sse_frames(text):
        try:
            value = json.loads(data)
        except ValueError:
            continue
        if not _is_record(value):
            continue
        if event == "message_start" and _is_record(value.get("message")) and _is_record(value["message"].get("usage")):
            answer["input"] = _num_or_none(value["message"]["usage"].get("input_tokens"))
        elif event == "content_block_delta" and _is_record(value.get("delta")):
            if value["delta"].get("type") == "text_delta" and isinstance(value["delta"].get("text"), str):
                answer["text"].append(value["delta"]["text"])
        elif event == "message_delta":
            if _is_record(value.get("usage")):
                answer["output"] = _num_or_none(value["usage"].get("output_tokens"))
            if _is_record(value.get("delta")) and isinstance(value["delta"].get("stop_reason"), str):
                answer["stop"] = value["delta"]["stop_reason"]
    answer["text"] = "".join(answer["text"])
    return answer


def _gemini_candidate(value):
    if not _is_record(value) or not isinstance(value.get("candidates"), list) or not value["candidates"]:
        return None
    candidate = value["candidates"][0]
    return candidate if _is_record(candidate) else None


def _gemini_parts_text(candidate):
    content = candidate.get("content") if _is_record(candidate.get("content")) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    return "".join(p["text"] for p in parts if _is_record(p) and isinstance(p.get("text"), str))


def _gemini_json_answer(text):
    try:
        value = json.loads(text)
    except ValueError:
        return None
    candidate = _gemini_candidate(value)
    if candidate is None:
        return None
    usage = value.get("usageMetadata") if _is_record(value.get("usageMetadata")) else {}
    return {
        "text": _gemini_parts_text(candidate),
        "input": _num_or_none(usage.get("promptTokenCount")),
        "output": _num_or_none(usage.get("candidatesTokenCount")),
        "stop": candidate.get("finishReason") if isinstance(candidate.get("finishReason"), str) else None,
    }


def _gemini_stream_answer(text):
    answer = {"text": [], "input": None, "output": None, "stop": None}
    for _event, data in _sse_frames(text):
        try:
            value = json.loads(data)
        except ValueError:
            continue
        candidate = _gemini_candidate(value)
        if candidate is None:
            continue
        answer["text"].append(_gemini_parts_text(candidate))
        if isinstance(candidate.get("finishReason"), str):
            answer["stop"] = candidate["finishReason"]
        if _is_record(value.get("usageMetadata")):
            answer["input"] = _num_or_none(value["usageMetadata"].get("promptTokenCount"))
            answer["output"] = _num_or_none(value["usageMetadata"].get("candidatesTokenCount"))
    answer["text"] = "".join(answer["text"])
    return answer


def extract_answer(wire, stream, data):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if wire == "openai":
        return _openai_stream_answer(text) if stream else _openai_json_answer(text)
    if wire == "anthropic":
        return _anthropic_stream_answer(text) if stream else _anthropic_json_answer(text)
    return _gemini_stream_answer(text) if stream else _gemini_json_answer(text)


def post_provider(env_url, provider, path, body_obj):
    """POST one provider route; (status, body bytes). Raises on transport failure."""
    base = join_base(env_url, provider["base"])
    headers = {"content-type": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    payload = json.dumps(body_obj).encode("utf-8")
    status, _headers, data = http_request(base, "POST", path, headers, payload, UPSTREAM_TIMEOUT_S)
    return status, data


# --------------------------------------------------------------------------
# Routing: the best free model that can serve, with failover.
# --------------------------------------------------------------------------

STOP_FAMILIES = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
STOP_WORDS = {
    "openai": ["stop", "length", "tool_calls", "content_filter"],
    "anthropic": ["end_turn", "max_tokens", "tool_use", "refusal"],
}


def stop_word(stop, api):
    """The client's word for the provider stop reason's family."""
    family = STOP_FAMILIES.get(stop, 0) if isinstance(stop, str) else 0
    return STOP_WORDS[api][family]


def _meets(requires, caps):
    if requires.get("tools") is True and caps["tools"] is not True:
        return False
    if requires.get("vision") is True and caps["vision"] is not True:
        return False
    if requires.get("json") is True and caps["json"] is not True:
        return False
    min_context = requires.get("min_context")
    if min_context is not None and (caps["context_length"] is None or caps["context_length"] < min_context):
        return False
    return True


class Router:
    """Candidates ranked by the quality table, and the trial's refusal memory."""

    def __init__(self, brief, probes):
        self.brief = brief
        self.providers = {}   # id -> directory entry, for providers that answered
        self.models = {}      # id -> {model id: probed model}
        for provider in brief.providers:
            models = probes.get(provider["id"])
            if models is None:
                continue
            self.providers[provider["id"]] = provider
            self.models[provider["id"]] = {m["id"]: m for m in models}
        self.refusals = {}    # provider id -> refusals seen this trial

    def candidates(self, requires, json_mode):
        """(provider, model id) pairs, best first; dead providers last."""
        ranked = []
        for index, provider in enumerate(self.brief.providers):
            if provider["id"] not in self.providers:
                continue
            for model_id, model in self.models[provider["id"]].items():
                row = self.brief.rating_row(provider["id"], model_id)
                if row is None or not model_free(model, row):
                    continue
                caps = {
                    "tools": row["tools"], "vision": row["vision"], "json": row["json"],
                    "context_length": row["context_length"] if row["context_length"] is not None else model["context_length"],
                }
                if not _meets(requires, caps):
                    continue
                # An Anthropic-wire provider parses JSON mode as absent, so it
                # can never echo a JSON-mode request back: it is not a route.
                if json_mode and provider["wire"] == "anthropic":
                    continue
                ranked.append((-row["rating_milli"], index, model_id, provider))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        alive = [item for item in ranked if self.refusals.get(item[3]["id"], 0) < DEAD_PROVIDER_REFUSALS]
        dead = [item for item in ranked if self.refusals.get(item[3]["id"], 0) >= DEAD_PROVIDER_REFUSALS]
        return [(item[3], item[2]) for item in alive + dead]

    def note_refusal(self, provider_id):
        self.refusals[provider_id] = self.refusals.get(provider_id, 0) + 1

    def served_models(self):
        """The free models GET /v1/models lists, per provider in brief order."""
        out = []
        for provider in self.brief.providers:
            if provider["id"] not in self.providers:
                continue
            for model_id in sorted(self.models[provider["id"]]):
                model = self.models[provider["id"]][model_id]
                row = self.brief.rating_row(provider["id"], model_id)
                if model_free(model, row):
                    out.append({"id": model_id, "provider": provider["id"], "display_name": model["display_name"]})
        return out


def _upstream_body(api, provider, model_id, features, raw_body):
    """The provider's request body: a pass-through on the client's own wire."""
    wire = provider["wire"]
    if wire == api:
        body = dict(raw_body)
        body["model"] = model_id
        if wire == "openai" and features["stream"]:
            options = dict(body.get("stream_options")) if _is_record(body.get("stream_options")) else {}
            options["include_usage"] = True
            body["stream_options"] = options
        return body
    if wire == "openai":
        return build_openai_body(features, model_id)
    if wire == "anthropic":
        return build_anthropic_body(features, model_id)
    return build_gemini_body(features, model_id)


def _provider_path(provider, model_id, stream):
    wire = provider["wire"]
    if wire == "openai":
        return "/v1/chat/completions"
    if wire == "anthropic":
        return "/v1/messages"
    verb = "streamGenerateContent" if stream else "generateContent"
    return "/v1beta/models/%s:%s" % (quote(model_id, safe=""), verb)


def serve_with_failover(router, api, raw_body, features, requires):
    """Answer one client request through the candidates, best first.

    Returns the extracted answer of the first provider that answers with a
    whole answer (text and both usage counts); when no provider gives a whole
    one, the first partial answer is returned so the response still carries
    what a provider said. Returns None when nothing answered at all.
    """
    partial = None
    for provider, model_id in router.candidates(requires, features["json_mode"]):
        body = _upstream_body(api, provider, model_id, features, raw_body)
        path = _provider_path(provider, model_id, features["stream"])
        attempts = 0
        while True:
            attempts += 1
            try:
                status, data = post_provider(ENV_URL, provider, path, body)
            except (OSError, http.client.HTTPException):
                status, data = 0, b""
            if 200 <= status < 300:
                answer = extract_answer(provider["wire"], features["stream"], data)
                if answer is not None:
                    answer["model"] = model_id
                    if answer["input"] is not None and answer["output"] is not None:
                        return answer
                    if partial is None:
                        partial = answer
                    break
            # A refusal: the failure schedules refuse single calls, so a
            # provider's first refusal of the trial earns one immediate retry
            # before the request moves on; a refusal after that, or a second
            # in a row, is read as a spent budget or an outage and is final.
            first_refusal = router.refusals.get(provider["id"], 0) == 0
            router.note_refusal(provider["id"])
            if not (first_refusal and attempts == 1):
                break
    return partial


# --------------------------------------------------------------------------
# Client-wire rendering: the provider's answer in the client's own shape.
# --------------------------------------------------------------------------

def _pieces(text):
    pieces = [text[i:i + SSE_PIECE_CHARS] for i in range(0, len(text), SSE_PIECE_CHARS)]
    return pieces if pieces else [""]


def _usage_of(answer):
    return (answer["input"] or 0, answer["output"] or 0)


def render_openai_json(answer):
    prompt, completion = _usage_of(answer)
    return {
        "id": "chatcmpl-aggregator",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": answer["model"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer["text"]},
            "finish_reason": stop_word(answer["stop"], "openai"),
        }],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
    }


def render_openai_stream(answer):
    prompt, completion = _usage_of(answer)
    model = answer["model"]

    def chunk(delta, finish):
        return {"id": "chatcmpl-aggregator", "object": "chat.completion.chunk", "created": FIXED_CREATED,
                "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    frames = ["data: %s\n\n" % json.dumps(chunk({"role": "assistant", "content": ""}, None))]
    for piece in _pieces(answer["text"]):
        frames.append("data: %s\n\n" % json.dumps(chunk({"content": piece}, None)))
    frames.append("data: %s\n\n" % json.dumps(chunk({}, stop_word(answer["stop"], "openai"))))
    usage_chunk = {"id": "chatcmpl-aggregator", "object": "chat.completion.chunk", "created": FIXED_CREATED,
                   "model": model, "choices": [],
                   "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}}
    frames.append("data: %s\n\n" % json.dumps(usage_chunk))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def render_anthropic_json(answer):
    prompt, completion = _usage_of(answer)
    return {
        "id": "msg_aggregator",
        "type": "message",
        "role": "assistant",
        "model": answer["model"],
        "content": [{"type": "text", "text": answer["text"]}],
        "stop_reason": stop_word(answer["stop"], "anthropic"),
        "stop_sequence": None,
        "usage": {"input_tokens": prompt, "output_tokens": completion},
    }


def _sse_event(event, data):
    return "event: %s\ndata: %s\n\n" % (event, json.dumps(data))


def render_anthropic_stream(answer):
    prompt, completion = _usage_of(answer)
    frames = [_sse_event("message_start", {
        "type": "message_start",
        "message": {"id": "msg_aggregator", "type": "message", "role": "assistant", "model": answer["model"],
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": prompt, "output_tokens": 0}},
    })]
    frames.append(_sse_event("content_block_start", {
        "type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}))
    for piece in _pieces(answer["text"]):
        frames.append(_sse_event("content_block_delta", {
            "type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece}}))
    frames.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(_sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_word(answer["stop"], "anthropic"), "stop_sequence": None},
        "usage": {"output_tokens": completion}}))
    frames.append(_sse_event("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def error_document(api, message):
    if api == "anthropic":
        return {"type": "error", "error": {"type": "api_error", "message": message}}
    return {"error": {"message": message, "type": "api_error", "code": None}}


def openai_model_list(models):
    return {"object": "list", "data": [
        {"id": m["id"], "object": "model", "created": FIXED_CREATED, "owned_by": m["provider"]} for m in models]}


def anthropic_model_list(models):
    return {"data": [
        {"type": "model", "id": m["id"], "display_name": m["display_name"], "created_at": FIXED_CREATED_AT}
        for m in models], "has_more": False}


# --------------------------------------------------------------------------
# The HTTP server: the two client routes and the model list.
# --------------------------------------------------------------------------

class State:
    """Everything the handlers share: the brief, the router, the position."""

    def __init__(self, brief, router):
        self.brief = brief
        self.router = router
        self.index = 0  # the manifest is in arrival order, one at a time
        self.lock = threading.Lock()

    def next_requires(self, features):
        if self.index < len(self.brief.manifest):
            entry = self.brief.manifest[self.index]
            self.index += 1
            return entry["requires"]
        self.index += 1
        return {
            "tools": len(features["tools"]) > 0,
            "vision": any(p["kind"] == "image" for m in features["messages"] for p in m["parts"]),
            "json": features["json_mode"],
        }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "aggregator/1"

    def log_message(self, *args):
        pass

    def _send(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status, document):
        self._send(status, "application/json", json.dumps(document).encode("utf-8"))

    def _read_body(self):
        length = self.headers.get("content-length")
        if length is None:
            return b""
        try:
            remaining = int(length)
        except ValueError:
            return b""
        chunks = []
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/v1/models":
            models = self.server.state.router.served_models()
            if self.headers.get("x-api-key") is not None:
                self._send_json(200, anthropic_model_list(models))
            else:
                self._send_json(200, openai_model_list(models))
            return
        self._send_json(404, error_document("openai", "no route GET %s" % path))

    def do_POST(self):
        path = urlsplit(self.path).path
        if path == "/v1/chat/completions":
            api = "openai"
        elif path == "/v1/messages":
            api = "anthropic"
        else:
            self._send_json(404, error_document("openai", "no route POST %s" % path))
            return
        raw = self._read_body()
        if len(raw) > MAX_BODY_BYTES:
            self._send_json(400, error_document(api, "the request body is too large"))
            return
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            body = None
        if not _is_record(body):
            self._send_json(400, error_document(api, "the request body is not one JSON object"))
            return
        state = self.server.state
        with state.lock:
            features = parse_openai_client(body) if api == "openai" else parse_anthropic_client(body)
            requires = state.next_requires(features)
            answer = serve_with_failover(state.router, api, body, features, requires)
        if answer is None:
            self._send_json(503, error_document(api, "no free model on the directory could serve this request"))
            return
        if features["stream"]:
            text = render_openai_stream(answer) if api == "openai" else render_anthropic_stream(answer)
            self._send(200, "text/event-stream; charset=utf-8", text.encode("utf-8"))
        elif api == "openai":
            self._send_json(200, render_openai_json(answer))
        else:
            self._send_json(200, render_anthropic_json(answer))


# --------------------------------------------------------------------------
# Startup: probe, listen, register, poll /work, exit 0 when it says done.
# --------------------------------------------------------------------------

def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def register(port, catalog):
    payload = json.dumps({
        "protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog,
    }).encode("utf-8")
    for _attempt in range(75):
        try:
            status, _headers, _data = http_request(
                ENV_URL, "POST", "/register", {"content-type": "application/json"}, payload, REGISTER_TIMEOUT_S)
            if 200 <= status < 300:
                return True
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(0.2)
    return False


def poll_until_done(server):
    while True:
        try:
            status, _headers, data = http_request(ENV_URL, "GET", "/work", {}, None, 2)
            if status == 200:
                work = json.loads(data.decode("utf-8"))
                if _is_record(work) and work.get("done") is True:
                    break
        except (OSError, http.client.HTTPException, ValueError, UnicodeDecodeError):
            pass
        time.sleep(POLL_INTERVAL_S)
    server.shutdown()


def main(argv):
    brief = read_brief(PROMPT_FILE)
    if brief.error is not None:
        log(brief.error)
    probes = {}
    for provider in brief.providers:
        probes[provider["id"]] = probe_provider(ENV_URL, provider) if ENV_URL != "" else None
    catalog = build_catalog(brief, probes)
    if "--list" in argv:
        sys.stdout.write(json.dumps(catalog) + "\n")
        sys.stdout.flush()
        return 0
    if ENV_URL == "":
        log("MAKEFUN_ENV_URL is not set")
        return 1
    router = Router(brief, probes)
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    server.state = State(brief, router)
    port = server.server_address[1]
    if port == 8080:
        log("the OS picked 8080, which the env owns on one layout")
        return 1
    log("listening on 0.0.0.0:%d with %d providers and %d free models" % (
        port, len(catalog["providers"]), len(router.served_models())))
    if not register(port, catalog):
        log("the env never accepted the register call")
        return 1

    def _term(_signum, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    threading.Thread(target=poll_until_done, args=(server,), daemon=True).start()
    server.serve_forever(poll_interval=0.05)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
