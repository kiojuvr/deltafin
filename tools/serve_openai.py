#!/usr/bin/env python3
"""Deltafin OpenAI-compatible API server.

Exposes /v1/chat/completions, /v1/completions, and /v1/models over HTTP so any
OpenAI-SDK client, chat UI, or coding agent can talk to Kimi K3 running locally:

    OPENAI_BASE_URL=http://127.0.0.1:8000/v1  OPENAI_API_KEY=none  <your tool>

Design notes, honestly stated:
  * Decoding is greedy and reproducible; temperature/top_p are accepted and
    ignored. Omitted max_tokens means "until the model finishes" for chat
    (K3 emits an end token) and 256 for raw completions (which never end on
    their own). Explicit max_tokens is honored as-is; operators can set a
    ceiling with K3_SERVER_MAX_TOKENS.
  * One generation at a time (a global lock). Concurrency would be meaningless
    at this speed.
  * Chat mode renders K3's template, which includes a thinking section; the
    response splits it into `reasoning_content` and `content` (DeepSeek-style).
  * Each request uses a fresh KV/state cache by default. K3_PREFIX_STATE=1 is
    an experimental exact-within-segmented-mode prefix snapshot. It is not in
    the selected profile because segmented prefill changes long-prompt
    floating-point numerics relative to the monolithic runtime.
  * K3_PREFIX_ACTIVATION=1 retains only routed-MoE inputs/outputs for the fixed
    chat prefix, keyed by the full prompt length. It preserves monolithic shape,
    validates every retained input/route/weight bit, and is bounded by
    K3_PREFIX_ACTIVATION_ENTRIES.
  * A chat request's prompt is ~60+ tokens; on a COLD expert cache the prefill
    can take hours because most experts get fetched. Warm up with short
    completions first, or let the cache grow across sessions.

Usage:  python tools/serve_openai.py [--port 8000]
(device and spine format are auto-detected; see K3_DEV / K3_SPINE to override)

No dependencies beyond the standard library + what kimi_run already needs.
"""
import argparse
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimi_run as kr  # noqa: E402
from prefix_state import (  # noqa: E402
    PrefixStateSnapshot,
    derive_chat_prefix,
)
from prefix_activation import PrefixActivationCache  # noqa: E402
from response_memo import DeterministicResponseMemo  # noqa: E402

MODEL_ID = "deltafin-kimi-k3"
MAX_TOKENS_CAP = int(os.environ.get("K3_SERVER_MAX_TOKENS", "0"))   # 0 = no cap
RESPONSE_MEMO_ENTRIES = int(
    os.environ.get("K3_RESPONSE_MEMO_ENTRIES", "32"))
PREFIX_STATE_ENABLED = os.environ.get("K3_PREFIX_STATE", "0") == "1"
PREFIX_ACTIVATION_ENABLED = (
    os.environ.get("K3_PREFIX_ACTIVATION", "0") == "1"
)
PREFIX_ACTIVATION_ENTRIES = int(
    os.environ.get("K3_PREFIX_ACTIVATION_ENTRIES", "2")
)
RESPONSE_MARKER = "<|open|>response<|sep|>"
THINK_CLOSE = "<|close|>think<|sep|>"

_lock = threading.Lock()
_tok = None
_layers = None
_embed = None
_memo = DeterministicResponseMemo(RESPONSE_MEMO_ENTRIES)
_request_metrics = None
_chat_prefix_ids = ()
_prefix_snapshot = None
_prefix_builds = 0
_prefix_activations = None


def _configure_request_metrics():
    global _request_metrics
    path = os.environ.get("K3_SERVER_METRICS_JSONL")
    if not path:
        return
    from long_lived_metrics import LongLivedRequestMetrics
    from validate_m1d_resident import memory_snapshot

    loader = getattr(kr, "direct_shard_loader", None)

    def stats():
        return dict(loader.stats) if loader is not None else {}

    def routes():
        return {
            int(layer): [int(expert) for expert in experts]
            for layer, experts in kr._LAST_SEL.items()
        }

    _request_metrics = LongLivedRequestMetrics(
        Path(path),
        snapshot=lambda label: memory_snapshot(label),
        stats=stats,
        routes=routes,
        expert_bytes=17_547_264,
    )
    print(f"[serve] request metrics: {Path(path).expanduser()}", flush=True)


def _metric_begin(
    rid, mode, ids, max_new, memo_hit, prefix_state=None,
    prefix_activation=None,
):
    if _request_metrics is None:
        return None
    try:
        return _request_metrics.begin(
            request_id=rid,
            mode=mode,
            input_tokens=len(ids),
            max_new_tokens=max_new,
            memo_hit=memo_hit,
            prefix_state=prefix_state,
            prefix_activation=prefix_activation,
        )
    except Exception as exc:
        print(f"[serve] metrics begin failed: {exc!r}", flush=True)
        return None


def _metric_finish(session, status, output_tokens=0, error=None):
    if session is None or _request_metrics is None:
        return
    try:
        _request_metrics.finish(
            session,
            status=status,
            output_tokens=output_tokens,
            error=error,
        )
    except Exception as exc:
        print(f"[serve] metrics finish failed: {exc!r}", flush=True)


def _boot():
    global _tok, _layers, _embed, _chat_prefix_ids, _prefix_activations
    if PREFIX_STATE_ENABLED and PREFIX_ACTIVATION_ENABLED:
        raise RuntimeError(
            "K3_PREFIX_STATE and K3_PREFIX_ACTIVATION are mutually exclusive"
        )
    print("[serve] loading tokenizer + layer skeletons...", flush=True)
    _tok = kr.k3_official.load_tokenizer(kr.ROOT)
    if PREFIX_STATE_ENABLED or PREFIX_ACTIVATION_ENABLED:
        _chat_prefix_ids = derive_chat_prefix(_tok)
    if PREFIX_STATE_ENABLED:
        print(
            f"[serve] exact chat prefix state enabled: "
            f"{len(_chat_prefix_ids)} fixed tokens (lazy build)",
            flush=True,
        )
    if PREFIX_ACTIVATION_ENABLED:
        _prefix_activations = PrefixActivationCache(
            _chat_prefix_ids, PREFIX_ACTIVATION_ENTRIES
        )
        print(
            f"[serve] shape-stable chat prefix activation enabled: "
            f"{len(_chat_prefix_ids)} fixed tokens, "
            f"{PREFIX_ACTIVATION_ENTRIES} shape slot(s) (lazy build)",
            flush=True,
        )
    kr.check_expert_pool()
    _layers = kr.build_layers()
    _embed = kr.LazyEmbed()
    _configure_request_metrics()
    print("[serve] ready", flush=True)


def _split_reasoning(text):
    """K3 chat output: <think...><|close|>think<|sep|><|open|>response<|sep|><answer>."""
    if RESPONSE_MARKER in text:
        pre, ans = text.rsplit(RESPONSE_MARKER, 1)
        reasoning = pre.replace(THINK_CLOSE, "").replace("<|open|>think<|sep|>", "").strip()
        return reasoning or None, ans
    return None, text


def _prefix_plan(mode, ids):
    tokens = tuple(int(token) for token in ids)
    eligible = (
        PREFIX_STATE_ENABLED
        and mode == "chat"
        and bool(_chat_prefix_ids)
        and len(tokens) > len(_chat_prefix_ids)
        and tokens[: len(_chat_prefix_ids)] == _chat_prefix_ids
    )
    return {
        "enabled": PREFIX_STATE_ENABLED,
        "eligible": eligible,
        "used": False,
        "hit": bool(eligible and _prefix_snapshot is not None),
        "prefix_tokens": len(_chat_prefix_ids) if eligible else 0,
    }


def _prefix_activation_plan(mode, ids):
    if _prefix_activations is None:
        return {
            "enabled": PREFIX_ACTIVATION_ENABLED,
            "eligible": False,
            "used": False,
            "hit": False,
            "action": "none",
            "prefix_tokens": 0,
            "total_positions": 0,
        }
    return _prefix_activations.preview(mode, ids)


def _prepare_cache(mode, ids):
    global _prefix_snapshot, _prefix_builds
    cache = kr.ml.KimiDynamicCache(kr.config)
    plan = _prefix_plan(mode, ids)
    if not plan["eligible"]:
        return cache, 0, plan
    if _prefix_snapshot is None:
        started = time.perf_counter()
        kr._step_ctx["step"] = 0
        kr.forward_pass(
            _layers,
            cache,
            _embed(_chat_prefix_ids),
            step=0,
            verbose=False,
        )
        kr._device_synchronize()
        _prefix_snapshot = PrefixStateSnapshot.capture(
            cache, _chat_prefix_ids
        )
        _prefix_builds += 1
        plan["build_seconds"] = time.perf_counter() - started
        plan["snapshot"] = _prefix_snapshot.snapshot()
        print(
            f"[serve] exact prefix state built in "
            f"{plan['build_seconds']:.3f}s: "
            f"{plan['snapshot']['storage_bytes'] / 1e9:.3f} GB",
            flush=True,
        )
    else:
        _prefix_snapshot.restore(cache)
        plan["hit"] = True
        plan["snapshot"] = _prefix_snapshot.snapshot()
        print(
            f"[serve] exact prefix state hit "
            f"({_prefix_snapshot.description().prefix_tokens} tokens)",
            flush=True,
        )
    plan["used"] = True
    plan["builds"] = _prefix_builds
    return cache, len(_chat_prefix_ids), plan


def _gen(
    ids,
    max_new,
    on_delta=None,
    metric_session=None,
    mode="completion",
):
    """Run one generation under the global lock; stream decoded-text deltas."""
    cache, prefill_offset, prefix_plan = _prepare_cache(mode, ids)
    activation_session = None
    activation_plan = _prefix_activation_plan(mode, ids)
    if _prefix_activations is not None:
        activation_session, activation_plan = _prefix_activations.begin(
            mode, ids
        )
    if metric_session is not None:
        metric_session["prefix_state"] = prefix_plan
        metric_session["prefix_activation"] = activation_plan
    toks = []
    decoder = kr.IncrementalTokenDecoder(_tok) if on_delta else None

    def on_token(t):
        if metric_session is not None and _request_metrics is not None:
            try:
                _request_metrics.observe_token(metric_session)
            except Exception as exc:
                print(f"[serve] metrics token failed: {exc!r}", flush=True)
        if t == kr.EOS_ID:
            return
        toks.append(t)
        delta = decoder.append(t) if decoder is not None else ""
        if on_delta and delta:
            on_delta(delta)

    try:
        out = kr.generate(
            _layers,
            cache,
            _embed,
            ids,
            max_new,
            on_token=on_token,
            prefill_offset=prefill_offset,
            prefill_activation_session=activation_session,
        )
    except Exception:
        if _prefix_activations is not None:
            _prefix_activations.abort(
                activation_session, activation_plan
            )
        raise
    if activation_session is not None:
        _prefix_activations.complete(
            activation_session,
            activation_plan,
            expected_layers=kr.NL - 1,
        )
        action = activation_plan["action"]
        snapshot = activation_plan["activation"]
        print(
            f"[serve] prefix activation {action}: "
            f"shape={activation_plan['total_positions']}, "
            f"owned={snapshot['owned_bytes'] / 2**20:.1f} MiB, "
            f"skipped_edges={snapshot['skipped_route_edges']}",
            flush=True,
        )
    if prefix_plan["used"]:
        _prefix_snapshot.assert_intact()
    if decoder is not None:
        tail = decoder.finish()
        if tail:
            on_delta(tail)
    if kr.EOS_ID in out:
        out = out[:out.index(kr.EOS_ID)]
        finish = "stop"
    else:
        finish = "length"
    return out, _tok.decode(out), finish


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        print(f"[serve] {self.address_string()} {fmt % a}", flush=True)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg):
        self._json(code, {"error": {"message": msg, "type": "invalid_request_error"}})

    def do_GET(self):
        if self.path in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "owned_by": "deltafin"}]})
        else:
            self._err(404, f"no route {self.path}")

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
        except json.JSONDecodeError:
            return self._err(400, "invalid JSON body")
        chat = self.path in ("/v1/chat/completions", "/chat/completions")
        comp = self.path in ("/v1/completions", "/completions")
        if not (chat or comp):
            return self._err(404, f"no route {self.path}")

        if chat:
            messages = body.get("messages")
            if not messages:
                return self._err(400, "messages required")
            ids = _tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        else:
            prompt = body.get("prompt")
            if not isinstance(prompt, str):
                return self._err(400, "prompt (string) required")
            ids = _tok.encode(prompt)
        # OpenAI semantics: omitted max_tokens means the model decides. Chat ends
        # naturally at EOS; raw completions have no terminator, so only THEY get
        # a default cap (256) — an explicit max_tokens is always honored.
        req_max = body.get("max_tokens")
        if req_max:
            max_new = int(req_max)
        else:
            max_new = 1_000_000 if chat else 256
        if MAX_TOKENS_CAP:
            max_new = min(max_new, MAX_TOKENS_CAP)
        stream = bool(body.get("stream"))
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:20]
        created = int(time.time())

        if not _lock.acquire(timeout=5):
            return self._err(429, "a generation is already running (Deltafin serves one at a time)")
        try:
            mode = "chat" if chat else "completion"
            cached = _memo.get(mode, ids, max_new)
            prefix_plan = _prefix_plan(mode, ids)
            activation_plan = _prefix_activation_plan(mode, ids)
            if cached is not None:
                prefix_plan["hit"] = False
                prefix_plan["skip_reason"] = "response-memo"
                activation_plan["hit"] = False
                activation_plan["skip_reason"] = "response-memo"
            metric_session = _metric_begin(
                rid,
                mode,
                ids,
                max_new,
                cached is not None,
                prefix_state=prefix_plan,
                prefix_activation=activation_plan,
            )
            if cached is not None:
                print(f"[serve] deterministic response memo hit "
                      f"({len(cached.token_ids)} tokens)", flush=True)
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                def sse(obj):
                    self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                    self.wfile.flush()

                def on_delta(delta):
                    if chat:
                        sse({"id": rid, "object": "chat.completion.chunk", "created": created,
                             "model": MODEL_ID,
                             "choices": [{"index": 0, "delta": {"content": delta},
                                          "finish_reason": None}]})
                    else:
                        sse({"id": rid, "object": "text_completion", "created": created,
                             "model": MODEL_ID,
                             "choices": [{"index": 0, "text": delta, "finish_reason": None}]})

                if cached is None:
                    out, text, finish = _gen(
                        ids,
                        max_new,
                        on_delta=on_delta,
                        metric_session=metric_session,
                        mode=mode,
                    )
                    _memo.put(mode, ids, max_new, out, text, finish)
                else:
                    out = list(cached.token_ids)
                    text = cached.text
                    finish = cached.finish_reason
                    if text:
                        on_delta(text)
                _metric_finish(
                    metric_session,
                    "memo-hit" if cached is not None else "ok",
                    len(out),
                )
                key = "delta" if chat else "text"
                sse({"id": rid, "object": "chat.completion.chunk" if chat else "text_completion",
                     "created": created, "model": MODEL_ID,
                     "choices": [{"index": 0, key: {} if chat else "", "finish_reason": finish}]})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            if cached is None:
                out, text, finish = _gen(
                    ids,
                    max_new,
                    metric_session=metric_session,
                    mode=mode,
                )
                _memo.put(mode, ids, max_new, out, text, finish)
            else:
                out = list(cached.token_ids)
                text = cached.text
                finish = cached.finish_reason
            _metric_finish(
                metric_session,
                "memo-hit" if cached is not None else "ok",
                len(out),
            )
            usage = {"prompt_tokens": len(ids), "completion_tokens": len(out),
                     "total_tokens": len(ids) + len(out)}
            if chat:
                reasoning, content = _split_reasoning(text)
                msg = {"role": "assistant", "content": content}
                if reasoning:
                    msg["reasoning_content"] = reasoning
                self._json(200, {"id": rid, "object": "chat.completion", "created": created,
                                 "model": MODEL_ID, "usage": usage,
                                 "choices": [{"index": 0, "message": msg,
                                              "finish_reason": finish}]})
            else:
                self._json(200, {"id": rid, "object": "text_completion", "created": created,
                                 "model": MODEL_ID, "usage": usage,
                                 "choices": [{"index": 0, "text": text,
                                              "finish_reason": finish}]})
        except BrokenPipeError:
            _metric_finish(
                locals().get("metric_session"),
                "client-disconnected",
                len(locals().get("out", ())),
            )
            print("[serve] client disconnected mid-generation", flush=True)
        except Exception as e:
            _metric_finish(
                locals().get("metric_session"),
                "error",
                len(locals().get("out", ())),
                repr(e),
            )
            print(f"[serve] error: {e!r}", flush=True)
            try:
                self._err(500, str(e))
            except Exception:
                pass
        finally:
            _lock.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    _boot()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] Deltafin OpenAI-compatible API on http://{args.host}:{args.port}/v1",
          flush=True)
    print("[serve] note: ~1 token/min; first chat request on a cold expert cache "
          "is very slow (large prefill fetch). Warm up with short completions.", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
