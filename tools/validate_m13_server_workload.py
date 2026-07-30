#!/usr/bin/env python3
"""Run a two-request M12 capture/replay workload through the real API server."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m13_prefix_workload import load_records, summarize  # noqa: E402
from validate_m1d_resident import memory_snapshot, write_evidence  # noqa: E402
from validate_m2_serial_token import (  # noqa: E402
    model_tree_fingerprint,
    require_profile,
)


ROOT = pathlib.Path(__file__).resolve().parent.parent


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--boot-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument(
        "--metrics",
        type=pathlib.Path,
        default=ROOT / "bench-results/m13-server-requests.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=pathlib.Path,
        default=ROOT / "bench-results/m13-prefix-workload-summary.json",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "bench-results/m13-server-workload.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def post_chat(port: int, content: str, timeout: float) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": "deltafin-kimi-k3",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1,
            "stream": False,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
        status = response.status
    return {
        "content": content,
        "status": status,
        "wall_seconds": time.perf_counter() - started,
        "body": body,
    }


def stop_server(process: subprocess.Popen) -> int:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    return int(process.returncode)


def validate_records(records, report):
    if len(records) != 2:
        raise AssertionError(f"expected two server records, got {len(records)}")
    capture, replay = records
    first = capture["prefix_activation"]
    second = replay["prefix_activation"]
    if first.get("action") != "capture" or first.get("hit"):
        raise AssertionError(f"first request did not capture: {first}")
    if second.get("action") != "replay" or not second.get("hit"):
        raise AssertionError(f"second request did not replay: {second}")
    if (
        first.get("total_positions") != 89
        or second.get("total_positions") != 89
    ):
        raise AssertionError("official Hello/World prompts are not both T=89")
    detail = second.get("activation") or {}
    if detail.get("skipped_route_edges") != 108_928:
        raise AssertionError(
            f"unexpected skipped route edges: "
            f"{detail.get('skipped_route_edges')}"
        )
    if detail.get("skipped_unique_experts", 0) <= 0:
        raise AssertionError("replay did not avoid any expert loads")
    if replay.get("logical_expert_bytes", 0) >= capture.get(
        "logical_expert_bytes", 0
    ):
        raise AssertionError("replay logical expert bytes did not decrease")
    final_cache = report["final_cache"]
    if final_cache.get("resident_shapes_lru") != [89]:
        raise AssertionError(f"unexpected final LRU: {final_cache}")
    if final_cache.get("owned_bytes") != 195_198_976:
        raise AssertionError(f"unexpected activation ownership: {final_cache}")
    if report["captures"] != 1 or report["replays"] != 1:
        raise AssertionError("capture/replay aggregate differs")


def main(argv=None) -> int:
    args = parse_args(argv)
    require_profile()
    args.metrics = args.metrics.expanduser().resolve()
    args.summary_output = args.summary_output.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    paths = (args.metrics, args.summary_output, args.output)
    for path in paths:
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{path} exists; pass --overwrite to replace it"
            )
        if path.exists():
            path.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)

    baseline = memory_snapshot("baseline", detailed=True)
    model_dir = pathlib.Path(os.environ["K3_MODEL_DIR"]).resolve()
    tree_before = model_tree_fingerprint(model_dir)
    evidence: dict[str, Any] = {
        "schema": "deltafin.m13-server-workload.v1",
        "status": "running",
        "created_at": now(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "baseline": baseline,
        "official_tree_before": tree_before,
        "official_model_modified": False,
        "model_conversion_performed": False,
        "requests": [],
    }
    write_evidence(args.output, evidence)

    environment = os.environ.copy()
    environment.update(
        {
            "K3_PREFIX_ACTIVATION": "1",
            "K3_PREFIX_ACTIVATION_ENTRIES": "2",
            "K3_PREFIX_STATE": "0",
            "K3_RESPONSE_MEMO_ENTRIES": "0",
            "K3_SERVER_MAX_TOKENS": "1",
            "K3_SERVER_METRICS_JSONL": str(args.metrics),
            "K3_PROFILE": "0",
        }
    )
    command = [
        sys.executable,
        str(ROOT / "tools/serve_openai.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    process = None
    output_lines: deque[str] = deque(maxlen=200)
    pump_thread = None
    ready = threading.Event()
    failure = None
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def pump():
            assert process.stdout is not None
            for line in process.stdout:
                output_lines.append(line.rstrip())
                print(line, end="", flush=True)
                if "Deltafin OpenAI-compatible API on" in line:
                    ready.set()

        pump_thread = threading.Thread(target=pump, daemon=True)
        pump_thread.start()
        deadline = time.monotonic() + args.boot_timeout
        while not ready.wait(timeout=1):
            if process.poll() is not None:
                raise RuntimeError(
                    f"server exited during boot with {process.returncode}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("server boot timed out")

        for content in ("Hello", "World"):
            print(f"[M13] request {content!r}", flush=True)
            response = post_chat(
                args.port, content, args.request_timeout
            )
            if response["status"] != 200:
                raise RuntimeError(f"HTTP status {response['status']}")
            evidence["requests"].append(response)
            write_evidence(args.output, evidence)

        records = load_records([args.metrics])
        report = summarize(records)
        validate_records(records, report)
        args.summary_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        evidence["metrics_records"] = records
        evidence["workload_summary"] = report
        evidence["response_exact"] = (
            evidence["requests"][0]["body"]
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content")
            == evidence["requests"][1]["body"]
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )
        if not evidence["response_exact"]:
            raise AssertionError("Hello/World one-token responses differ")
        tree_after = model_tree_fingerprint(model_dir)
        evidence["official_tree_after"] = tree_after
        if tree_after != tree_before:
            raise AssertionError("official model tree changed")
        evidence["status"] = "passed"
    except BaseException as exc:
        failure = exc
        evidence["status"] = "failed"
        evidence["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        if process is not None:
            evidence["server_returncode"] = stop_server(process)
        if pump_thread is not None:
            pump_thread.join(timeout=5)
        evidence["server_log_tail"] = list(output_lines)
        released = memory_snapshot("released", detailed=True)
        evidence["release"] = {
            "snapshot": released,
            "swap_delta_from_baseline": (
                released["swap"].get("used_bytes", 0)
                - baseline["swap"].get("used_bytes", 0)
            ),
            "server_exited": process is None or process.poll() is not None,
        }
        evidence["release"]["passed"] = (
            evidence["release"]["server_exited"]
            and evidence["release"]["swap_delta_from_baseline"]
            <= 16 * 1024 * 1024
        )
        if evidence["status"] == "passed" and not evidence["release"]["passed"]:
            evidence["status"] = "failed"
            evidence["error"] = {
                "type": "ReleaseValidationError",
                "message": "server did not exit cleanly or swap increased",
            }
        evidence["completed_at"] = now()
        write_evidence(args.output, evidence)

    if failure is not None:
        raise failure
    if evidence["status"] != "passed":
        raise RuntimeError(evidence["error"]["message"])
    print(f"PASS: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
