"""Shape-stable routed-expert activation capture for K3.

The session retains only fixed-prefix latent inputs, routes, weights, and
routed-expert outputs. Replay keeps the original full position dimension, uses
empty Metal route rows for the prefix, computes only suffix experts, and then
splices the captured prefix output back into its original rows.
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any

import torch


class PrefixActivationSession:
    def __init__(self, prefix_tokens: int, total_positions: int):
        prefix_tokens = int(prefix_tokens)
        total_positions = int(total_positions)
        if not 0 < prefix_tokens < total_positions:
            raise ValueError(
                "prefix_tokens must be within the full position dimension: "
                f"{prefix_tokens}/{total_positions}"
            )
        self.prefix_tokens = prefix_tokens
        self.total_positions = total_positions
        self.mode = "capture"
        self._inputs: dict[int, torch.Tensor] = {}
        self._routes: dict[int, tuple[tuple[int, ...], ...]] = {}
        self._weights: dict[int, tuple[tuple[float, ...], ...]] = {}
        self._outputs: dict[int, torch.Tensor] = {}
        self._capture_layers: list[int] = []
        self._replay_layers: list[int] = []
        self.input_validation_seconds = 0.0
        self.full_unique_experts = 0
        self.demand_unique_experts = 0
        self.skipped_route_edges = 0

    @staticmethod
    def _route_tuple(rows) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(int(expert) for expert in row) for row in rows
        )

    @staticmethod
    def _weight_tuple(rows) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(float(weight) for weight in row) for row in rows
        )

    def _validate_shape(self, layer: int, x: torch.Tensor, rows, weights):
        if x.ndim != 2 or x.shape[0] != self.total_positions:
            raise ValueError(
                f"layer {layer}: expected latent input "
                f"[{self.total_positions}, hidden], got {tuple(x.shape)}"
            )
        if len(rows) != self.total_positions or len(weights) != len(rows):
            raise ValueError(
                f"layer {layer}: route rows do not match position dimension"
            )
        if any(len(ids) != len(ws) for ids, ws in zip(rows, weights)):
            raise ValueError(
                f"layer {layer}: route ID/weight lengths differ"
            )

    def prepare(
        self,
        layer: int,
        x: torch.Tensor,
        rows,
        weights,
    ) -> int:
        """Validate/capture prefix inputs and return replayed row count."""
        layer = int(layer)
        self._validate_shape(layer, x, rows, weights)
        routes = self._route_tuple(rows[: self.prefix_tokens])
        weight_rows = self._weight_tuple(weights[: self.prefix_tokens])
        started = time.perf_counter()
        prefix_input = (
            x[: self.prefix_tokens]
            .detach()
            .to(device="cpu", dtype=x.dtype, copy=True)
            .contiguous()
        )
        self.input_validation_seconds += time.perf_counter() - started
        if self.mode == "capture":
            if layer in self._inputs:
                raise RuntimeError(f"layer {layer}: duplicate activation capture")
            self._inputs[layer] = prefix_input
            self._routes[layer] = routes
            self._weights[layer] = weight_rows
            self._capture_layers.append(layer)
            return 0
        if self.mode != "replay":
            raise RuntimeError(f"prefix activation session is {self.mode}")
        if layer not in self._inputs or layer not in self._outputs:
            raise RuntimeError(f"layer {layer}: activation was not captured")
        expected_input = self._inputs[layer]
        if not torch.equal(prefix_input, expected_input):
            maximum = float(
                (prefix_input.to(torch.float32)
                 - expected_input.to(torch.float32)).abs().max()
            )
            raise RuntimeError(
                f"layer {layer}: prefix latent input changed "
                f"(max_abs={maximum})"
            )
        if routes != self._routes[layer]:
            raise RuntimeError(f"layer {layer}: prefix routes changed")
        if weight_rows != self._weights[layer]:
            raise RuntimeError(f"layer {layer}: prefix route weights changed")
        self._replay_layers.append(layer)
        return self.prefix_tokens

    def record_demand(
        self,
        full_ids,
        demand_ids,
        *,
        route_edges_skipped: int,
    ) -> None:
        self.full_unique_experts += len(full_ids)
        self.demand_unique_experts += len(demand_ids)
        self.skipped_route_edges += int(route_edges_skipped)

    def finish(self, layer: int, output: torch.Tensor) -> torch.Tensor:
        layer = int(layer)
        expected = (self.total_positions, output.shape[-1])
        if output.ndim != 2 or tuple(output.shape) != expected:
            raise ValueError(
                f"layer {layer}: expected expert output {expected}, "
                f"got {tuple(output.shape)}"
            )
        if self.mode == "capture":
            self._outputs[layer] = (
                output[: self.prefix_tokens].detach().clone()
            )
            return output
        if self.mode != "replay":
            raise RuntimeError(f"prefix activation session is {self.mode}")
        cached = self._outputs.get(layer)
        if cached is None:
            raise RuntimeError(f"layer {layer}: prefix output was not captured")
        if cached.shape != output[: self.prefix_tokens].shape:
            raise RuntimeError(f"layer {layer}: cached output shape changed")
        return torch.cat(
            (cached, output[self.prefix_tokens :]),
            dim=0,
        )

    def arm_replay(self, *, expected_layers: int | None = None) -> None:
        if self.mode != "capture":
            raise RuntimeError(f"cannot arm replay from mode {self.mode}")
        if expected_layers is not None and len(self._outputs) != expected_layers:
            raise RuntimeError(
                f"captured {len(self._outputs)}/{expected_layers} layers"
            )
        if set(self._inputs) != set(self._outputs):
            raise RuntimeError("captured input/output layer sets differ")
        self.mode = "replay"
        self.begin_replay()

    def begin_replay(self) -> None:
        """Reset per-pass counters before reusing an armed session."""
        if self.mode != "replay":
            raise RuntimeError(f"cannot begin replay from mode {self.mode}")
        self._replay_layers.clear()
        self.full_unique_experts = 0
        self.demand_unique_experts = 0
        self.skipped_route_edges = 0

    def finish_replay(self, *, expected_layers: int | None = None) -> None:
        if self.mode != "replay":
            raise RuntimeError(f"cannot finish replay from mode {self.mode}")
        if expected_layers is not None and len(self._replay_layers) != expected_layers:
            raise RuntimeError(
                f"replayed {len(self._replay_layers)}/{expected_layers} layers"
            )

    def snapshot(self) -> dict[str, Any]:
        input_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self._inputs.values()
        )
        output_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in self._outputs.values()
        )
        return {
            "mode": self.mode,
            "prefix_tokens": self.prefix_tokens,
            "total_positions": self.total_positions,
            "captured_layers": len(self._outputs),
            "replayed_layers": len(self._replay_layers),
            "input_validation_bytes": input_bytes,
            "cached_output_bytes": output_bytes,
            "owned_bytes": input_bytes + output_bytes,
            "input_validation_seconds": self.input_validation_seconds,
            "full_unique_experts": self.full_unique_experts,
            "demand_unique_experts": self.demand_unique_experts,
            "skipped_unique_experts": (
                self.full_unique_experts - self.demand_unique_experts
            ),
            "skipped_route_edges": self.skipped_route_edges,
        }

    def value_digest(self) -> str:
        """Hash retained input/output values for focused mutation checks."""
        digest = hashlib.sha256()
        for kind, tensors in (
            ("input", self._inputs),
            ("output", self._outputs),
        ):
            for layer, tensor in sorted(tensors.items()):
                cpu = tensor.detach().contiguous().to("cpu")
                digest.update(kind.encode())
                digest.update(str(layer).encode())
                digest.update(str(cpu.dtype).encode())
                digest.update(str(tuple(cpu.shape)).encode())
                digest.update(cpu.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def close(self) -> None:
        self.mode = "closed"
        self._inputs.clear()
        self._routes.clear()
        self._weights.clear()
        self._outputs.clear()
        self._capture_layers.clear()
        self._replay_layers.clear()


class PrefixActivationCache:
    """Bounded, shape-keyed cache for the server's fixed chat prefix.

    Sessions are keyed by the complete monolithic position dimension. This is
    deliberately stricter than a token-prefix cache: K3 must see the same full
    shape that produced the retained activations.
    """

    def __init__(self, prefix_token_ids, max_entries: int = 2):
        self.prefix_token_ids = tuple(int(token) for token in prefix_token_ids)
        self.max_entries = int(max_entries)
        if not self.prefix_token_ids:
            raise ValueError("prefix_token_ids must not be empty")
        if self.max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._sessions: OrderedDict[int, PrefixActivationSession] = (
            OrderedDict()
        )
        self.builds = 0
        self.hits = 0
        self.evictions = 0

    def preview(self, mode: str, token_ids) -> dict[str, Any]:
        tokens = tuple(int(token) for token in token_ids)
        eligible = (
            mode == "chat"
            and len(tokens) > len(self.prefix_token_ids)
            and tokens[: len(self.prefix_token_ids)] == self.prefix_token_ids
        )
        total = len(tokens) if eligible else 0
        return {
            "enabled": True,
            "eligible": eligible,
            "used": False,
            "hit": bool(eligible and total in self._sessions),
            "action": (
                "replay"
                if eligible and total in self._sessions
                else "capture" if eligible else "none"
            ),
            "prefix_tokens": len(self.prefix_token_ids) if eligible else 0,
            "total_positions": total,
            "entries": len(self._sessions),
            "max_entries": self.max_entries,
        }

    def begin(
        self, mode: str, token_ids
    ) -> tuple[PrefixActivationSession | None, dict[str, Any]]:
        plan = self.preview(mode, token_ids)
        if not plan["eligible"]:
            return None, plan
        total = plan["total_positions"]
        session = self._sessions.pop(total, None)
        if session is None:
            session = PrefixActivationSession(
                len(self.prefix_token_ids), total
            )
        else:
            session.begin_replay()
            self._sessions[total] = session
            self.hits += 1
        plan["used"] = True
        plan["hit"] = session.mode == "replay"
        plan["action"] = session.mode
        return session, plan

    def complete(
        self,
        session: PrefixActivationSession,
        plan: dict[str, Any],
        *,
        expected_layers: int,
    ) -> dict[str, Any]:
        if plan["action"] == "capture":
            session.arm_replay(expected_layers=expected_layers)
            total = int(plan["total_positions"])
            replaced = self._sessions.pop(total, None)
            if replaced is not None and replaced is not session:
                replaced.close()
            self._sessions[total] = session
            self.builds += 1
            while len(self._sessions) > self.max_entries:
                _shape, evicted = self._sessions.popitem(last=False)
                evicted.close()
                self.evictions += 1
        elif plan["action"] == "replay":
            session.finish_replay(expected_layers=expected_layers)
        else:
            raise RuntimeError(f"invalid activation action {plan['action']}")
        plan.update(
            {
                "entries": len(self._sessions),
                "builds": self.builds,
                "hits": self.hits,
                "evictions": self.evictions,
                "activation": session.snapshot(),
            }
        )
        return plan

    def abort(
        self,
        session: PrefixActivationSession | None,
        plan: dict[str, Any],
    ) -> None:
        if session is None:
            return
        if plan.get("action") == "capture":
            session.close()
            return
        if plan.get("action") == "replay":
            total = int(plan["total_positions"])
            if self._sessions.get(total) is session:
                self._sessions.pop(total)
                session.close()

    def close(self) -> None:
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()
