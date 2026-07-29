"""Online, byte-bounded route policy for direct expert-slab prefetch."""

from __future__ import annotations

import math


class AdaptiveRoutePrefetch:
    """Learn which previous-token router ranks persist into the next token.

    Every actual route supplies counterfactual observations for all previous
    ranks, including ranks that were not prefetched. Prediction uses a
    conservative Wilson lower bound, so early lucky hits cannot immediately
    spend the full byte budget.
    """

    def __init__(
        self,
        *,
        max_experts_per_layer: int,
        expert_bytes: int,
        token_budget_bytes: int,
        warmup_observations: int,
        min_wilson_precision: float,
        confidence_z: float = 1.96,
    ):
        if max_experts_per_layer <= 0:
            raise ValueError("max_experts_per_layer must be positive")
        if expert_bytes <= 0 or token_budget_bytes < 0:
            raise ValueError("byte sizes must be non-negative")
        if warmup_observations <= 0:
            raise ValueError("warmup_observations must be positive")
        if not 0.0 <= min_wilson_precision <= 1.0:
            raise ValueError("min_wilson_precision must be in [0, 1]")
        if confidence_z <= 0:
            raise ValueError("confidence_z must be positive")
        self.max_experts_per_layer = max_experts_per_layer
        self.expert_bytes = expert_bytes
        self.token_budget_bytes = token_budget_bytes
        self.warmup_observations = warmup_observations
        self.min_wilson_precision = min_wilson_precision
        self.confidence_z = confidence_z
        self.reset()

    def reset(self) -> None:
        width = self.max_experts_per_layer
        self._rank_hits = [0] * width
        self._rank_observations = [0] * width
        self._previous: dict[int, tuple[int, ...]] = {}
        self._issued_by_layer: dict[int, tuple[int, ...]] = {}
        self._token_prefetch_bytes = 0
        self._passes = 0
        self._transitions = 0
        self._observed_layers = 0
        self._predicted_experts = 0
        self._predicted_hits = 0

    @staticmethod
    def _wilson_lower(hits: int, observations: int, z: float) -> float:
        if observations == 0:
            return 0.0
        probability = hits / observations
        z2 = z * z
        denominator = 1.0 + z2 / observations
        center = probability + z2 / (2.0 * observations)
        radius = z * math.sqrt(
            (
                probability * (1.0 - probability)
                + z2 / (4.0 * observations)
            )
            / observations
        )
        return (center - radius) / denominator

    def begin_pass(self, previous: dict[int, tuple[int, ...]]) -> None:
        self._previous = {
            int(layer): tuple(int(expert) for expert in ranking)
            for layer, ranking in previous.items()
        }
        self._issued_by_layer.clear()
        self._token_prefetch_bytes = 0
        self._passes += 1
        if self._previous:
            self._transitions += 1

    def eligible_prefix(self) -> int:
        eligible = 0
        for hits, observations in zip(
            self._rank_hits, self._rank_observations
        ):
            if observations < self.warmup_observations:
                break
            lower = self._wilson_lower(
                hits, observations, self.confidence_z
            )
            if lower < self.min_wilson_precision:
                break
            eligible += 1
        return eligible

    def predict(self, layer: int) -> tuple[int, ...]:
        ranking = self._previous.get(int(layer), ())
        if not ranking or self.token_budget_bytes == 0:
            return ()
        remaining_bytes = (
            self.token_budget_bytes - self._token_prefetch_bytes
        )
        budget_experts = remaining_bytes // self.expert_bytes
        count = min(
            self.eligible_prefix(),
            self.max_experts_per_layer,
            len(ranking),
            budget_experts,
        )
        predicted = ranking[:count]
        if predicted:
            self._issued_by_layer[int(layer)] = predicted
            added = len(predicted) * self.expert_bytes
            self._token_prefetch_bytes += added
            self._predicted_experts += len(predicted)
        return predicted

    def observe(self, layer: int, actual_experts) -> None:
        layer = int(layer)
        actual = {int(expert) for expert in actual_experts}
        ranking = self._previous.get(layer, ())
        if ranking:
            self._observed_layers += 1
            for rank, expert in enumerate(
                ranking[:self.max_experts_per_layer]
            ):
                self._rank_observations[rank] += 1
                self._rank_hits[rank] += int(expert in actual)
        issued = self._issued_by_layer.pop(layer, ())
        self._predicted_hits += sum(expert in actual for expert in issued)

    def snapshot(self) -> dict[str, object]:
        observations = list(self._rank_observations)
        hits = list(self._rank_hits)
        precision = [
            hit / count if count else 0.0
            for hit, count in zip(hits, observations)
        ]
        wilson = [
            self._wilson_lower(hit, count, self.confidence_z)
            for hit, count in zip(hits, observations)
        ]
        misses = self._predicted_experts - self._predicted_hits
        return {
            "passes": self._passes,
            "transitions": self._transitions,
            "observed_layers": self._observed_layers,
            "eligible_prefix": self.eligible_prefix(),
            "rank_hits": hits,
            "rank_observations": observations,
            "rank_precision": precision,
            "rank_wilson_lower": wilson,
            "predicted_experts": self._predicted_experts,
            "predicted_hits": self._predicted_hits,
            "predicted_misses": misses,
            "predicted_precision": (
                self._predicted_hits / self._predicted_experts
                if self._predicted_experts else 0.0
            ),
            "current_token_prefetch_bytes": self._token_prefetch_bytes,
            "token_budget_bytes": self.token_budget_bytes,
        }
