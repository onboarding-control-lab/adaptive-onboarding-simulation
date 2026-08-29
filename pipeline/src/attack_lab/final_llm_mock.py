"""Deterministic mock of the external DeepSeek transport only.

This client never opens a network socket. It inspects the already-built
production A1/A3 prompt payload and returns schema-valid JSON so the real
attacker classes, parsers, provenance checks and Q/m/K gates can run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from attack_lab.attackers.a1_planner import LLMCompletion


class MockTransportError(TimeoutError):
    """Transport-layer failure that must not charge Q."""


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            payload, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(payload, dict):
            objects.append(payload)
        index = end
    return objects


def _usable_slots(
    slots: Sequence[Mapping[str, Any]],
    allowed_choice_ids: set[str] | None,
) -> list[tuple[str, list[str]]]:
    usable: list[tuple[str, list[str]]] = []
    for slot in slots:
        slot_id = str(slot.get("action_slot_id") or "")
        raw_ids = slot.get("allowed_choice_ids") or []
        choice_ids = [str(cid) for cid in raw_ids]
        if allowed_choice_ids:
            choice_ids = [cid for cid in choice_ids if cid in allowed_choice_ids]
        if slot_id and choice_ids:
            usable.append((slot_id, choice_ids))
    return usable


def build_a1_plan_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    q_max = int((payload.get("budget") or {}).get("q_max") or 5)
    plans = list(payload.get("static_plan_options") or [])
    if not plans:
        raise ValueError("A1 mock cannot plan without static_plan_options.")
    plan = next(
        (item for item in plans if int(item.get("residual_m") or 0) >= 1),
        plans[0],
    )
    residual = max(1, int(plan.get("residual_m") or 1))
    allowed = {str(x) for x in (plan.get("allowed_query_choice_ids") or [])}
    usable = _usable_slots(list(payload.get("action_slots") or []), allowed or None)
    if not usable:
        usable = _usable_slots(list(payload.get("action_slots") or []), None)
    if not usable:
        raise ValueError("A1 mock found no usable action slots.")
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    cursor = 0
    while len(candidates) < q_max:
        slot_id, choice_ids = usable[cursor % len(usable)]
        choice = choice_ids[cursor % len(choice_ids)]
        selections = {slot_id: choice}
        if residual >= 2 and len(usable) > 1:
            other_id, other_choices = usable[(cursor + 1) % len(usable)]
            if other_id != slot_id:
                selections[other_id] = other_choices[0]
        key = tuple(sorted(selections.items()))
        cursor += 1
        if key in seen:
            if cursor > q_max * 20:
                break
            continue
        seen.add(key)
        candidates.append(
            {
                "strategy_label": f"rehearsal_{len(candidates) + 1}",
                "selections": selections,
            }
        )
    if len(candidates) != q_max:
        raise ValueError("A1 mock could not build a unique Q-length plan.")
    return {"static_plan_id": str(plan["static_plan_id"]), "candidates": candidates}


def build_a1_repair_response(
    payload: Mapping[str, Any],
    feedback: Mapping[str, Any],
) -> dict[str, Any]:
    indices = [int(x) for x in (feedback.get("invalid_candidate_indices") or [])]
    residual = max(1, int(feedback.get("residual_m") or 1))
    allowed = None
    plans = list(payload.get("static_plan_options") or [])
    if plans:
        allowed = {str(x) for x in (plans[0].get("allowed_query_choice_ids") or [])}
    usable = _usable_slots(list(payload.get("action_slots") or []), allowed)
    if not usable:
        usable = _usable_slots(list(payload.get("action_slots") or []), None)
    replacements = []
    for offset, index in enumerate(indices or [1]):
        slot_id, choice_ids = usable[offset % len(usable)]
        selections = {slot_id: choice_ids[offset % len(choice_ids)]}
        if residual >= 2 and len(usable) > 1:
            other_id, other_choices = usable[(offset + 1) % len(usable)]
            if other_id != slot_id:
                selections[other_id] = other_choices[0]
        replacements.append(
            {
                "candidate_index": int(index),
                "strategy_label": f"repair_{index}",
                "selections": selections,
            }
        )
    return {"replacements": replacements}


def build_a3_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    budget = payload.get("budget") or {}
    query_index = int(budget.get("query_index") or 1)
    residual = max(1, int(budget.get("residual_m") or 1))
    slots = list(payload.get("action_slots") or payload.get("episode_action_slot_map") or [])
    usable = _usable_slots(slots, None)
    if not usable:
        raise ValueError("A3 mock found no usable action slots.")
    slot_id, choice_ids = usable[(query_index - 1) % len(usable)]
    selections = {slot_id: choice_ids[0]}
    if residual >= 2 and len(usable) > 1:
        other_id, other_choices = usable[query_index % len(usable)]
        if other_id != slot_id:
            selections[other_id] = other_choices[0]
    mode = "INITIALIZE" if query_index <= 1 else "REVISE"
    return {
        "reflection_update": {
            "mode": mode,
            "hypothesis": f"rehearsal query {query_index} uses reference-backed slots",
        },
        "strategy_label": f"rehearsal_q{query_index}",
        "selections": selections,
    }


def build_a3_repair_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    budget = payload.get("budget") or {}
    residual = max(1, int(budget.get("residual_m") or 1))
    slots = list(payload.get("action_slots") or payload.get("episode_action_slot_map") or [])
    usable = _usable_slots(slots, None)
    slot_id, choice_ids = usable[0]
    selections = {slot_id: choice_ids[0]}
    if residual >= 2 and len(usable) > 1:
        selections[usable[1][0]] = usable[1][1][0]
    return {"selections": selections}


@dataclass
class SchemaValidMockLLM:
    """External-transport double for A1/A3 rehearsal. No live API."""

    fail_first_n: int = 0
    calls: list[dict[str, Any]] = field(default_factory=list)
    transport_errors: int = 0
    responses_emitted: int = 0

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout_seconds: float,
        thinking_disabled: bool,
        reasoning_effort: str | None = None,
    ) -> LLMCompletion:
        self.calls.append(
            {
                "model": model,
                "thinking_disabled": thinking_disabled,
                "reasoning_effort": reasoning_effort,
                "n_messages": len(messages),
            }
        )
        if self.transport_errors < int(self.fail_first_n):
            self.transport_errors += 1
            raise MockTransportError("rehearsal transport timeout")
        user = ""
        for item in messages:
            if item.get("role") == "user":
                user = str(item.get("content") or "")
        objects = extract_json_objects(user)
        if not objects:
            raise ValueError("Mock LLM received no JSON planning context.")
        payload = objects[0]
        feedback = objects[-1] if len(objects) > 1 else {}
        if "return ONLY a JSON object with key 'replacements'" in user:
            body = build_a1_repair_response(payload, feedback)
        elif payload.get("local_repair") or (
            "repair_output_schema" in payload and "static_plan_options" not in payload
        ):
            body = build_a3_repair_response(payload)
        elif payload.get("static_plan_options"):
            body = build_a1_plan_response(payload)
        else:
            body = build_a3_response(payload)
        self.responses_emitted += 1
        return LLMCompletion(
            text=json.dumps(body, sort_keys=True),
            model=model,
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            cached_tokens=0,
            latency_ms=1.0,
            thinking_disabled=thinking_disabled,
        )


__all__ = [
    "MockTransportError",
    "SchemaValidMockLLM",
    "build_a1_plan_response",
    "build_a3_response",
    "extract_json_objects",
]
