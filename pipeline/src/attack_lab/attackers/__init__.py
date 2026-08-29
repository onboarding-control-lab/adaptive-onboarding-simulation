"""Attacker implementations (A0–A3).

Import concrete modules explicitly::

    from attack_lab.attackers.a0_random import ConstrainedRandomAttacker
    from attack_lab.attackers.a1_planner import OneShotLLMPlanner
    from attack_lab.attackers.a2_search import SurrogateGuidedSearcher
    from attack_lab.attackers.a3_agent import EpisodicLLMAgent

Package-level convenience re-exports also exist on ``attack_lab`` itself
(e.g. ``from attack_lab import EpisodicLLMAgent``).

This package ``__init__`` intentionally performs no eager imports, to avoid
circular load-order issues among A0–A3. There are no top-level shim modules
and no compatibility aliases under ``attack_lab.a0_random`` (etc.).
"""

__all__: list[str] = []
