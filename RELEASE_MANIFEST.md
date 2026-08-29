# Public Release Manifest

## Package Scope
This package contains the reproducible research code, frozen detection artefacts, experimental configurations, and aggregate published results for the simulation study on synthetic identity adaptation against layered onboarding controls.

## Package Contents
1. **Source Code (`pipeline/src/`, `cli.py`):**
   - Simulation harness, attacker implementations (A0 baseline, A1 planner, A2 searcher, A3 agent).
   - Defence layers: First-line statistical detector ($D_1$ XGBoost) and secondary consistency review ($D_2\text{-S}$).
   - Guarded evaluation runner and offline integrity checking tools.
2. **Frozen Configurations (`config/`):**
   - Experimental evaluation protocol (`final_month7_protocol.json`).
   - Feature handling and attacker feature governance definitions.
   - Reference pool selection parameters and dry-run fixtures.
3. **Model & Defence Artefacts (`artifacts/`):**
   - Primary $D_1$ XGBoost pipeline and Month-6 operating threshold selection.
   - Primary $D_2\text{-S}$ pairwise consistency reference and empirical review thresholds.
   - Prespecified optional secondary exploratory model ($D_2\text{-S}$ v1.1 Isolation Forest).
4. **Published Results (`results/`):**
   - Aggregate evaluation metrics and status manifests.
   - Outcome tables for first-line defence ($D_1$ ASR), layered defence (E2E bypass), paired bootstrap confidence intervals, and review capacity sensitivity.

## Deliberately Excluded Items
- **Raw BAF Benchmark Data:** The underlying dataset (`Base.csv`) is governed by third-party licensing and is not distributed in this repository.
- **API Credentials:** No live API keys, tokens, or environment secrets are included.
- **Private Development Logs:** Scratch exploration logs, raw intermediate traces, and operational attack prompts are excluded.

## Offline Inspection & Verification
The package is designed for immediate offline inspection and cryptographic verification without external dependencies (requires Python 3.10+):

```bash
python3 cli.py verify
python3 cli.py inspect-results
```

## Live Reproduction Boundary
Full end-to-end re-execution of the live evaluation pipeline requires externally supplied BAF dataset records and an active DeepSeek API key. Live execution is guarded and is not required for offline verification of the published artefacts and results.
