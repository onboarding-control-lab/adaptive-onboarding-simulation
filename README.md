# Controlled Financial Onboarding Simulation: Reproducibility Package

This package provides the research simulation codebase, frozen model artefacts, configuration schemas, and aggregate experimental evaluation results for evaluating adaptive synthetic identity generation against layered onboarding controls. It is a controlled experimental simulation and not an operational fraud evasion tool or production decisioning system.

### Requirements & Environment
- **Python:** Python 3.10 or higher (tested on Python 3.11)

### Included in this package
- Portable source code and pipeline execution modules
- Frozen detector artefacts (first-line XGBoost model and secondary consistency layer reference)
- Experimental protocol, feature governance, and reference pool configurations
- Published aggregate evaluation result tables and run metadata
- Command-line interface for offline integrity verification and result inspection

### Deliberately excluded
- Raw Bank Account Fraud (BAF) benchmark dataset records
- LLM API credentials, keys, and private environment files
- Intermediate private development traces and raw diagnostic logs

### Installation

```bash
python3 -m pip install -r requirements.txt
```

### Offline Inspection Commands

- **Structural and cryptographic integrity check:**
  ```bash
  python3 cli.py verify
  ```
- **Inspect aggregate evaluation results:**
  ```bash
  python3 cli.py inspect-results
  ```

### Guarded Live Execution

The `run-final` entry point is guarded. Executing live experiments requires externally supplied BAF dataset files and an active `DEEPSEEK_API_KEY`. Live execution is not required to inspect or verify the published package.

### Scientific Record

This repository serves as a reproducibility and verification companion. The accompanying academic dissertation manuscript remains the authoritative scientific record.
