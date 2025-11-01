# NetworkSecurity

NetworkSecurity is a Python project that applies machine learning to detect network threats (phishing, suspicious connections, anomalies). The repository contains data ingestion, preprocessing, model training/inference pipelines, utilities, and example notebooks for experimentation.

## Repository layout
- Network_Data/ — datasets (e.g., phisingData.csv)
- networksecurity/ — main package
  - cloud/ — cloud helpers
  - components/ — models and transformers
  - constant/ — configs and constants
  - entity/ — dataclasses and domain objects
  - exception/ — custom exceptions
  - logging/ — logging utilities
  - pipeline/ — training and inference pipelines
  - utils/ — helper functions
- notebooks/ — exploratory notebooks
- requirements.txt
- Dockerfile
- .github/workflows/ — CI

## Quick start (Windows)
1. Create and activate a venv:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1   # PowerShell
```