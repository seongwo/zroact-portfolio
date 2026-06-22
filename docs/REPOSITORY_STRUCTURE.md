# Repository Structure

This document describes the intended public-facing shape of the project.

The local workspace contains data, model weights, checkpoints, and generated outputs. The git repository should keep only source code, configuration, prompts, examples, and documentation.

## Keep In Git

```text
zroact-stage2/
├── README.md
├── docs/
├── requirements.txt
├── benchmark2/
│   ├── configs/
│   ├── labels/
│   ├── prompts/
│   ├── scripts/
│   └── training/
│       ├── README.md
│       ├── IMPLEMENTATION_REPORT_KO.md
│       ├── V2_RESULTS_REPORT_KO.md
│       ├── configs/
│       ├── scripts/
│       └── requirements.txt
├── pipeline/
├── pipeline_ver2/
└── serving/
```

## Exclude From Git

```text
benchmark2/data/                 # CCTV frame datasets
benchmark2/models/               # Qwen model weights
benchmark2/runs/                 # runtime/generated pipeline runs
benchmark2/results/              # generated benchmark outputs
benchmark2/training/datasets/    # generated JSONL training datasets
benchmark2/training/outputs/     # LoRA checkpoints and evaluation outputs
serving/jobs/                    # uploaded videos and runtime job outputs
unsloth_compiled_cache/          # generated Unsloth cache
```

## Why This Split

The useful portfolio signal is the engineering work:

- how the data is transformed,
- how Stage 1 and Stage 2 are connected,
- how the model is trained and evaluated,
- how the backend can call the system,
- what limitations were found.

The large binary assets are necessary for local execution, but they make the repository difficult to review and impossible to clone casually.

## Suggested Public Repo Sections

| Section | Reader Question It Answers |
|---|---|
| `README.md` | What is this project and why does it matter? |
| `docs/ARCHITECTURE.md` | How does the system work end to end? |
| `docs/EXPERIMENTS_SUMMARY.md` | What did the experiments show? |
| `docs/SETUP_AND_RUN.md` | How would someone run or adapt it? |
| `docs/PORTFOLIO_NOTES.md` | What work did I personally contribute? |
