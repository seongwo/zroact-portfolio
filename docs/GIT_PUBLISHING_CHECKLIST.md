# Git Publishing Checklist

Use this checklist before creating a public or portfolio repository.

## 1. Confirm This Directory Is A Repo

The current local folder may not be initialized as git yet.

```bash
cd /home/capstone2/zroact-stage2
git status
```

If it says this is not a git repository:

```bash
git init
```

## 2. Add Only Source, Config, Prompts, And Docs

Recommended first add:

```bash
git add \
  .gitignore \
  README.md \
  docs \
  requirements.txt \
  benchmark2/configs \
  benchmark2/labels \
  benchmark2/prompts \
  benchmark2/scripts \
  benchmark2/training/README.md \
  benchmark2/training/IMPLEMENTATION_REPORT_KO.md \
  benchmark2/training/V2_RESULTS_REPORT_KO.md \
  benchmark2/training/requirements.txt \
  benchmark2/training/configs \
  benchmark2/training/scripts \
  pipeline \
  pipeline_ver2 \
  serving
```

## 3. Inspect Before Commit

```bash
git status --short
```

No path should include:

```text
benchmark2/data/
benchmark2/models/
benchmark2/runs/
benchmark2/training/datasets/
benchmark2/training/outputs/
serving/jobs/
unsloth_compiled_cache/
*.safetensors
*.pth
*.pt
```

## 4. Check Large Files

```bash
git ls-files | xargs -r du -h | sort -h | tail -30
```

If any file is unexpectedly large, unstage it:

```bash
git restore --staged path/to/file
```

## 5. Suggested First Commit

```bash
git commit -m "Document ZroAct stage2 pipeline and training workflow"
```

## 6. Optional: Add A Small Demo Later

A good public repo can include a tiny non-private mock example:

```text
examples/
├── mock_stage1_labels.json
├── mock_manifest.jsonl
└── README.md
```

Avoid adding real CCTV frames unless they are explicitly cleared for public use.
