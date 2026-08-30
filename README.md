# Sahayak Health AI — Capstone Delivery Package

**Sahayak Health AI** (सहायक, "aid") — AI triage assistant for India's frontline health workers (ASHA/ANM).
Built on Google ADK + hermes3:8b (Ollama) / Gemini Flash.

## Folder structure

```
Sahayak_Capstone/
├── learner/          ← YOUR WORK: fill in the four notebooks + sahayak_starter.py
├── src/              ← provided Python modules (demo, eval, tools) — do not submit
├── data/             ← cases.db + raw JSONL dataset
├── tests/            ← pytest harness — run after Week 3 to self-check
├── docs/             ← capstone documents (problem, workflow, task brief)
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
cd src && python setup_db.py           # creates cases.db (if not present)
jupyter notebook ../learner/week1_adk_intro.ipynb
```

> Notebooks use `sys.path.insert(0, '../src')` — run from the `learner/` directory,
> or add the `src/` path to your PYTHONPATH.

## Document index

| File | Purpose |
|------|---------|
| docs/LEARNER_TASK_BRIEF.md | **Start here** — what to build, week by week |
| docs/CAPSTONE_PROBLEM_STATEMENT.md | Problem context, dataset, evaluation criteria |

## Learner notebooks

| Notebook | Mode | What the learner does |
|----------|------|----------------------|
| week1_adk_intro.ipynb | Taught | Run + observe ADK primitives, answer 5 reflection questions |
| week2_starter.ipynb | Scaffolded | EDA, triage mapping, policy baseline, FunctionTools exhibit |
| week3_starter.ipynb | Semi-scaffolded | Write 6 agent instructions, wire pipeline, run 20-case eval |
| week4_starter.ipynb | Open-ended | 50-case eval, failure analysis, implement 1 improvement, demo |

The `tests/` suite (12 tests) runs from the package root via `pytest tests/`.
Run it after Week 3 to verify your safety evaluator implementation.
