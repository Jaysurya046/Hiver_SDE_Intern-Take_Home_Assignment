# Hiver Support Agent

## Overview

Hiver SDE Intern take-home: build one AI support agent for **@AmazonHelp** that, for
every incoming customer message:

1. **Classifies** it into an 11-intent taxonomy derived from the data,
2. **Drafts a reply** grounded in how AmazonHelp historically answered similar messages,
3. **Decides** auto-handle vs. escalate-to-human, with a stated reason.

LLM inference runs fully locally via Ollama. `gemma4:e4b` is used for intent
classification and LLM+RAG reply generation; `qwen3.5:4b` (a different model family,
to avoid judge self-preference) is used as the reply-quality judge. Cached evaluation
artifacts in `results/` let the headline results be reproduced without rerunning every
LLM call, and the `tests/` suite needs no Ollama or network access.

## Dataset & Brand

- **Source:** Kaggle's Twitter Customer Support dataset (`twcs.csv`, ~2.8M tweets).
- **Brand:** AmazonHelp was chosen because it has the highest outbound volume
  (169,840) and is the only top brand whose median conversation runs 3 turns —
  AppleSupport's median is 2 (a single "DM us" deflection), i.e. nothing to imitate.
  Full comparison: `report/brand_selection.md`.
- **Corpus:** `data/amazonhelp_threads.csv` (15MB, committed) reconstructs 21,423
  AmazonHelp conversations (78,686 tweets) from Kaggle's `twcs.csv` by walking reply
  chains with turn alternation (`scripts/rebuild_corpus.py`). This is the project's
  reconstructed/canonical corpus; the underlying `twcs.csv` dataset remains an
  external resource (Kaggle), not a product of this repository.

You do **not** need the 516MB `twcs.csv`. To rebuild the corpus from it anyway:
`python scripts/rebuild_corpus.py /path/to/twcs.csv` (~2 min).

## Architecture

The agent is built from three interchangeable systems per task:

| Task | Baselines / alternates | Agent path |
|---|---|---|
| Intent | majority class, keyword rules, TF-IDF + Logistic Regression | LLM few-shot (`gemma4:e4b`) |
| Reply | canned template, nearest-neighbour retrieval | LLM + RAG, grounded in retrieved historical replies |
| Routing | policy over keyword / ML / LLM intents | policy over LLM intent + confidence + capability/risk layer |

The escalation policy is a set of **a priori** triggers: non-English, low intent
confidence, high-risk intent, lexicon signals (profanity, anger, fraud, legal,
churn, sensitive), explicit money amounts, no close retrieval precedent, and a
capability/risk layer for account lookups, delivery disputes, loss/fraud,
identity/security, explicit human requests, and obfuscated profanity.

Pipeline: `run_pipeline.py` runs prep → intent → route → drafts → judge → agreement
→ summary. Full detail: `report/report.md`.

## Intent Taxonomy

The 11 intent classes (defined in `src/agent.py`, `TAXONOMY`):

| Intent | Meaning |
|---|---|
| `ORDER_STATUS` | where is my order / tracking / delivery delays |
| `DAMAGED_OR_WRONG` | damaged, defective, wrong or missing item |
| `REFUND` | refund request, status, or double charges |
| `RETURN_CANCEL` | returns, exchanges, cancellations |
| `PAYMENT` | payment failures, gift cards, promos, unrecognized charges |
| `ACCOUNT` | login, hacked/locked accounts, Prime membership |
| `DELIVERY_ADDRESS` | address changes, courier behaviour, redelivery |
| `APP_WEBSITE` | bugs/errors on the app or website |
| `SERVICE_COMPLAINT` | complaints about Amazon support quality |
| `REVIEW_CONTENT` | review moderation, listings, sellers |
| `OTHER` | everything else |

Intents marked high-risk for routing: `SERVICE_COMPLAINT`, `ACCOUNT`, `REFUND`,
`PAYMENT`, `REVIEW_CONTENT`.

## Evaluation

**Golden set** (`golden_set.csv`): uniform random sample (seed 42) of 200
first-inbound English messages from non-overlapping conversations, **excluding**
conversations used elsewhere in the pipeline (no retrieval/leakage). Weak keyword
labels pre-filled a spreadsheet; every example was then read and corrected by hand
for intent and route. 10 pre-sampled items that were non-English despite the ASCII
filter were replaced with English alternates (IDs in `report/golden_protocol.md`).
Route labels followed a rubric fixed before any system output was seen
(money/fraud/account/complaint/anger/non-English ⇒ human).

**Metrics:**
- Intent: accuracy + macro-F1 vs. the three baselines.
- Routing: accuracy, escalation rate, and the asymmetric error counts — misses
  (true human auto-handled) vs. false alarms.
- Reply quality: LLM-as-judge on a 50-message subsample (seed 42), 1–5 on four
  rubric dimensions, plus judge-human agreement (Cohen's kappa, Spearman).

The escalation thresholds (confidence 0.45, similarity 0.12, lexicon triggers)
were fixed **a priori** — not tuned on the golden set.

## Results

### Intent classification (n=200)

| System | Accuracy | Macro-F1 |
|---|---|---|
| Majority class (trivial) | 0.210 | 0.032 |
| Keyword rules (trivial) | 0.525 | 0.591 |
| TF-IDF + LogReg (simple) | 0.555 | 0.534 |
| **LLM gemma4:e4b (agent)** | **0.750** | **0.720** |

### Routing (LLM intent)

| Policy | Route acc | Missed escalations | False alarms | Escalation rate |
|---|---|---|---|---|
| Original | 0.660 | 28 | 40 | 0.530 |
| Improved (capability/risk layer) | 0.675 | 25 | 40 | 0.545 |

Of 94 true-human messages, 28 were missed under the original policy; the improved
policy safely recovers 3 without adding false alarms.

### Reply quality (judge, 50 replies per system, overall)

| System | Overall |
|---|---|
| Canned (trivial) | 4.36 |
| Retrieval NN (simple) | 3.48 |
| **LLM+RAG (agent)** | **4.61** |

Judge-human agreement is honest but weak on the substantive dimensions (Spearman
≈0.24–0.29 for groundedness/helpfulness; near-degenerate on safety/voice where both
rate almost everything 5/5). Full tables: `Hiver_AmazonHelp_Agent.ipynb` §5, §7 and
`report/report.md`.

## Repository Structure

```
Hiver_AmazonHelp_Agent.ipynb   narrative walkthrough (primary deliverable)
run_pipeline.py                staged pipeline: prep -> intent -> route -> drafts -> judge -> agreement
src/agent.py                   classifiers (keyword/ML/LLM), retrieval, RAG drafter, escalation policy
src/evaluate.py                metrics, LLM-as-judge, judge-human agreement
data/amazonhelp_threads.csv    reconstructed AmazonHelp conversations (committed)
golden_set.csv                 200 hand-labelled examples (intent + auto/human route)
results/                       committed LLM outputs + metrics (intent_preds, route_preds,
                               drafts_50, judge_scores, hand_ratings, agreement, summary.json)
tests/                         lightweight pytest suite (no Ollama/network needed)
scripts/rebuild_corpus.py      twcs.csv -> data/amazonhelp_threads.csv
scripts/make_golden.py         how the 200-example golden set was compiled (labels + swaps)
scripts/build_notebook.py      regenerates the notebook from results/ (nbformat)
report/report.md               the 6-page report
report/golden_protocol.md      how the golden set was sampled and labelled
report/brand_selection.md      why AmazonHelp (numbers for 6 candidate brands)
report/failures.md             raw failure evidence (confusions, misses, false alarms)
requirements-core.txt          minimal runtime deps (no Jupyter)
requirements.txt               full deps including notebook
```

## Setup

```bash
pip install -r requirements-core.txt   # lightweight: no Jupyter overhead
ollama pull gemma4:e4b && ollama pull qwen3.5:4b   # ~13GB total, one-time
```

## Reproducing Results

```bash
python run_pipeline.py                 # cached mode (typically well under the 15-minute
                                       # reproduction budget): retrains ML models, loads
                                       # committed LLM outputs from results/
python run_pipeline.py --fresh         # fresh mode — regenerates every LLM call; may take
                                       # substantially longer depending on hardware/model config
python run_pipeline.py --sample 30     # judge on 30-message subsample (default 50)
python -m pytest tests/ -v             # 59 tests, no Ollama/network needed
```

The default mode genuinely re-runs every fast stage (corpus prep, weak labelling,
TF-IDF+LogReg training, retrieval, routing policy, all metrics) and only loads the
LLM-inference outputs, which are committed in `results/` for the 15-minute budget.

The notebook — `Hiver_AmazonHelp_Agent.ipynb` — is the narrative walkthrough
(problem framing, data, golden set, agent, results, failure analysis, decision
log). It runs entirely off the committed corpus and cached results.

## Limitations

- **First-turn, English-only evaluation.** The agent only serves the first inbound
  message; real @AmazonHelp traffic is multilingual, multi-turn, and noisy. The
  corpus has heavy Hindi/Spanish/Portuguese lanes, which the agent detects and
  escalates instead of answering.
- **Reply quality is judged, not measured.** A 4B judge is not a reliable surrogate
  for human judgement; its agreement with the author's hand ratings is itself weak
  on the substantive dimensions.
- **Route accuracy hides asymmetric costs.** Accuracy treats a missed escalation
  (an angry customer auto-answered) the same as a false alarm (a human glances at
  a routine question); the miss/false-alarm split matters more than the single number.
- **No fine-tuning and no outcome labels.** Weak labels + 200 golden examples are
  too small to fine-tune safely; ~95% of threads just stop, so resolution quality
  is unmeasurable from the data.

## Future Work

- Route on required capability (account lookup, order-system action) instead of topic.
- Replace the common-word-fraction language filter with a trained character-ngram
  language-ID model.
- Calibrate the confidence/similarity thresholds on held-out weakly-labelled data
  instead of the single a-priori point.
- Multi-turn reply drafting, conditioned on the full thread.
- Pairwise (A/B) judging instead of absolute 1–5 scores, plus a second human rater.

## References

- Kaggle: *Customer Support on Twitter* (`twcs.csv`) —
  https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
- Ollama (local LLM serving) — https://ollama.com/
- Project deep-dive: `report/report.md`, golden protocol: `report/golden_protocol.md`,
  brand selection: `report/brand_selection.md`, failure evidence: `report/failures.md`.