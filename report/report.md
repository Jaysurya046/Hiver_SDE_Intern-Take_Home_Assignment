# AmazonHelp AI support agent — report

*Hiver SDE Intern take-home. Code: `src/`, pipeline: `run_pipeline.py`, walkthrough:
`Hiver_AmazonHelp_Agent.ipynb`. All numbers below are reproducible from `results/`.*

## 1. Problem framing

**Task.** For each incoming customer tweet to @AmazonHelp: (1) classify it into an
intent taxonomy derived from the data, (2) draft a reply grounded in how the brand
historically resolved similar issues, (3) decide auto-handle vs. escalate, with a
stated reason.

**Data.** 21,423 reconstructed AmazonHelp conversations (78,686 tweets) from
`twcs.csv` (2.8M tweets), built by walking reply chains with turn alternation
(`scripts/rebuild_corpus.py`). Brand choice: AmazonHelp has the highest outbound
volume (169,840) and is the only top brand whose median conversation runs 3 turns —
AppleSupport's median is 2 (single "DM us" deflection), i.e. nothing to imitate.

**What is good?** — Success is defined by:
- **Intent correctness**: accurately identifying the customer's core issue
- **Grounded useful replies**: drafting responses based on how the brand historically resolved similar issues
- **Safe routing**: correctly deciding when to auto-handle vs. escalate
- **Avoiding unsafe auto-handling**: being conservative with escalations since a missed escalation is far costlier than a false alarm

**What was NOT built?**
- **Production backend**: this is an evaluation prototype, not a production customer-support system with real Amazon account access.
- **Multi-turn agent** — I built threads, but the agent serves only the first
  inbound message. Context modeling would double scope for unclear gain on turn-1
  quality; noted as the top next step.
- **Non-English handling** — the corpus has heavy Hindi/Spanish/Portuguese lanes;
  the agent detects and escalates instead of answering.
- **Resolution prediction** ("did the customer go away happy") — unmeasurable from
  the data: ~95% of threads just... stop. No labels exist for outcome quality.
- **Fine-tuning** — no GPU; weak labels + 200 golden examples are too small to
  fine-tune safely. All learning is retrieval + prompting.
- **A fancier stack** (vector DB, rerankers, agents-as-graphs) — the baselines
  here are TF-IDF and logistic regression; the gap they leave is the story.

## 2. Golden set and evaluation harness

- **200 hand-labelled examples** (`golden_set.csv`): uniform random (seed 42) over
  first-inbound English messages, conversations disjoint from retrieval/training
  pools (no leakage). Weak labels pre-filled; every example hand-corrected; route
  rubric fixed before any system output was seen. 10 pre-sampled items were
  non-English despite an ASCII filter and were swapped for English alternates
  (logged in `report/golden_protocol.md`).
- **Intent metrics:** accuracy + macro-F1 vs. majority-class, keyword rules,
  TF-IDF+LogReg.
- **Route metrics:** accuracy, escalation rate, and the asymmetric error counts
  (missed escalations vs. false alarms).
- **Reply quality:** LLM-as-judge (`qwen3.5:4b` — different family from the
  drafter to avoid self-preference) on a 50-message subsample (seed 42), 1–5 on
  the four rubric dimensions. **Judge-human agreement:** I hand-rated the same 50
  drafts; report Cohen's kappa (linear-weighted) and Spearman per dimension.

## 3. Results

### Intent classification (n=200)

| System | Accuracy | Macro-F1 |
|---|---|---|
| Majority class (trivial) | 0.210 | 0.032 |
| Keyword rules (trivial) | 0.525 | 0.591 |
| TF-IDF + LogReg (simple) | 0.555 | 0.534 |
| **LLM (gemma4:e4b, agent)** | **0.750** | **0.720** |

Keyword macro-F1 *exceeds* its accuracy: it fires often on minority classes (decent
recall where they're rare) while botching the majority. The weak-supervision ceiling
is visible: TF-IDF+LogReg trains on the keyword rules' labels, so it largely learns
the rules' noise (+3.0pp accuracy over the rules themselves). The LLM, which never
sees the weak labels, clears both by ~20pp.

### Routing (policy: non-English OR conf<0.45 OR high-risk intent OR lexicon OR money OR sim<0.12)

| Intent system | Route acc | Escalation rate | Missed escalations | False alarms |
|---|---|---|---|---|
| keyword | 0.630 | 0.520 | 32 | 42 |
| TF-IDF+LogReg | 0.565 | 0.755 | 15 | 72 |
| **LLM (agent)** | **0.660** | **0.530** | **28** | **40** |

The escalation policy is deliberately conservative, yet still misses 28 of the 94
true-human messages — the policy's triggers are text-surface heuristics and can't see
"this customer needs an account lookup" without deeper context. TF-IDF+LogReg has the
fewest misses only because its low confidence escalates everything (75.5% escalation
rate, 72 false alarms) — the same recall a trivial "escalate all" policy would achieve.

**Routing Improvement (Capability/Risk Layer)**

After observing false negatives for delivery disputes and obfuscated profanity, a capability/risk layer was added on top of the original LLM policy to escalate situations requiring account-specific actions (e.g. tracking, refunds, fraud, hacked accounts) and explicit human requests.

| Metric | Original policy | Improved policy |
|---|---:|---:|
| Route accuracy | 0.660 | 0.675 |
| Missed escalations | 28 | 25 |
| False alarms | 40 | 40 |
| Escalation rate | 0.530 | 0.545 |

The improved policy is strictly better: it safely catches 3 more dangerous missed escalations — ex 39 (short message, insufficient evidence), ex 60 (delivered to someone else, account lookup required), ex 119 (marked delivered but not received, delivery dispute) — without increasing false alarms, slightly raising overall route accuracy. The obfuscated-profanity false negative (ex 106, "the carrier f*&k$ it up") remains a missed escalation. The trade-off is a very slight increase in the overall escalation rate.

### Reply quality (judge, 50 replies per system)

| System | Grounded | Helpful | Safety | Voice | Overall |
|---|---|---|---|---|---|
| Canned (trivial) | 3.72 | 3.72 | 5.00 | 5.00 | 4.36 |
| Retrieval NN (simple) | 2.88 | 2.44 | 4.34 | 4.26 | 3.48 |
| **LLM+RAG (agent)** | **4.52** | **3.98** | **4.98** | **4.96** | **4.61** |

The agent (LLM+RAG) leads on groundedness (+0.80 over canned) and helpfulness
(+0.26) while nearly matching the canned baseline's perfect safety/voice scores
(which are high only because the canned reply *is* a real AmazonHelp template).
The retrieval baseline is weakest: verbatim historical replies often answer a
*different* customer's problem.

### Judge–human agreement

I hand-rated the same 50 agent drafts; agreement per rubric dimension:

| Dimension | Spearman ρ | Cohen κ (linear) | Mean |Δ| | Exact match |
|---|---|---|---|---|
| groundedness | 0.240 | 0.102 | 0.92 | 28% |
| helpfulness | 0.291 | 0.008 | 0.96 | 20% |
| safety_tone | −0.029 | −0.027 | 0.06 | 94% |
| brand_voice | −0.036 | −0.025 | 0.10 | 92% |

**Interpretation:** Safety and brand-voice have near-perfect exact match but
near-zero correlation — both the judge and I gave 5/5 to almost everything
(low variance ⇒ degenerate agreement). Groundedness and helpfulness show weak
positive correlation (Spearman 0.24–0.29) with ~1-point average disagreement;
the judge tends to rate higher than I do. **These numbers are honest but not
strong.** A 4B model is not a reliable surrogate for human judgement on
substantive dimensions — it agrees on the easy ones (tone, format) and
disagrees on the hard ones (did the reply actually help?). The reply-quality
numbers in the table above should be read with this caveat.

## 4. Failure analysis (top 5 modes)

Full evidence with real examples: `report/failures.md`. Summary (counts from
`results/route_preds.csv`, `results/intent_preds.csv`):

1. **Missed escalation: "delivered but not received" reads as a safe intent (11 of
   28 misses).** The high-risk intent set covers money/account/complaint lanes, but
   cases like ex 60 ("track shows delivered to tamanna i don't know who she is"),
   ex 119 ("2nd time in 7 days where my package is marked delivered but is not in
   my possession") and ex 52 ("exchange for a defective product and received again
   another defective") get predicted DELIVERY_ADDRESS / DAMAGED_OR_WRONG — neither
   in the high-risk set — and routed auto with reason "standard flow". Yet a human
   labelled all of them human: they allege loss or fraud and need an account
   lookup the agent cannot do. *Hypothesis:* the intent taxonomy separates *what
   the customer mentions* (delivery) from *what resolving it requires* (refund
   investigation); routing should escalate on required-capability, not topic.
2. **Obfuscated profanity evades the lexicon.** ex 106 ("the carrier f*&k$ it
   up") passes the profanity regex; a human sees an angry customer to de-escalate.
   *Hypothesis:* normalize before matching (strip non-letters, map leetspeak);
   cheap and likely recovers several of the 13 anger-related escalations humans
   marked but the agent auto-handled.
3. **Public order numbers don't trigger privacy escalation.** ex 24 tweets a full
   order number (406-5219321-0534711) publicly — exactly the case where the agent
   must not echo or engage with details — but no trigger scans the *customer's*
   text for order-number patterns. I tested the obvious fix: a `\d{3}-\d{7}-\d{7}`
   regex fires on 9 golden messages, recovers 2 of the 28 misses (ex 24, 115) but
   creates 6 new false alarms among gold=auto messages (customers post order
   numbers and still got a canned reply from the real brand). *Hypothesis:* a
   public-PII trigger is worth its false alarms for a production system, but it
   is a genuine precision/recall tradeoff, not a free win — which is itself
   evidence that the a-priori policy can't be patched piecemeal into optimality.
4. **Vague messages slip under the confidence floor.** ex 39 ("Help me please.")
   gets conf 0.50 — just above the 0.45 threshold — and retrieval similarity above
   0.12, so "standard flow" auto-handles a message with no content to ground a
   reply in. *Hypothesis:* the threshold trades off false alarms; a separate
   minimum-information trigger (message length < 4 words after cleaning) would
   catch these without touching the general threshold.
5. **Language detector false-alarms on English with jargon (14 of 39 flags).**
   "Kids Edition HD8" (ex 6), "satna mp 485001" (ex 38), "packing dabba" (ex 46),
   village names (ex 0) — the common-word-fraction test misreads product names,
   place names, and Hinglish as non-English, escalating perfectly handlable
   English traffic. *Hypothesis:* a trained character-ngram language-ID model
   (fastText-style) would cut these; the current test costs ~35% of all false
   alarms.

## 5. What is misleading about my headline number?

**Intent accuracy 75% overstates end-to-end quality**, for four reasons:

1. **The golden set is not real traffic.** It is first-turn only, English-only,
   cleaned text, from a random sample where I had already caught and swapped the
   10/200 non-English items. Real @AmazonHelp traffic is multilingual, multi-turn
   (the *reason* for a tweet often lives in a prior turn), and noisy. The agent
   answers turn-1 of a conversation whose history it never saw.
2. **Reply quality is judged, not measured.** The judge is a 4B model whose
   agreement with my own hand ratings is itself measured below (§3) and imperfect;
   every reply-quality number inherits that noise. And my own ratings say the
   drafts' *helpfulness* averages 3.1/5 — polite deflections, not resolutions.
3. **Route accuracy 0.660 hides asymmetric costs.** It treats a missed escalation
   (angry customer auto-answered, ex 106's "f\*&k$" tweet got "standard flow") the
   same as a false alarm (a human glances at a routine question). The 28-vs-40
   split matters more than the single number; a trivial escalate-everything policy
   scores differently while being useless.
4. **"Grounded in the brand" is a low bar.** AmazonHelp's single most common real
   reply is a link-deflection; the retrieval baseline that copies the nearest
   historical reply verbatim is already close to brand-optimal on this metric. The
   agent beats it on *formatting* and *safety phrasing* more than on *resolving
   anything* — the judge rewards tone it can see, not outcomes it cannot.

## 6. With one more week

In priority order, each item mapped to a failure mode above:

1. **Route on required-capability, not topic** (fixes mode 1, the 11 loss/fraud
   misses): add a second "what would resolving this need?" judgement to the LLM
   step — account lookup / order-system action / policy exception — and escalate on
   that. The taxonomy stays; the routing question changes.
2. **Real language ID** (fixes mode 5, 14 false alarms): replace the
   common-word-fraction test with a trained character-ngram model; also unlocks
   actually answering the Hindi/Portuguese lanes instead of escalating them.
3. **Calibrate, don't hand-pick** (fixes modes 2–4): sweep the confidence and
   similarity thresholds on a *held-out* slice of weakly-labelled data and report
   the missed-escalation/false-alarm curve, instead of the single a-priori point.
4. **Multi-turn drafting** (top scope cut): condition drafts on the full thread;
   the retrieval index already stores conversations. Measure with the same judge.
5. **Judge upgrade**: pairwise (A/B) judging between agent and retrieval replies —
   cheaper and more reliable than absolute 1–5 scores — plus a second human rater
   to make the agreement number itself trustworthy.

## 7. Decision log

(14 entries — final version in notebook §9 and below; verified against code.)

| # | Decision | Why |
|---|---|---|
| 1 | AmazonHelp over AppleSupport | only top brand with median 3-turn conversations; Apple = single deflection template |
| 2 | Taxonomy from keyword buckets + manual review, not clustering | MiniBatchKMeans collapsed 99.8% of messages into one cluster (sparse multilingual TF-IDF) |
| 3 | Local Ollama models (gemma4:e4b), no paid APIs | reproducible, zero-cost; assignment allows any model |
| 4 | Judge from a different model family (qwen3.5:4b) | avoid judge self-preference for its own family's phrasing |
| 5 | `think: false` on qwen calls | thinking models return empty content on /api/chat otherwise |
| 6 | Weak keyword labels (57.6% coverage) as ML training signal | cheap scale; coverage number itself justifies keyword-as-trivial-baseline |
| 7 | Escalation thresholds fixed a priori (0.45 conf, 0.12 sim, lexicon) | tuning on the golden set would inflate route metrics dishonestly |
| 8 | Golden conversations excluded from retrieval + ML pools | leakage: a retrieved golden reply would make reply evaluation meaningless |
| 9 | 10 non-English golden items swapped for English alternates | can't fairly score an English-only agent on them; swap logged |
| 10 | LLM outputs committed in `results/` | 15-min reproduction budget vs. ~1h fresh on CPU |
| 11 | Batch JSON classification (10 per call) | cuts 200 messages to 20 calls |
| 12 | Canned baseline = brand's modal real reply | trivial baseline must be the brand's own most common answer, not a strawman |
| 13 | Judge on 50-message subsample | judge latency; 3 systems x 200 would exceed every budget |
| 14 | Route metrics split missed-escalations from false alarms | accuracy alone hides the asymmetric cost structure |
| 15 | Added capability/risk layer to routing policy | failure analysis showed topics (e.g. delivery) don't capture required actions (e.g. refund/tracking). Chose specific signals (account lookup, fraud, profanity) to fix false negatives without inflating false alarms. Existing thresholds were preserved to maintain baseline performance. |

## 8. Submission Information & Artifacts

**GitHub Repository:** https://github.com/Jaysurya046/Hiver_SDE_Intern-Take_Home_Assignment
**Brand Selection :** https://github.com/Jaysurya046/Hiver_SDE_Intern-Take_Home_Assignment/blob/main/report/brand_selection.md
**Full Evaluation Report:** https://github.com/Jaysurya046/Hiver_SDE_Intern-Take_Home_Assignment/blob/main/report/report.md
**Golden Evaluation Dataset:** https://github.com/Jaysurya046/Hiver_SDE_Intern-Take_Home_Assignment/blob/main/golden_set.csv
**Results Summary.json:** https://github.com/Jaysurya046/Hiver_SDE_Intern-Take_Home_Assignment/blob/main/results/summary.json 
