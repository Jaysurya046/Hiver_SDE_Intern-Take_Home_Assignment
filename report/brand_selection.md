# Brand selection: why @AmazonHelp

All 6 major support brands in `twcs.csv`, reconstructing conversations with the same
algorithm (first-inbound tweet mentioning the brand, alternating customer/brand turns,
earliest reply per hop, depth cap 30):

| brand | outbound tweets | conversations | thread tweets | median turns | tweets/conv |
|---|---|---|---|---|---|
| AppleSupport | 106,860 | 49,261 | 132,999 | 2 | 2.7 |
| **AmazonHelp** | **169,840** | **21,423** | **78,686** | **3** | **3.7** |
| Delta | 42,253 | 24,134 | 63,801 | 2 | 2.6 |
| Uber_Support | 56,270 | 18,877 | 50,501 | 2 | 2.7 |
| Tesco | 38,573 | 14,769 | 42,421 | 2 | 2.9 |
| SpotifyCares | 43,265 | 12,687 | 35,298 | 2 | 2.8 |

(computed from `cache_all_inbound+cand.parquet`; reproduce with `scripts/rebuild_corpus.py`
adapted per brand)

**Decision: AmazonHelp.**

1. **Reply volume:** highest outbound count of any support brand (169,840).
2. **Real dialogue, not deflection:** it is the only brand whose median conversation
   runs 3 turns. AppleSupport — the obvious volume pick — has a median of 2: a single
   privacy-deflection ("DM us") and done. For an agent whose job is drafting replies,
   the brand's replies must contain something worth imitating.
3. **Resolution content:** qualitative review of AmazonHelp replies shows a mix of
   link-deflection (the modal archetype, ~used for account-specific issues), refund
   guidance, delivery instructions, and follow-up questions — and 254 distinct agents
   sign replies with `^XX` sigils (96.7% of its 36,411 in-thread replies), giving a learnable brand voice.
4. **Known failure mode accepted:** Amazon's support lanes include heavy Indian-English
   traffic (place names, "kindly do the needful"), which stresses the English filter —
   kept deliberately because it's realistic.
