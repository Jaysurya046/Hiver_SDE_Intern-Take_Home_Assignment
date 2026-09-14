# Golden set: sampling and labelling protocol

**File:** `golden_set.csv` — 200 examples, columns `ex_id, conv_id, text, brand_reply,
intent, route`.

## Sampling

1. Pool = all 21,423 reconstructed AmazonHelp conversations, reduced to their
   **first-inbound message** (the turn the agent actually serves).
2. English filter: `>= 30%` of alphabetic tokens in a 200-word common-word list.
   (Chosen after inspecting the multilingual mix of the corpus — Hindi/Portuguese/
   Spanish tweets are common because of Amazon's regional support lanes.)
3. Uniform random sample of 200 with `random_state=42`, **conversations disjoint from
   the retrieval index and the weak-label training pool** (leakage control: a retrieved
   golden reply would make reply evaluation meaningless).
4. Stratification: none — a uniform sample was deliberately kept so the class
   distribution reflects real traffic (ORDER_STATUS-heavy, REVIEW_CONTENT rare).

## Labelling

- **Intent:** the 11-class taxonomy in `src/agent.py` (`TAXONOMY`). Weak keyword labels
  pre-filled the sheet; every example was then **read and corrected by hand**.
  Contested calls and their resolutions:
  - "Where is my refund" → REFUND (not ORDER_STATUS); "where is my order, it's 5 days
    late" → ORDER_STATUS.
  - Double charges / "you stole my money" → REFUND with route human (money at stake).
  - Cashback/promo status questions → PAYMENT, usually auto (informational).
  - "Courier left parcel in the rain" → DELIVERY_ADDRESS (driver behaviour lane).
  - Complaints about a *product* (not the service) → OTHER or DAMAGED_OR_WRONG, not
    SERVICE_COMPLAINT (that class is about Amazon *support* quality).
- **Route:** fixed rubric, written before any system output was seen:
  - **human** if: money/fraud/account-access involved, account hacked/locked, complaint
    about support, anger/profanity, legal/churn threat, non-English, or the issue needs
    looking up in the customer's account.
  - **auto** if: routine status/ETA/how-to questions a safe template answer handles.

## Corrections applied

- 10 pre-sampled items (original `ex_id`s 0, 34, 57, 66, 94, 106, 131, 171, 182, 194)
  were Spanish/Portuguese/Italian/German despite passing the ASCII-based English filter
  (e.g. "necesito comprar 3 tablets samsung..."). They were replaced with English
  alternates drawn from the same pool under `eng_score > 0.35`; both the swap and the
  filter lesson are documented in the report's failure analysis.
- The final set was checked conversation-by-conversation for duplicates
  (same `conv_id` twice): none.

## Distribution

| intent | n | route auto | route human |
|---|---|---|---|
| ORDER_STATUS | 42 | 30 | 12 |
| OTHER | 29 | 25 | 4 |
| DELIVERY_ADDRESS | 29 | 15 | 14 |
| ACCOUNT | 20 | 6 | 14 |
| SERVICE_COMPLAINT | 17 | 2 | 15 |
| PAYMENT | 16 | 6 | 10 |
| DAMAGED_OR_WRONG | 13 | 5 | 8 |
| RETURN_CANCEL | 13 | 8 | 5 |
| REFUND | 12 | 4 | 8 |
| APP_WEBSITE | 5 | 4 | 1 |
| REVIEW_CONTENT | 4 | 1 | 3 |
| **total** | **200** | **106** | **94** |
