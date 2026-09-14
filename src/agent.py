"""Core components of the AmazonHelp AI support agent.

Intent classification (3 systems), retrieval-grounded reply drafting (3 systems),
and the auto-handle / escalate policy. The LLM work runs on a local Ollama server
so the whole pipeline reproduces without any paid API key.
"""
import html
import json
import re
import time

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

OLLAMA_URL = "http://localhost:11434/api/chat"
DRAFT_MODEL = "gemma4:e4b"
JUDGE_MODEL = "qwen3.5:4b"

TAXONOMY = {
    "ORDER_STATUS": "asking where an order/package/parcel is, tracking, delivery delays or a missed/failed delivery",
    "DAMAGED_OR_WRONG": "item arrived damaged, broken, defective, counterfeit, wrong or missing item, or stopped working",
    "REFUND": "asking for a refund, refund status/delays, money back, or double charges",
    "RETURN_CANCEL": "how to return an item, cancel an order, exchange, or replacement logistics",
    "PAYMENT": "payment failures, card declines, billing, gift cards, promo codes, cashback, unrecognized charges",
    "ACCOUNT": "login/password problems, hacked or locked accounts, Prime membership issues, account settings",
    "DELIVERY_ADDRESS": "delivery address changes, courier/driver behavior, parcels left unsafe, redelivery",
    "APP_WEBSITE": "bugs or errors on the Amazon app/website, checkout problems",
    "SERVICE_COMPLAINT": "complaints about Amazon support quality, unresponsive agents, chat/call experience",
    "REVIEW_CONTENT": "review moderation, product listings, sellers, pricing on listings",
    "OTHER": "anything else: product questions, feedback, off-topic, promotions, unclear messages",
}
INTENTS = list(TAXONOMY)
HIGH_RISK_INTENTS = {"SERVICE_COMPLAINT", "ACCOUNT", "REFUND", "PAYMENT", "REVIEW_CONTENT"}

# Trivial baseline: first keyword regex that matches wins (order matters).
KEYWORD_RULES = {
    "ORDER_STATUS": r"where.*(order|package|parcel|delivery)|when.*(deliver|arrive)|track|order.*(status|update|late|delay)|(still|yet).*(wait|not.*(arriv|deliver|come))|delivery.*(late|delay|fail|no show|didn)|has.*(delivered|shipped).*(yet|been)|shipped yet",
    "DAMAGED_OR_WRONG": r"damaged|broken|crushed|defective|faulty|broke|stop(ed)? working|not working|(wrong|incorrect|different|another).*(item|product|thing|order|one)|empty box|missing.*(item|part|pieces)|fake|counterfeit|knockoff|used item|not.*original",
    "REFUND": r"\brefund|money back|reimburse|charged twice|double charg|charge.*twice|cancel.*charge.*still|refund.*(status|pending|delay|not)",
    "RETURN_CANCEL": r"\breturn\b|\bcancel\b|cancel.*order|return.*item|exchange|replace.*(item|product)|replacement",
    "PAYMENT": r"gift card|voucher|promo code|coupon|cashback|pay balance|payment.*(method|fail|error|declin)|card.*(declin|fail|error)|billing|charged.*(wrong|twice|amount)|apay",
    "ACCOUNT": r"log ?in|login|password|sign ?in|account.*(hack|compromis|suspend|locked|blocked|close|delet)|prime.*(member|subscription|renew|trial|charg|cancel)|\bprime\b|email.*(change|address)|2fa|two.factor",
    "DELIVERY_ADDRESS": r"address|courier|driver|doorstep|neighbou?r|left.*(porch|door|outside)|redeliver|missed delivery|reschedul|safe place|mail room",
    "APP_WEBSITE": r"\bapp\b|\bwebsite\b|check ?out|search bar|site.*(error|bug|not work|crash|load|down)|app.*(error|bug|not work|crash|load|down)|can.?t.*(order|buy|pay|checkout|search)",
    "SERVICE_COMPLAINT": r"customer service|support team|support agent|\bchat\b.*(rude|useless|no help)|on hold|waiting.*(days|week|hour).*(reply|response)|no (one|body).*(respond|reply|help)|(terrible|awful|worst|pathetic|disgusting|useless).*(service|experience|support)|complaint",
    "REVIEW_CONTENT": r"\breview|listing|seller|product page|profile.*(removed|content)",
}

LEXICON = {
    "profanity": r"\b(f+u+c*k+|shit|sh1t|bitch|bastard|arse|a\*+hole|wtf|bollocks)\b",
    "anger": r"\b(pathetic|disgusting|ridiculous|unacceptable|abysmal|fuming|appall(?:ing|ed)|worst|joke|insult(?:ing|ed)?)\b",
    "fraud_security": r"\b(hack(?:ed|er)?|stole|stolen|fraud|scam|phish|counterfeit|chargeback|identit(?:y|ies))\b",
    "legal": r"\b(lawyer|legal action|sue|suing|police|court|consumer (?:court|forum|rights))\b",
    "churn_threat": r"(cancel.{0,20}(my )?(account|prime|membership)|never (?:buy|order|shop)|done with amazon|last straw|taking my business|shop elsewhere|delete my account|goodbye amazon)",
    "sensitive": r"\b(rape|sexual|death|injur|assault|discriminat)\b",
}
MONEY_RE = r"(\$|rs\.?|inr|\u20b9|\u00a3|eur\s?)\s?\d[\d,.]*|\b\d[\d,.]*\s?(dollars|rupees|pounds|euros)\b"


def clean_text(t):
    """Normalize a raw tweet for modeling: unescape, mask links, drop mentions."""
    t = html.unescape(str(t))
    t = re.sub(r"https?://t\.co/\w+", "<link>", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def looks_english(s, common_frac=0.30):
    words = re.findall(r"[a-z']+", str(s).lower())
    if not words:
        return False
    common = set("a an the is are was were be been to of in on at for with my me you your it this that i have has had do does did not no yes and or but if so as we they he she them his her its from by about into over after before between during without within please help order delivery item package card account refund return cancel prime".split())
    return sum(w in common for w in words) / len(words) >= common_frac


# ---------------------------------------------------------------- intent models

class KeywordClassifier:
    """Trivial baseline: ordered keyword rules over the raw (lowercased) message."""

    def predict(self, texts):
        out = []
        for t in texts:
            t = str(t).lower()
            label = "OTHER"
            for intent, pat in KEYWORD_RULES.items():
                if re.search(pat, t):
                    label = intent
                    break
            out.append(label)
        return np.array(out)

    def predict_proba(self, texts):
        """One-hot confidence (1.0 for the matched rule, 0 elsewhere)."""
        preds = self.predict(texts)
        P = np.zeros((len(texts), len(INTENTS)))
        for i, p in enumerate(preds):
            P[i, INTENTS.index(p)] = 1.0
        return P


class MLIntentClassifier:
    """Simple baseline: TF-IDF + Logistic Regression trained on keyword weak labels."""

    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=50000)
        self.clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")

    def fit(self, texts, weak_labels):
        X = self.vec.fit_transform(texts)
        self.clf.fit(X, weak_labels)
        return self

    def predict(self, texts):
        return self.clf.predict(self.vec.transform(texts))

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.vec.transform(texts))

    @property
    def classes_(self):
        return self.clf.classes_


class LLMIntentClassifier:
    """Agent path: local LLM few-shot classification, batched."""

    def __init__(self, model=DRAFT_MODEL, batch_size=10, url=OLLAMA_URL, max_retries=2):
        self.model = model
        self.batch_size = batch_size
        self.url = url
        self.max_retries = max_retries
        self.sys = (
            "You classify customer support messages sent to AmazonHelp on Twitter.\n"
            "Choose exactly one intent per message.\n"
            + "\n".join(f"- {k}: {v}" for k, v in TAXONOMY.items())
            + '\n\nRespond with ONLY a JSON array, one object per message: '
              '[{"id": 1, "intent": "...", "confidence": 0.0-1.0}]. No other text.'
        )

    def _chat(self, user, num_predict=700):
        payload = {
            "model": self.model, "stream": False, "think": False,
            "messages": [{"role": "system", "content": self.sys}, {"role": "user", "content": user}],
            "options": {"temperature": 0.1, "num_predict": num_predict, "num_ctx": 8192},
        }
        r = requests.post(self.url, json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def predict(self, texts):
        preds, confs = [], []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for bi, i in enumerate(range(0, len(texts), self.batch_size)):
            batch = texts[i:i + self.batch_size]
            user = "\n".join(f"{j+1}. {clean_text(t)}" for j, t in enumerate(batch))
            raw = None
            t0 = time.time()
            for attempt in range(self.max_retries + 1):
                try:
                    raw = self._chat(user)
                    arr = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))
                    assert len(arr) == len(batch)
                    break
                except Exception as e:
                    if attempt == self.max_retries:
                        arr = [{"id": j + 1, "intent": "OTHER", "confidence": 0.2} for j in range(len(batch))]
                    else:
                        print(f"    batch {bi+1} attempt {attempt+1} failed ({type(e).__name__}), retrying", flush=True)
                        time.sleep(2)
            print(f"    intent batch {bi+1}/{n_batches} done ({time.time()-t0:.0f}s)", flush=True)
            for j, item in enumerate(arr):
                preds.append(item.get("intent", "OTHER") if item.get("intent") in INTENTS else "OTHER")
                confs.append(float(item.get("confidence", 0.5)))
        return np.array(preds), np.array(confs)


# ---------------------------------------------------------------- retrieval

class Retriever:
    """TF-IDF nearest-neighbour search over historical (customer message -> brand reply)."""

    def __init__(self, messages, replies):
        keep = ~pd.Index(messages).duplicated()
        self.messages = list(pd.Series(messages)[keep])
        self.replies = list(pd.Series(replies)[keep])
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=80000)
        self.X = self.vec.fit_transform([clean_text(m) for m in self.messages])

    def topk(self, query, k=3):
        q = self.vec.transform([clean_text(query)])
        sims = (self.X @ q.T).toarray().ravel()
        idx = sims.argsort()[::-1][:k]
        return [(self.messages[i], self.replies[i], float(sims[i])) for i in idx]

    def max_sim(self, query):
        q = self.vec.transform([clean_text(query)])
        return float((self.X @ q.T).toarray().max())


# ---------------------------------------------------------------- reply drafting

CANNED_REPLY = (
    "We're sorry for the trouble. Please get in touch with us here: <link> and "
    "we'll be happy to help. ^AG"
)


def retrieval_reply(query, retriever):
    """Simple baseline: return the historical reply most similar to the query."""
    msg, reply, sim = retriever.topk(query, k=1)[0]
    reply = re.sub(r"^@\d+\s*", "", reply).strip()
    return reply, sim


def llm_reply(query, retriever, model=DRAFT_MODEL, url=OLLAMA_URL, k=3):
    """Agent path: draft grounded in the brand's own historical resolutions."""
    exemplars = retriever.topk(query, k=k)
    ex_txt = "\n".join(
        f"Customer: {clean_text(m)}\nAmazonHelp replied: {re.sub(r'^@\\d+\\s*', '', r).strip()}"
        for m, r, _ in exemplars
    )
    sys = (
        "You draft public Twitter replies for AmazonHelp (Amazon customer support). "
        "Rules: 1) At most 2 short sentences. 2) Be warm and apologetic like AmazonHelp. "
        "3) Ground the reply in how AmazonHelp resolved similar cases in the examples: "
        "usually direct the customer to the secure support link <link> or DM, never ask for "
        "order details in public. 4) Do not invent policies or refunds. "
        "5) End with a two-letter agent initial like ^AG. Output only the reply text."
    )
    user = f"Past similar cases:\n{ex_txt}\n\nNew customer message: {clean_text(query)}\n\nDraft the reply:"
    payload = {
        "model": model, "stream": False, "think": False,
        "messages": [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        "options": {"temperature": 0.3, "num_predict": 120},
    }
    for attempt in range(2):
        try:
            r = requests.post(url, json=payload, timeout=600)
            r.raise_for_status()
            out = r.json()["message"]["content"].strip()
            out = re.sub(r"^@\w+\s*", "", out).strip('" ')
            if out:
                return out
        except Exception:
            time.sleep(1)
    return CANNED_REPLY  # graceful fallback if the local server is down


# ---------------------------------------------------------------- escalation policy

def capability_risk_signals(text):
    """Extra capability/risk layer to catch false-negative escalations."""
    reasons = []
    
    clean_t = clean_text(text)
    if len(clean_t.split()) < 4:
        reasons.append("insufficient evidence (message too short to ground a safe reply)")
        
    low = str(text).lower()
    
    # Obfuscated profanity: normalize by dropping non-alpha characters to catch f*u*c*k, f.u.c.k, f u c k
    norm = re.sub(r'[^a-z]', '', low)
    if re.search(r'(fuck|shit|bitch|bastard|asshole|arsehole)', norm):
        reasons.append("obfuscated profanity")
        
    # Capability and risk detection
    if re.search(r'(shows delivered|delivered to someone else|account.*lookup|investigate.*account)', low):
        reasons.append("account lookup required")
        
    if re.search(r'(refund dispute|duplicate charge|charged twice|double charge|payment verification|unauthorized transaction)', low):
        reasons.append("transaction verification required")
        
    if re.search(r'(delivered but not received|package missing|missing package|package lost|lost package|marked delivered|not in my possession)', low):
        reasons.append("delivery dispute")
        
    if re.search(r'\b(stolen|fraud|scam|unauthorized|theft)\b', low):
        reasons.append("loss/fraud")
        
    if re.search(r'\b(hacked|compromised|suspicious activity|identity|security)\b', low):
        reasons.append("identity/security")
        
    if re.search(r'\b(speak to a human|real person|human agent|manager|representative|connect me to a human|someone real)\b', low):
        reasons.append("explicit human request")
        
    return reasons


def escalation_reasons(text, intent, confidence, max_sim,
                        conf_threshold=0.45, sim_threshold=0.12,
                        high_risk=HIGH_RISK_INTENTS, lexicon=LEXICON):
    """A priori policy: any single trigger routes the message to a human."""
    reasons = []
    if not looks_english(text):
        reasons.append("non-English message: out of scope for the English-only agent")
    if confidence < conf_threshold:
        reasons.append(f"low intent confidence ({confidence:.2f} < {conf_threshold})")
    if intent in high_risk:
        reasons.append(f"high-risk intent '{intent}' (money/account/support-complaint lanes)")
    low = str(text).lower()
    for cat, pat in lexicon.items():
        if re.search(pat, low):
            reasons.append(f"lexicon trigger '{cat}'")
    if re.search(MONEY_RE, low):
        reasons.append("specific money amount mentioned")
    if max_sim < sim_threshold:
        reasons.append(f"no close precedent in retrieval history (sim={max_sim:.2f})")
        
    # Apply capability/risk layer
    reasons.extend(capability_risk_signals(text))
    
    return reasons


class SupportAgent:
    """End-to-end agent: classify intent, draft grounded reply, decide route."""

    def __init__(self, intent_system="llm", reply_system="llm_rag", retriever=None):
        self.intent_system = intent_system  # 'keyword' | 'ml' | 'llm'
        self.reply_system = reply_system     # 'canned' | 'retrieval' | 'llm_rag'
        self.retriever = retriever

    def handle_batch(self, texts, llm_clf=None, ml_clf=None):
        if self.intent_system == "llm":
            preds, confs = llm_clf.predict(texts)
        else:
            clf = ml_clf if self.intent_system == "ml" else KeywordClassifier()
            preds = clf.predict(texts)
            P = clf.predict_proba(texts)
            confs = P.max(axis=1) if P.shape[1] else np.zeros(len(texts))
        rows = []
        for t, intent, conf in zip(texts, preds, confs):
            max_sim = self.retriever.max_sim(t) if self.retriever else 1.0
            reasons = escalation_reasons(t, intent, float(conf), max_sim)
            if self.reply_system == "llm_rag":
                reply = llm_reply(t, self.retriever)
            elif self.reply_system == "retrieval":
                reply, _ = retrieval_reply(t, self.retriever)
            else:
                reply = CANNED_REPLY
            rows.append({
                "text": t, "intent": intent, "confidence": float(conf),
                "reply": reply, "max_sim": max_sim,
                "route": "human" if reasons else "auto",
                "reasons": "; ".join(reasons) if reasons else "standard flow: safe to auto-handle",
            })
        return pd.DataFrame(rows)
