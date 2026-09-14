"""Tests for the escalation / routing policy.

Representative cases for each trigger, plus a normal safe request.
No Ollama or network needed — all tests use escalation_reasons() directly.
"""
from agent import escalation_reasons


# Helper: standard params for tests where we control only the text
def reasons(text, intent="ORDER_STATUS", confidence=0.9, max_sim=0.4):
    return escalation_reasons(text, intent, confidence, max_sim)


# ------- Normal safe request should auto-handle
def test_normal_safe_request_auto():
    r = reasons("Where is my order? Tracking says in transit.", "ORDER_STATUS", 0.9, 0.4)
    assert r == [], f"Expected auto, got reasons: {r}"


# ------- Non-English detection
def test_non_english_escalates():
    r = reasons("necesito comprar 3 tablets samsung galaxy tab")
    assert any("non-English" in x for x in r)


# ------- Low confidence escalates
def test_low_confidence_escalates():
    r = reasons("something something", "OTHER", 0.3, 0.4)
    assert any("confidence" in x for x in r)


# ------- High-risk intents escalate
def test_high_risk_intent_service_complaint():
    r = reasons("your support is terrible", "SERVICE_COMPLAINT", 0.9, 0.4)
    assert any("high-risk" in x for x in r)


def test_high_risk_intent_account():
    r = reasons("my account is locked", "ACCOUNT", 0.9, 0.4)
    assert any("high-risk" in x for x in r)


def test_high_risk_intent_refund():
    r = reasons("I need my refund", "REFUND", 0.9, 0.4)
    assert any("high-risk" in x for x in r)


def test_high_risk_intent_payment():
    r = reasons("payment failed", "PAYMENT", 0.9, 0.4)
    assert any("high-risk" in x for x in r)


# ------- Lexicon triggers
def test_profanity_escalates():
    r = reasons("what the fuck is wrong with my order", "ORDER_STATUS", 0.9, 0.4)
    assert any("profanity" in x for x in r)


def test_anger_lexicon_escalates():
    r = reasons("this is absolutely disgusting and pathetic", "ORDER_STATUS", 0.9, 0.4)
    assert any("anger" in x for x in r)


def test_fraud_lexicon_escalates():
    r = reasons("someone hacked my account and stole my card info", "ORDER_STATUS", 0.9, 0.4)
    assert any("fraud" in x for x in r)


def test_legal_threat_escalates():
    r = reasons("I am going to talk to my lawyer about this", "ORDER_STATUS", 0.9, 0.4)
    assert any("legal" in x for x in r)


def test_churn_threat_escalates():
    r = reasons("I am done with amazon, never buying from you again", "ORDER_STATUS", 0.9, 0.4)
    assert any("churn" in x for x in r)


# ------- Money amount mentioned
def test_money_amount_escalates():
    r = reasons("I was charged $45.99 twice for this order", "ORDER_STATUS", 0.9, 0.4)
    assert any("money" in x for x in r)


# ------- Low retrieval similarity
def test_low_retrieval_similarity_escalates():
    r = reasons("Some random unusual question nobody asked before", "OTHER", 0.9, 0.01)
    assert any("retrieval" in x.lower() or "precedent" in x.lower() for x in r)


# ------- Auto-handlable intents with good confidence and similarity
def test_order_status_normal_is_auto():
    assert reasons("when will my package arrive?", "ORDER_STATUS", 0.9, 0.5) == []


def test_delivery_address_normal_is_auto():
    assert reasons("can I change my delivery address?", "DELIVERY_ADDRESS", 0.9, 0.5) == []


def test_damaged_normal_is_auto():
    assert reasons("the item looks a bit scratched", "DAMAGED_OR_WRONG", 0.9, 0.5) == []


# ------- Capability/Risk layer tests
def test_explicit_human_request_escalates():
    r = reasons("I want to speak to a human about this", "ORDER_STATUS", 0.9, 0.5)
    assert any("explicit human request" in x for x in r)


def test_account_action_escalates():
    r = reasons("can you investigate my account please", "ACCOUNT", 0.9, 0.5)
    assert any("account lookup" in x for x in r)


def test_fraud_loss_escalates():
    r = reasons("my package was stolen from my porch", "DELIVERY_ADDRESS", 0.9, 0.5)
    assert any("loss/fraud" in x or "stolen" in x for x in r)


def test_delivered_but_not_received_escalates():
    r = reasons("tracking shows delivered but not received yet", "ORDER_STATUS", 0.9, 0.5)
    assert any("delivery dispute" in x for x in r)


def test_obfuscated_profanity_escalates():
    r = reasons("this is f*u*c*k*i*n*g unbelievable", "ORDER_STATUS", 0.9, 0.5)
    assert any("profanity" in x for x in r)

def test_insufficient_evidence_escalates():
    r = reasons("hi", "OTHER", 0.9, 0.5)
    assert any("insufficient evidence" in x for x in r)
