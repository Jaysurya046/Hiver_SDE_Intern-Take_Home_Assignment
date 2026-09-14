"""Tests for data schema: corpus and golden set structure."""


def test_corpus_columns(threads_df):
    expected = {"conv_id", "depth", "tweet_id", "author_id", "inbound", "created_at", "text"}
    assert expected.issubset(set(threads_df.columns)), f"Missing: {expected - set(threads_df.columns)}"


def test_corpus_has_inbound_and_outbound(threads_df):
    assert threads_df.inbound.any(), "No inbound tweets found"
    assert (~threads_df.inbound).any(), "No outbound tweets found"


def test_corpus_depth_zero_is_inbound(threads_df):
    depth0 = threads_df[threads_df.depth == 0]
    assert depth0.inbound.all(), "Some depth-0 tweets are not inbound"


def test_golden_columns(golden_df):
    expected = {"ex_id", "conv_id", "text", "brand_reply", "intent", "route"}
    assert expected == set(golden_df.columns), f"Got: {set(golden_df.columns)}"


def test_golden_count(golden_df):
    assert len(golden_df) == 200, f"Expected 200 golden examples, got {len(golden_df)}"


def test_golden_unique_conversations(golden_df):
    assert golden_df.conv_id.nunique() == 200, "Golden set should have 200 unique conversations"


def test_golden_route_values(golden_df):
    assert set(golden_df.route.unique()) == {"auto", "human"}


def test_golden_route_distribution(golden_df):
    auto = (golden_df.route == "auto").sum()
    human = (golden_df.route == "human").sum()
    assert auto == 106, f"Expected 106 auto, got {auto}"
    assert human == 94, f"Expected 94 human, got {human}"


def test_golden_all_intents_present(golden_df):
    from agent import INTENTS
    golden_intents = set(golden_df.intent.unique())
    assert golden_intents == set(INTENTS), f"Missing: {set(INTENTS) - golden_intents}"
