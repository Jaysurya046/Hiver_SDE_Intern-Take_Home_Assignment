"""Rebuild data/amazonhelp_threads.csv from the original twcs.csv.

A conversation = a first-inbound customer tweet that mentions @AmazonHelp,
is not itself a reply, and has at least one AmazonHelp response. The thread
then alternates customer -> brand -> customer ... following the earliest
matching-direction reply edge at each hop (depth cap 30).

Run from the repo root:
    python scripts/rebuild_corpus.py <path-to-twcs.csv>

Takes ~2 minutes and ~3GB RAM on the 516MB file. The committed
data/amazonhelp_threads.csv (78,686 tweets / 21,423 conversations) is the
canonical copy; this script reproduces it deterministically.
"""
import os
import sys

import pandas as pd

BRAND = "AmazonHelp"


def parse_ids(s):
    if pd.isna(s):
        return []
    return [int(float(x.strip())) for x in str(s).split(",") if x.strip()]


def main(twcs_path):
    print(f"reading {twcs_path} ...")
    df = pd.read_csv(
        twcs_path,
        usecols=["tweet_id", "author_id", "inbound", "created_at", "text",
                 "response_tweet_id", "in_response_to_tweet_id"],
        dtype={"tweet_id": "int64", "author_id": "str", "inbound": "bool",
               "created_at": "str", "text": "str",
               "response_tweet_id": "str", "in_response_to_tweet_id": "float64"},
    )
    df = df[(df.inbound) | (df.author_id == BRAND)]
    print(f"customer + {BRAND} tweets: {len(df):,}")

    recs = {}
    for r in df.itertuples(index=False):
        recs[r.tweet_id] = (r.author_id, r.inbound, r.created_at, r.text,
                            r.response_tweet_id, r.in_response_to_tweet_id)

    first = [tid for tid, (a, i, ca, tx, resp, in_resp) in recs.items()
             if i and pd.isna(in_resp) and "@amazonhelp" in str(tx).lower() and not pd.isna(resp)]
    print(f"first-inbound conversation seeds: {len(first):,}")

    def walk(start):
        thread, cur = [], start
        for depth in range(30):
            if cur not in recs:
                break
            author, inbound, created, text, resp, in_resp = recs[cur]
            thread.append((depth, cur, author, bool(inbound), created, text))
            want_inbound = not bool(inbound)
            cands = [rid for rid in parse_ids(resp)
                     if rid in recs and bool(recs[rid][1]) == want_inbound]
            if not cands:
                break
            cur = min(cands)  # earliest reply
        return thread

    rows = []
    for start in first:
        th = walk(start)
        if len(th) >= 2:  # at least one brand reply
            for depth, tid, author, inbound, created, text in th:
                rows.append({"conv_id": start, "depth": depth, "tweet_id": tid,
                             "author_id": author, "inbound": inbound,
                             "created_at": created, "text": text})

    thr = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    thr.to_csv("data/amazonhelp_threads.csv", index=False)
    print(f"rows: {len(thr):,}  convs: {thr.conv_id.nunique():,}  "
          f"size: {os.path.getsize('data/amazonhelp_threads.csv')/1e6:.1f}MB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "twcs.csv")
