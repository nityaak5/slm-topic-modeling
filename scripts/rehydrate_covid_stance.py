"""
Rehydrate COVID-19-Stance dataset from tweet IDs.
Downloads label CSVs from GitHub, fetches tweet text via Twitter API v2,
saves combined output to data_in/covid_stance_test.csv.

Requires: pip install tweepy requests
Requires: Twitter API v2 Bearer Token (Basic tier or higher, $100/month)
  -> Set env var: TWITTER_BEARER_TOKEN=<your token>
  -> Or pass --bearer-token <token>
"""

import csv
import os
import sys
import time
import argparse
import requests
from io import StringIO

try:
    import tweepy
except ImportError:
    print("Run: pip install tweepy")
    sys.exit(1)

GITHUB_BASE = "https://raw.githubusercontent.com/kglandt/stance-detection-in-covid-19-tweets/main/dataset"

SPLITS = {
    "test": [
        ("fauci_test.csv",              "Anthony S. Fauci, M.D."),
        ("school_closures_test.csv",    "Keeping Schools Closed"),
        ("stay_at_home_orders_test.csv","Stay at Home Orders"),
        ("face_masks_test.csv",         "Wearing a Face Mask"),
    ],
    "train": [
        ("fauci_train.csv",              "Anthony S. Fauci, M.D."),
        ("school_closures_train.csv",    "Keeping Schools Closed"),
        ("stay_at_home_orders_train.csv","Stay at Home Orders"),
        ("face_masks_train.csv",         "Wearing a Face Mask"),
    ],
    "val": [
        ("fauci_val.csv",              "Anthony S. Fauci, M.D."),
        ("school_closures_val.csv",    "Keeping Schools Closed"),
        ("stay_at_home_orders_val.csv","Stay at Home Orders"),
        ("face_masks_val.csv",         "Wearing a Face Mask"),
    ],
}


def download_labels(split="test"):
    rows = []
    for filename, target_full in SPLITS[split]:
        url = f"{GITHUB_BASE}/{filename}"
        print(f"  Downloading {filename} ...")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        reader = csv.DictReader(StringIO(r.text))
        for row in reader:
            rows.append({
                "tweet_id": row["Tweet Id"].strip(),
                "target":   target_full,
                "stance":   row["Stance"].strip(),
                "opinion":  row.get("Opinion Towards", "").strip(),
                "sentiment":row.get("Sentiment", "").strip(),
            })
    print(f"  Downloaded {len(rows)} labelled rows for split='{split}'")
    return rows


def fetch_tweets(tweet_ids, client, batch_size=100):
    """Fetch tweet text in batches of 100 (API max). Returns dict id->text."""
    texts = {}
    batches = [tweet_ids[i:i+batch_size] for i in range(0, len(tweet_ids), batch_size)]
    for i, batch in enumerate(batches):
        print(f"  Fetching batch {i+1}/{len(batches)} ({len(batch)} tweets)...")
        try:
            resp = client.get_tweets(
                ids=batch,
                tweet_fields=["id", "text"],
            )
            if resp.data:
                for tweet in resp.data:
                    texts[str(tweet.id)] = tweet.text
            # respect rate limit: 300 req/15min on basic tier
            if i < len(batches) - 1:
                time.sleep(0.5)
        except tweepy.TooManyRequests:
            print("  Rate limited — sleeping 60s...")
            time.sleep(60)
            # retry once
            resp = client.get_tweets(ids=batch, tweet_fields=["id", "text"])
            if resp.data:
                for tweet in resp.data:
                    texts[str(tweet.id)] = tweet.text
        except Exception as e:
            print(f"  Warning: batch {i+1} failed: {e}")
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bearer-token", default=os.environ.get("TWITTER_BEARER_TOKEN"),
                        help="Twitter API v2 Bearer Token")
    parser.add_argument("--split", default="test", choices=["test", "train", "val", "all"],
                        help="Which split(s) to hydrate (default: test)")
    parser.add_argument("--out-dir", default="data_in",
                        help="Output directory (default: data_in)")
    args = parser.parse_args()

    if not args.bearer_token:
        print("ERROR: Twitter Bearer Token required.")
        print("  Set env var: export TWITTER_BEARER_TOKEN=<token>")
        print("  Or pass:     --bearer-token <token>")
        print("  Get one at:  https://developer.twitter.com/en/portal/dashboard")
        sys.exit(1)

    client = tweepy.Client(bearer_token=args.bearer_token, wait_on_rate_limit=True)

    splits = ["test", "train", "val"] if args.split == "all" else [args.split]

    for split in splits:
        print(f"\n=== Split: {split} ===")
        rows = download_labels(split)

        tweet_ids = list({r["tweet_id"] for r in rows})
        print(f"  Unique tweet IDs: {len(tweet_ids)}")

        print("  Fetching tweet text from Twitter API v2...")
        texts = fetch_tweets(tweet_ids, client)
        print(f"  Retrieved {len(texts)} tweets ({len(tweet_ids)-len(texts)} deleted/unavailable)")

        # merge
        out_rows = []
        for r in rows:
            text = texts.get(r["tweet_id"])
            if text:
                out_rows.append({
                    "tweet_id": r["tweet_id"],
                    "text":     text,
                    "target":   r["target"],
                    "stance":   r["stance"],
                    "opinion":  r["opinion"],
                    "sentiment":r["sentiment"],
                })

        out_path = os.path.join(args.out_dir, f"covid_stance_{split}.csv")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["tweet_id","text","target","stance","opinion","sentiment"])
            writer.writeheader()
            writer.writerows(out_rows)

        print(f"  Saved {len(out_rows)} rows -> {out_path}")

        # stats
        from collections import Counter
        print("  Stance distribution:")
        for stance, n in sorted(Counter(r["stance"] for r in out_rows).items()):
            print(f"    {stance}: {n}")
        print("  Per target:")
        for target in ["Anthony S. Fauci, M.D.", "Keeping Schools Closed",
                       "Stay at Home Orders", "Wearing a Face Mask"]:
            n = sum(1 for r in out_rows if r["target"] == target)
            print(f"    {target}: {n}")


if __name__ == "__main__":
    main()
