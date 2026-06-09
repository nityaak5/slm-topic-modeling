"""
Normalize opinionated/multilingual queries in html_out_stance_sample100.csv
to neutral English stance targets.

- Adds `query_original` column (preserves raw search query)
- Overwrites `query` column with normalized neutral English target
- Uses the same LLM backend as the rest of the pipeline (openai or claude)

Usage:
    python scripts/normalize_queries.py --backend openai
    python scripts/normalize_queries.py --backend claude
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Load .env from project root if keys not already in environment
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from genai_functions import complete_openai_request, complete_claude_request

SRC = "data_in/html_out/html_out_stance_sample100.csv"
OUT = "data_in/html_out/html_out_stance_sample100_normalized.csv"

PROMPT_TEMPLATE = """\
You are given a web search query written in any language. The query was used to find articles about anti-LGBT rhetoric or far-right politics in Europe.

Your task: convert the query into a short, neutral English noun phrase that can serve as a stance detection target (like "Geert Wilders", "LGBT rights", "same-sex marriage", "gender ideology").

Rules:
- Output neutral English only — remove any opinion or sentiment framing
- Keep it concise (1-6 words)
- If the query targets a named person, use their name
- If the query targets a policy or topic, use a neutral label for that topic

Query: {query}

Return JSON in exactly this format: {{"target": "..."}}"""


def normalize(query: str, backend: str) -> str:
    prompt = PROMPT_TEMPLATE.format(query=query)
    try:
        if backend == "openai":
            result = complete_openai_request(prompt, model=os.getenv("OPENAI_MODEL", "gpt-4o"))
        else:
            result = complete_claude_request(prompt, model=os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest"))
        return (result or {}).get("target", query).strip()
    except Exception as e:
        print(f"  WARNING: failed for [{query}]: {e}")
        return query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["openai", "claude"], default="openai")
    args = parser.parse_args()

    df = pd.read_csv(SRC)

    # Preserve original query
    if "query_original" not in df.columns:
        df.insert(df.columns.get_loc("query") + 1, "query_original", df["query"])

    # Normalize unique queries (avoid duplicate API calls)
    unique_queries = df["query"].unique()
    print(f"Normalizing {len(unique_queries)} unique queries via {args.backend}...")

    cache = {}
    for i, q in enumerate(unique_queries, 1):
        print(f"  [{i}/{len(unique_queries)}] {q!r}")
        cache[q] = normalize(q, args.backend)
        print(f"    → {cache[q]!r}")
        time.sleep(0.2)

    df["query"] = df["query_original"].map(cache)

    df.to_csv(OUT, index=False)
    print(f"\nSaved → {OUT}")
    print("\nSample (original → normalized):")
    sample = df[["query_original", "query"]].drop_duplicates().head(10)
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
