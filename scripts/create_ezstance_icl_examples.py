"""
Build EZStance-native few-shot ICL examples from SubtaskA noun_phrase training set.

Selects 5 clean examples per class (FAVOR / AGAINST / NONE) = 15 master examples,
nested into subsets of 3/6/9/12/15 shots, and saves to data_in/ezstance_icl_master.json.

The output format mirrors SemEval master.json so it can be loaded by
build_nested_few_shot_sets() in genai_functions.py.
"""

import json
import random
import re
import os
import csv

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE, "ezstance", "subtaskA", "noun_phrase", "raw_train_all_onecol.csv")
OUT_PATH = os.path.join(BASE, "data_in", "ezstance", "ezstance_icl_master.json")

SEED = 42
PREFERRED_ROW_IDS = {
    "FAVOR": [7560, 1752, 3477, 1948, 2659],
    "AGAINST": [2973, 6623, 3555, 2873, 6476],
    "NONE": [3388, 6050, 987, 9645, 7812],
}


def _normalize_whitespace(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def _is_clean(doc):
    if not doc or doc.lower() in {"nan", "none"}:
        return False
    if len(doc) < 25 or len(doc) > 180:
        return False
    if "�" in doc:
        return False
    if doc.count('"') > 6:
        return False
    return True


def _is_on_topic_none(query, doc):
    query_tokens = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2}
    doc_tokens = set(re.findall(r"[a-z0-9]+", doc.lower()))
    return bool(query_tokens & doc_tokens)


def _sort_key(doc, rng):
    lowered = doc.lower()
    return (
        int(lowered.startswith("rt ")),
        int("http" in lowered),
        int(doc.count("@") > 2),
        abs(len(doc) - 95),
        rng.random(),
    )


def build_nested_few_shot_sets(master):
    grouped = {"FAVOR": [], "AGAINST": [], "NONE": []}
    for ex in master:
        grouped[ex["stance"]].append(ex)
    shots_per_label = len(master) // 3
    nested = {}
    for n in range(1, shots_per_label + 1):
        subset = []
        for i in range(n):
            subset += [grouped["FAVOR"][i], grouped["AGAINST"][i], grouped["NONE"][i]]
        nested[n * 3] = subset
    nested["master"] = list(master)
    return nested


def main():
    rng = random.Random(SEED)

    # Build example dicts
    examples = []
    with open(TRAIN_PATH, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            doc = _normalize_whitespace(row["Text"])
            query = _normalize_whitespace(row["Target 1"])
            stance = str(row["Stance 1"]).strip().upper()
            if stance not in ("FAVOR", "AGAINST", "NONE"):
                continue
            if not _is_clean(doc):
                continue
            examples.append({
                "row_id": str(idx),
                "query": query,
                "document": doc,
                "stance": stance,
            })

    print(f"Clean examples after filtering: {len(examples)}")
    by_stance = {"FAVOR": [], "AGAINST": [], "NONE": []}
    for ex in examples:
        by_stance[ex["stance"]].append(ex)

    selected = {"FAVOR": [], "AGAINST": [], "NONE": []}
    examples_by_id = {ex["row_id"]: ex for ex in examples}
    used_docs = set()   # deduplicate by document text
    used_queries = set()  # deduplicate by query across the whole master pool

    def pick(pool, target_stance):
        for ex in pool:
            if len(selected[target_stance]) == 5:
                break
            if ex["document"] in used_docs:
                continue
            if ex["query"].casefold() in used_queries:
                continue
            selected[target_stance].append(ex)
            used_docs.add(ex["document"])
            used_queries.add(ex["query"].casefold())

    # Start from a manually reviewed, reproducible set. Fall back to heuristics
    # if any preferred row is missing or collides with a duplicate query/doc.
    for stance, row_ids in PREFERRED_ROW_IDS.items():
        preferred = []
        for row_id in row_ids:
            example = examples_by_id.get(str(row_id))
            if example is not None:
                preferred.append(example)
        pick(preferred, stance)

    # NONE: prefer on-topic first
    none_on  = sorted(
        [e for e in by_stance["NONE"] if _is_on_topic_none(e["query"], e["document"])],
        key=lambda e: _sort_key(e["document"], rng),
    )
    none_off = sorted(
        [e for e in by_stance["NONE"] if not _is_on_topic_none(e["query"], e["document"])],
        key=lambda e: _sort_key(e["document"], rng),
    )
    pick(none_on + none_off, "NONE")

    for stance in ("FAVOR", "AGAINST"):
        pool = sorted(by_stance[stance], key=lambda e: _sort_key(e["document"], rng))
        pick(pool, stance)

    # Verify
    for stance, items in selected.items():
        if len(items) < 5:
            raise ValueError(f"Only found {len(items)} clean {stance} examples — need 5.")
        print(f"  {stance}: {len(items)} selected")
        for ex in items:
            print(f"    [{ex['row_id']}] query={ex['query']!r}  doc={ex['document'][:60]!r}")

    # Interleave F/A/N for master list
    master = []
    for i in range(5):
        master += [selected["FAVOR"][i], selected["AGAINST"][i], selected["NONE"][i]]

    nested = build_nested_few_shot_sets(master)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nSaved {len(master)} master examples -> {OUT_PATH}")
    print("Nested shot counts:", {k: len(v) for k, v in nested.items() if k != "master"})


if __name__ == "__main__":
    main()
