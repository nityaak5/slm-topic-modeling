"""
Build additional few-shot exemplar pools (seed ablation for Reviewer 1).

The existing master pools (master.json, ezstance_icl_master.json, mtcsd_icl_master.json)
pin most/all of the 15 exemplars via hardcoded PREFERRED_*_IDS lists, so re-running
select_few_shot_examples() with a different seed barely changes the prompt content
(verified: only 1-2 of 15 exemplars move for SemEval, 0 for EZStance).

This script builds genuinely random pools instead: same hard quality filters (clean
text, length bounds, no garbled encoding) and the same on-topic-preferred rule for
NONE examples, but no pinned IDs -- candidates are shuffled with the given seed and
the first 5 clean/deduped examples per class are kept. Produces pools for 2 new
seeds x 3 datasets (6 files), so combined with the existing pool that's 3 seeds per
dataset, matching Reviewer 1's "at least three seeds" recommendation.
"""

import csv
import json
import os
import random
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

OUT_DIR = os.path.join(BASE, "data_in", "few_shot_seeds")
NEW_SEEDS = [101, 202]


def _is_clean(doc):
    if not doc or str(doc).lower() in {"nan", "none"}:
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


def select_random_pool(examples, seed):
    """Same class balance / quality bar / on-topic-NONE preference as the original
    pools, but no pinned IDs -- candidates are shuffled by `seed` and the first
    valid 5 per class are kept."""
    rng = random.Random(seed)
    cleaned = [e for e in examples if _is_clean(e["document"])]

    by_stance = {"FAVOR": [], "AGAINST": [], "NONE": []}
    for e in cleaned:
        if e["stance"] in by_stance:
            by_stance[e["stance"]].append(e)

    selected = {"FAVOR": [], "AGAINST": [], "NONE": []}
    used_docs = set()

    def pick(pool, stance):
        for e in pool:
            if len(selected[stance]) == 5:
                break
            if e["document"] in used_docs:
                continue
            selected[stance].append(e)
            used_docs.add(e["document"])

    none_pool = by_stance["NONE"][:]
    rng.shuffle(none_pool)
    on_topic = [e for e in none_pool if _is_on_topic_none(e["query"], e["document"])]
    off_topic = [e for e in none_pool if not _is_on_topic_none(e["query"], e["document"])]
    pick(on_topic + off_topic, "NONE")

    for stance in ("FAVOR", "AGAINST"):
        pool = by_stance[stance][:]
        rng.shuffle(pool)
        pick(pool, stance)

    for stance, items in selected.items():
        if len(items) < 5:
            raise ValueError(f"seed {seed}: only found {len(items)} clean {stance} examples, need 5")

    master = []
    for i in range(5):
        master += [selected["FAVOR"][i], selected["AGAINST"][i], selected["NONE"][i]]
    return master


# ---- dataset-specific loaders -----------------------------------------------

def load_semeval_examples():
    import genai_functions as gf
    train_path = os.path.join(BASE, "data_in", "semeval", "semeval2016-task6-trainingdata.txt")
    return gf.load_semeval_training_examples(train_path)


def load_ezstance_examples():
    train_path = os.path.join(
        BASE, "data_in", "raw", "ezstance", "subtaskA", "noun_phrase", "raw_train_all_onecol.csv"
    )
    examples = []
    with open(train_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            doc = re.sub(r"\s+", " ", str(row["Text"])).strip()
            query = re.sub(r"\s+", " ", str(row["Target 1"])).strip()
            stance = str(row["Stance 1"]).strip().upper()
            if stance not in ("FAVOR", "AGAINST", "NONE"):
                continue
            examples.append({"row_id": str(idx), "query": query, "document": doc, "stance": stance})
    return examples


def load_mtcsd_examples():
    train_path = os.path.join(BASE, "data_in", "mtcsd", "mtcsd_train.csv")
    examples = []
    with open(train_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            doc = re.sub(r"\s+", " ", str(row["content"])).strip()
            query = re.sub(r"\s+", " ", str(row["query"])).strip()
            stance = str(row["stance_label_original"]).strip().upper()
            if stance not in ("FAVOR", "AGAINST", "NONE"):
                continue
            examples.append({"row_id": str(row["id"]), "query": query, "document": doc, "stance": stance})
    return examples


DATASETS = {
    "semeval": load_semeval_examples,
    "ezstance": load_ezstance_examples,
    "mtcsd": load_mtcsd_examples,
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, loader in DATASETS.items():
        examples = loader()
        print(f"\n=== {name}: {len(examples)} raw training examples ===")
        for seed in NEW_SEEDS:
            master = select_random_pool(examples, seed=seed)
            out_path = os.path.join(OUT_DIR, f"{name}_master_seed{seed}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(master, f, indent=2, ensure_ascii=False)
                f.write("\n")
            ids_by_stance = {"FAVOR": [], "AGAINST": [], "NONE": []}
            for e in master:
                ids_by_stance[e["stance"]].append(e["row_id"])
            print(f"  seed {seed} -> {out_path}")
            for stance, ids in ids_by_stance.items():
                print(f"    {stance}: {ids}")


if __name__ == "__main__":
    main()
