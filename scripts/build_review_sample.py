"""
Build manual-review sample of model errors for MT²-CSD.
Models: gpt-5.2, phi-4-mini-instruct, Qwen3-14B
Setting: task_definition (no ICL, no CoT)

Error categories:
  trump_favor_against : Target=Trump, gold=FAVOR, pred=AGAINST
  biden_favor_none    : Target=Biden, gold=FAVOR,   pred=NONE
  biden_against_none  : Target=Biden, gold=AGAINST, pred=NONE
  tesla_any_error     : Target=Tesla, any gold != pred

Sampling: intersection (all 3 agree on error) first, then 2-model, then 1-model.
Limits: 30 per category except Tesla=20.
"""

import os
import pandas as pd
import numpy as np

SEED = 42
BASE = os.path.join(
    os.path.dirname(__file__),
    "..", "data_out", "stance", "mtcsd_test_results"
)
GOLD_PATH = os.path.join(os.path.dirname(__file__), "..", "data_in", "mtcsd", "mtcsd_test.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "review_sample_biden_trump.csv")

MODEL_DIRS = {
    "gpt52":   "GenAIStanceOneShot_2371_mtcsd_test_gpt-5.2_equal_task_definition",
    "phi4mini": "GenAIStanceOneShot_2371_mtcsd_test_phi-4-mini-instruct_equal_task_definition",
    "qwen14b": "GenAIStanceOneShot_2371_mtcsd_test_Qwen3-14B_equal_task_definition",
}

# ── Load predictions ──────────────────────────────────────────────────────────

preds = {}
for key, dirname in MODEL_DIRS.items():
    path = os.path.join(BASE, dirname, "document_assignments.csv")
    df = pd.read_csv(path)[["doc_id", "predicted_stance"]].rename(
        columns={"predicted_stance": f"pred_{key}"}
    )
    preds[key] = df

# Merge all three on doc_id
merged = preds["gpt52"].merge(preds["phi4mini"], on="doc_id").merge(preds["qwen14b"], on="doc_id")

# ── Load gold labels ──────────────────────────────────────────────────────────

gold = pd.read_csv(GOLD_PATH)[["id", "content", "query", "stance_label"]].rename(
    columns={"id": "doc_id", "content": "comment_text", "query": "target",
             "stance_label": "gold"}
)
gold["gold"] = gold["gold"].str.upper()

df = gold.merge(merged, on="doc_id")

# ── Summary table: gold label distribution per target ─────────────────────────

print("\n=== Gold label distribution in test set ===")
summary = (
    df.groupby(["target", "gold"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=["FAVOR", "AGAINST", "NONE"], fill_value=0)
)
summary["TOTAL"] = summary.sum(axis=1)
print(summary.to_string())
print()

# ── Define error categories ───────────────────────────────────────────────────

CATEGORIES = {
    "trump_favor_against": dict(target="Trump",  gold_val="FAVOR",   pred_val="AGAINST"),
    "biden_favor_none":    dict(target="Biden",  gold_val="FAVOR",   pred_val="NONE"),
    "biden_against_none":  dict(target="Biden",  gold_val="AGAINST", pred_val="NONE"),
    "tesla_any_error":     dict(target="Tesla",  gold_val=None,      pred_val=None),
}

LIMITS = {
    "trump_favor_against": 30,
    "biden_favor_none":    30,
    "biden_against_none":  30,
    "tesla_any_error":     20,
}

MODEL_KEYS = ["gpt52", "phi4mini", "qwen14b"]

# ── Sampling helper ───────────────────────────────────────────────────────────

def is_error(row, gold_val, pred_val, model_key):
    """True if this model made the target error on this row."""
    if gold_val is not None:
        # specific gold→pred error
        return row["gold"] == gold_val and row[f"pred_{model_key}"] == pred_val
    else:
        # any error
        return row["gold"] != row[f"pred_{model_key}"]


def sample_category(df_cat, gold_val, pred_val, limit, cat_name):
    """
    Returns a DataFrame of up to `limit` rows, with n_models_agreeing_on_error.
    Fills: intersection first, then 2-model agreement, then 1-model, random seed=42.
    """
    rng = np.random.default_rng(SEED)

    # flag per model whether it made the error
    error_flags = pd.DataFrame({
        mk: df_cat.apply(lambda r: is_error(r, gold_val, pred_val, mk), axis=1)
        for mk in MODEL_KEYS
    }, index=df_cat.index)
    n_errors = error_flags.sum(axis=1)

    # any-error pool (at least 1 model erred)
    pool = df_cat[n_errors >= 1].copy()
    pool["n_models_agreeing_on_error"] = n_errors[pool.index]

    # split by agreement level
    by_level = {3: pool[pool["n_models_agreeing_on_error"] == 3],
                2: pool[pool["n_models_agreeing_on_error"] == 2],
                1: pool[pool["n_models_agreeing_on_error"] == 1]}

    print(f"\n  {cat_name}: intersection={len(by_level[3])}, 2-agree={len(by_level[2])}, 1-agree={len(by_level[1])}, total pool={len(pool)}")

    rows_taken = []
    remaining = limit

    for lvl in [3, 2, 1]:
        if remaining <= 0:
            break
        sub = by_level[lvl]
        if len(sub) == 0:
            continue
        take = sub if len(sub) <= remaining else sub.sample(n=remaining, random_state=SEED)
        rows_taken.append(take)
        remaining -= len(take)

    if not rows_taken:
        return pd.DataFrame()

    result = pd.concat(rows_taken).copy()
    result["category"] = cat_name
    return result


# ── Build all categories ──────────────────────────────────────────────────────

all_samples = []

for cat_name, spec in CATEGORIES.items():
    tgt = spec["target"]
    gv  = spec["gold_val"]
    pv  = spec["pred_val"]
    lim = LIMITS[cat_name]

    # subset to target
    df_sub = df[df["target"] == tgt].copy()

    sampled = sample_category(df_sub, gv, pv, lim, cat_name)
    all_samples.append(sampled)

result = pd.concat(all_samples, ignore_index=True)

# ── Assemble output columns ───────────────────────────────────────────────────

result = result.rename(columns={"doc_id": "instance_id"})

# gold column is already uppercase; pred columns already exist
# pred_llama is the legacy column name requested — map from qwen14b
result = result.rename(columns={"pred_qwen14b": "pred_llama"})

result["full_conversation_context"] = ""   # not available in processed CSV
result["depth"] = ""                        # not available in processed CSV
result["stance_read"] = ""
result["notes"] = ""

out_cols = [
    "instance_id", "target", "category", "gold",
    "pred_gpt52", "pred_phi4mini", "pred_llama",
    "n_models_agreeing_on_error",
    "comment_text",
    "full_conversation_context",
    "depth",
    "stance_read",
    "notes",
]

result = result[out_cols].sort_values(["category", "n_models_agreeing_on_error"],
                                       ascending=[True, False])

result.to_csv(OUT_PATH, index=False)

print(f"\n=== Output written to: {OUT_PATH} ===")
print(f"Total rows: {len(result)}")
print("\nRows per category:")
print(result.groupby("category")["n_models_agreeing_on_error"]
      .value_counts()
      .rename("count")
      .reset_index()
      .pivot(index="category", columns="n_models_agreeing_on_error", values="count")
      .fillna(0)
      .astype(int)
      .to_string())
print("\nn_models breakdown (total per category):")
print(result.groupby("category").size().to_string())
