"""
Build single-column-width confusion matrix figures for the appendix (Reviewer 2, Q1).

Sources the same canonical result folders as notebooks/carbon_vs_f1_analysis.ipynb
(DATASET_DIRS), so the numbers here match the paper's actual reported results rather
than whatever an older exploratory notebook happened to filter.

For each dataset: per model, average the row-normalized confusion matrix across all
11 prompting strategies, keep only the 6 off-diagonal "error" transitions (the
diagonal FAVOR->FAVOR/AGAINST->AGAINST/NONE->NONE cells are correct predictions, not
shift), and plot at a size that actually fits a single ACL column.
"""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from sklearn.metrics import confusion_matrix

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STANCE_DIR = os.path.join(BASE, "data_out", "stance")
OUT_DIR = os.path.join(BASE, "data_out", "stance", "stance_analysis")

DATASET_DIRS = {
    "semeval": [os.path.join(STANCE_DIR, "semeval_results")],
    "ezstance": [os.path.join(STANCE_DIR, "ezstance_test")],
    "mtcsd": [os.path.join(STANCE_DIR, "mtcsd_test_results"), os.path.join(STANCE_DIR, "mtcsd_test")],
}

CLASSES = ["FAVOR", "AGAINST", "NONE"]
ALL_TRANS = [f"{tc}→{pc}" for tc in CLASSES for pc in CLASSES]
KEY_TRANS = ["FAVOR→AGAINST", "FAVOR→NONE", "AGAINST→FAVOR", "AGAINST→NONE", "NONE→FAVOR", "NONE→AGAINST"]

PROMPT_NAMES = {
    "default_no_label_definitions", "default", "task_definition", "task_definition_scale",
    "question", "cot", "few_shot_3", "few_shot_6", "few_shot_9", "few_shot_12", "few_shot_15",
}

SHORT_NAMES = {
    "Qwen/Qwen3-1.7B": "Qwen3-1.7B",
    "Qwen/Qwen3-4B": "Qwen3-4B",
    "Qwen/Qwen3-4B-Instruct-2507": "Qwen3-4B-IT",
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "Qwen/Qwen3-14B": "Qwen3-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "DS-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "DS-Qwen-14B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "microsoft/phi-4": "Phi-4",
    "microsoft/phi-4-mini-instruct": "Phi-4-mini",
    "nvidia/Nemotron-Mini-4B-Instruct": "Nemotron-4B",
    "Nemotron-Mini-4B-Instruct": "Nemotron-4B",
    "gpt-5-mini": "GPT-5-mini",
    "gpt-5.2": "GPT-5.2",
    "claude-haiku-4-5-20251001": "Haiku-4.5",
}


def normalize_label(x):
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if s == "FAVOR":
        return "FAVOR"
    if s == "AGAINST":
        return "AGAINST"
    if s in ("NONE", "NEUTRAL", "NEITHER"):
        return "NONE"
    return None


def load_confusion_rows(dataset_dirs):
    rows = []
    for d in dataset_dirs:
        if not os.path.isdir(d):
            continue
        for folder in sorted(os.listdir(d)):
            folder_path = os.path.join(d, folder)
            summary_path = os.path.join(folder_path, "summary.json")
            assignments_path = os.path.join(folder_path, "document_assignments.csv")
            if not (os.path.isdir(folder_path) and os.path.exists(summary_path) and os.path.exists(assignments_path)):
                continue
            import json
            s = json.load(open(summary_path))
            cfg = s.get("configuration", {})
            backend = cfg.get("llm_backend", "")
            if backend == "vllm":
                model = cfg.get("vllm_model", "")
            elif backend == "openai":
                model = cfg.get("openai_model", "")
            elif backend == "claude":
                model = cfg.get("claude_model", "")
            else:
                continue
            prompt = cfg.get("stance_prompt_name", "")
            if prompt not in PROMPT_NAMES:
                continue

            df = pd.read_csv(assignments_path)
            if "run" in df.columns:
                df = df[df["run"] == 0]
            df["true"] = df["original_stance"].apply(normalize_label)
            df["pred"] = df["predicted_stance"].apply(normalize_label)
            valid = df[df["true"].notna() & df["pred"].notna()]
            if len(valid) < 100:
                continue

            cm = confusion_matrix(valid["true"], valid["pred"], labels=CLASSES).astype(float)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_norm = cm / row_sums

            row = {"model": SHORT_NAMES.get(model, model.split("/")[-1]), "prompt": prompt}
            for ri, tc in enumerate(CLASSES):
                for ci, pc in enumerate(CLASSES):
                    row[f"{tc}→{pc}"] = cm_norm[ri, ci]
            rows.append(row)
    return pd.DataFrame(rows)


def plot_confusion(cm_model, title, out_path, columns, group_size=3, fig_width=6.5, vmax=None):
    rlabels = list(cm_model.index)
    data = cm_model[columns].values

    if vmax is None:
        vmax = min(1.0, data.max() * 1.1)

    fig, ax = plt.subplots(figsize=(fig_width, max(3.2, len(rlabels) * 0.34)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)

    for g in range(group_size, len(columns), group_size):
        ax.axvline(g - 0.5, color="white", linewidth=1.8)

    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            v = data[r, c]
            ax.text(c, r, f"{v*100:.1f}", ha="center", va="center", fontsize=7.5,
                    color="black" if v < 0.55 else "white")

    ax.set_yticks(range(len(rlabels)))
    ax.set_yticklabels(rlabels, fontsize=8)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([t.replace("→", "→\n") for t in columns], fontsize=7)
    ax.tick_params(length=2, pad=2)

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.ax.tick_params(labelsize=7)
    cb.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    plt.tight_layout(pad=0.3)
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}  ({len(rlabels)} models)")
    print(f"  models: {rlabels}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    titles = {
        "semeval": "SemEval-2016 — key error transitions (mean across prompts)",
        "ezstance": "EZStance-NP — key error transitions (mean across prompts)",
        "mtcsd": "MT²-CSD — key error transitions (mean across prompts)",
    }
    cm_models = {}
    for name, dirs in DATASET_DIRS.items():
        cm_df = load_confusion_rows(dirs)
        if cm_df.empty:
            print(f"WARNING: no data found for {name}")
            continue
        cm_models[name] = cm_df.groupby("model")[ALL_TRANS].mean().sort_index()

    shared_vmax = min(1.0, max(cm[KEY_TRANS].values.max() for cm in cm_models.values()) * 1.1)
    print(f"shared 6-col color scale vmax: {shared_vmax:.3f}")

    for name, cm_model in cm_models.items():
        out_path_9 = os.path.join(OUT_DIR, f"{name}_confusion_matrix.png")
        plot_confusion(cm_model, titles[name], out_path_9, ALL_TRANS, group_size=3, fig_width=6.5)

        out_path_6 = os.path.join(OUT_DIR, f"{name}_confusion_matrix_6col.png")
        plot_confusion(cm_model, titles[name], out_path_6, KEY_TRANS, group_size=2, fig_width=6.0, vmax=shared_vmax)


if __name__ == "__main__":
    main()
