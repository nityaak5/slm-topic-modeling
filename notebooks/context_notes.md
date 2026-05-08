# Context Notes — Stance Detection Analysis
_Last updated: 2026-05-04_

---

## Project

Evaluating LLMs for stance detection on **SemEval 2016 Task 6** test set (1,249 tweets, 5 topics).  
Main notebook: `notebooks/class_analysis.ipynb`  
Results folder: `data_out/semeval_results/`

**14 models** (3 closed, 11 open) × **11 prompts** = 154 (model, prompt) combinations.

**Closed:** claude-haiku-4-5-20251001, gpt-5.2, gpt-5-mini  
**Open:** Qwen3-1.7B/4B/4B-v2/8B/14B, DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-14B, Llama-3.1-8B-Instruct, phi-4, phi-4-mini-instruct, Nemotron-Mini-4B-Instruct

**Prompts (in order):** default_no_label_definitions, default, task_definition, question, cot, task_definition_scale, few_shot_3/6/9/12/15  
**Classes:** FAVOR, AGAINST, NONE  
**SemEval NLP baseline:** macro F1 = 0.69 (only closed models beat this)

**Ground truth distribution:** FAVOR 24.4%, AGAINST 57.3%, NONE 18.4%  
**Mean predicted distribution:** FAVOR 31.4%, AGAINST 35.2%, NONE 33.4%

---

## Notebook Structure

| Section | Content |
|---------|---------|
| 0 | Overview: macro F1 across models and prompts (line charts) |
| 1–11 | Default prompt only: per-class F1 heatmap, predicted vs ground truth, NONE P/R scatter, Qwen3 scaling, confusion matrices, per-topic NONE analysis |
| 12 | Neutrality bias analysis: bias scores (pred% − gt%), heatmaps, zero-sum redistribution, few-shot effect on NONE bias, NONE bias vs macro F1 |
| 13 | Prompt impact: F1 bar charts per class + P/R table + P/R scatter (base only, all prompts) |
| 14 | Model impact: P/R/F1 table + grouped bar charts + line charts (F1 per model per prompt) + per-model P/R scatter (points = prompts) |
| 15 | Topic analysis: bias heatmaps + P/R/F1 table + grouped bar charts + per-topic P/R scatter |

**Key variable:** `bias_df` — one row per (model, prompt), has p/r/f1 per class, bias scores, macro F1. Built in Section 12.1.  
**Key variable:** `topic_pr_df` — one row per (model, prompt, topic, class), has P/R/F1. Built in Section 15.4a.

---

## Core Findings

### 1. Homogenisation (structural, not prompt-specific)
All models redistribute AGAINST → NONE/FAVOR. Holds across all 11 prompts and all 14 models. Not a prompting artefact — it's model-level.

### 2. The P/R signature
- **AGAINST:** P >> R everywhere — models only commit when certain (P=0.88–0.97), but rarely commit (R=0.19–0.72). Under-predicted.
- **NONE:** R >> P everywhere — models catch most true NONEs but sweep in AGAINST too (P=0.26–0.59, R=0.43–0.92). Over-predicted.
- **FAVOR:** R > P mildly — slight over-prediction across the board.

### 3. Why: alignment-driven conservatism
Each prediction is independent — models don't "see" the distribution. Homogenisation is the sum of individual hedging decisions. RLHF/safety training makes models avoid assertive opposition stances. NONE is the path of least resistance under uncertainty.

### 4. LLM F1 improvement over NLP baseline is a macro F1 artefact
LLMs appear better on FAVOR and NONE F1 because high recall inflates F1 despite low precision. NLP baseline (SVM-ngrams) was strong on AGAINST (majority class) by leaning into it — LLMs sacrifice AGAINST recall.  
**NLP baseline per-class values need to be confirmed from the paper (currently using placeholders: FAVOR=0.58, AGAINST=0.76, NONE=0.29).**

### 5. Model results
| Model | AGAINST recall | NONE precision | Macro F1 |
|-------|---------------|----------------|---------|
| Claude Haiku | **0.72** (best) | 0.59 | 0.756 |
| GPT-5.2 | 0.68 | 0.55 | 0.744 |
| GPT-5-mini | 0.65 | 0.56 | 0.738 |
| Phi-4 (best open) | 0.60 | 0.48 | 0.669 |
| Qwen3-14B | 0.45 | 0.40 | 0.605 |
| Qwen3-1.7B | 0.19 (worst) | 0.26 | 0.359 |

**Outliers:** Llama-3.1-8B and Nemotron-4B are the ONLY models that under-predict NONE (NONE R < P) — they shift uncertainty to FAVOR instead. Completely different failure mode from all other models.  
**Qwen3 scaling:** AGAINST recall gets WORSE with size (4B→8B→14B: 0.49→0.47→0.45). Larger = more conservative alignment.

### 6. Prompt results
- **Scale** breaks FAVOR (P=0.44, R=0.86) — most extreme over-prediction; worst overall (macro 0.508)
- **Default** has worst NONE inflation (P=0.38, R=0.88)
- **CoT and Question** best zero-shot: AGAINST recall 0.56–0.58 vs Default 0.48
- **12-shot** is the sweet spot: AGAINST recall 0.61, NONE precision 0.50; 15-shot regresses slightly (context crowding)
- Few-shot reduces NONE recall (0.76→0.73) while raising NONE precision (0.46→0.50) — examples calibrate NONE selectivity

### 7. Topic results
**Climate Change inverts the entire pattern:**
- FAVOR: P=0.938, R=0.673 → UNDER-predicted (only topic where this happens)
- AGAINST: P=0.538, R=0.724 → OVER-predicted (only topic where this happens)
- Models are conservative about predicting support for climate change (treat it as nuanced), while climate skepticism in tweets is explicit enough to be detected and over-applied

**Atheism worst overall:** NONE P=0.315 (lowest in dataset), AGAINST R=0.457. Models most reluctant to take any stance on religion.

**Hillary Clinton best:** AGAINST recall 0.643, NONE most calibrated (P=0.619, R=0.712). Explicit partisan tweets are easiest to classify.

**Abortion and Feminist:** Very high AGAINST precision (0.888, 0.907), very low recall (0.481, 0.556) — models only commit to opposition on these sensitive topics when completely unambiguous.

---

## Next Steps (not yet done)

1. **Confirm NLP baseline per-class F1** from the paper to replace placeholders in `NLP_BASELINE` dict (cell ae827790)
2. **Per-topic × model analysis** — does each model show the same topic-level inversion on Climate? Or is it model-specific?
3. **Per-topic × prompt analysis** — does few-shot help more for Atheism (worst bias) than Hillary (least bias)?
4. **Write up results section** — the core narrative is ready; need to anchor NLP baseline comparison

---

## Key Observations for Paper

- The improvement on FAVOR/NONE F1 is partially a **macro F1 artefact** (recall inflation) — state this explicitly
- Homogenisation is driven by **individual hedging**, not cross-document awareness — each prediction is independent
- **Alignment hypothesis** is supported by: (a) closed models less biased despite stronger alignment (better calibration), (b) topic sensitivity correlates with NONE inflation (Atheism > Abortion > Feminist > Climate ≈ Hillary), (c) Qwen3 scaling makes conservatism worse
- **Climate Change is a methodologically important exception** — any aggregate analysis that masks topic-level variation will miss this inversion
- **Llama/Nemotron outliers** suggest not all open models fail the same way — worth a separate paragraph
- Rogers & Zhang (2025) connection: same homogenising mechanism as book recommendation bias — cite explicitly
