#!/usr/bin/env python3
import argparse
import csv
import subprocess


def get_row_by_exp_id(csv_path, exp_id):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("exp_id", "")).strip() == exp_id:
                return row
    return None


def add_optional_arg(cmd, flag, value):
    if value is not None and str(value).strip() != "":
        cmd.extend([flag, str(value).strip()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id")
    parser.add_argument("--csv", default="assignments/assignments.csv")
    args = parser.parse_args()

    row = get_row_by_exp_id(args.csv, args.exp_id)
    if row is None:
        raise ValueError(f"No row found for exp_id={args.exp_id} in {args.csv}")

    cmd = ["python", "RunModels.py"]

    add_optional_arg(cmd, "--custom-dataset", row.get("custom_dataset"))
    add_optional_arg(cmd, "--text-column", row.get("text_column"))
    add_optional_arg(cmd, "--query-column", row.get("query_column"))
    add_optional_arg(cmd, "--category-column", row.get("category_column"))
    add_optional_arg(cmd, "--task-type", row.get("task_type"))
    add_optional_arg(cmd, "--method-type", row.get("method_type"))
    add_optional_arg(cmd, "--prompt-name", row.get("prompt_name"))

    llm_backend = str(row.get("llm_backend", "")).strip()
    if llm_backend == "":
        raise ValueError("llm_backend is required")

    cmd.extend(["--set", f"LLM_BACKEND={llm_backend}"])

    if llm_backend == "openai":
        openai_model = str(row.get("openai_model", "")).strip()
        if openai_model == "":
            raise ValueError("openai_model is required when llm_backend=openai")
        cmd.extend(["--set", f"OPENAI_MODEL={openai_model}"])
    elif llm_backend in {"claude", "anthropic"}:
        claude_model = str(row.get("claude_model", "")).strip()
        if claude_model == "":
            raise ValueError("claude_model is required when llm_backend=claude")
        cmd.extend(["--set", f"CLAUDE_MODEL={claude_model}"])
    elif llm_backend == "vllm":
        vllm_model = str(row.get("vllm_model", "")).strip()
        if vllm_model == "":
            raise ValueError("vllm_model is required when llm_backend=vllm")
        cmd.extend(["--set", f"VLLM_MODEL={vllm_model}"])

        vllm_max_model_len = str(row.get("vllm_max_model_len", "")).strip()
        if vllm_max_model_len != "":
            cmd.extend(["--set", f"VLLM_MAX_MODEL_LEN={vllm_max_model_len}"])
    else:
        raise ValueError(f"Unsupported llm_backend: {llm_backend}")

    optional_set_columns = {
        "n_documents": "N_documents",
        "n_runs": "N_runs",
        "sampling_method": "SAMPLING_METHOD",
        "use_all_documents": "USE_ALL_DOCUMENTS",
        "claude_batch_size": "CLAUDE_BATCH_SIZE",
        "claude_batch_sleep_seconds": "CLAUDE_BATCH_SLEEP_SECONDS",
        "claude_max_tokens": "CLAUDE_MAX_TOKENS",
        "few_shot_master_path": "FEW_SHOT_MASTER_PATH",
    }
    for csv_column, env_key in optional_set_columns.items():
        value = row.get(csv_column)
        if value is not None and str(value).strip() != "":
            cmd.extend(["--set", f"{env_key}={str(value).strip()}"])

    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
