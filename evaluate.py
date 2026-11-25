#!/usr/bin/env python3
"""Evaluate one merged CapPO checkpoint on the main ten benchmarks."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


KIT = Path(__file__).resolve().parent / "VLMEvalKit"
BENCHMARKS = {
    "MathVista_MINI": "MathVista",
    "MathVerse_MINI": "MathVerse",
    "MathVision_MINI": "MathVision",
    "WeMath_COT": "WeMath",
    "DynaMath": "Dynamath",
    "LogicVista": "LogicVista",
    "ScienceQA_TEST": "ImageMCQDataset",
    "MMStar": "ImageMCQDataset",
    "MMMU_DEV_VAL": "MMMUDataset",
    "MMMU_Pro_10c_COT": "MMMUProDataset",
}
JUDGE_35_DATASETS = {"ScienceQA_TEST", "MMStar", "MMMU_DEV_VAL", "MMMU_Pro_10c_COT"}
MMMU_1020_SHA256 = "66507fbd929eb6c79889521402698eae0bcf428521d735b453b98e69da487c5e"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_score(output, dataset):
    pattern = "*_cotpost_acc.csv" if dataset == "MMMU_Pro_10c_COT" else "*.csv"
    files = [
        path for path in output.rglob(pattern)
        if f"_{dataset}_" in path.name and path.name.endswith(("_score.csv", "_acc.csv"))
    ]
    if not files:
        raise FileNotFoundError(f"No final score for {dataset} under {output}")
    path = max(files, key=lambda p: (p.stat().st_mtime_ns, p.is_symlink()))
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    def value(text):
        return float(text.strip().rstrip("%"))
    if dataset == "MathVerse_MINI":
        score = sum(value(row["Overall"]) for row in rows) / len(rows)
    elif dataset in {"MathVista_MINI", "MathVision_MINI", "LogicVista"}:
        score = value(next(row["acc"] for row in rows if "Overall" in row.values()))
    elif dataset == "WeMath_COT":
        score = value(rows[0]["Score (Strict)"])
    elif dataset == "DynaMath":
        score = 100 * value(next(row["Overall"] for row in rows if row["Setting"] == "Average"))
    elif dataset in {"ScienceQA_TEST", "MMMU_DEV_VAL"}:
        split = "test" if dataset == "ScienceQA_TEST" else "validation"
        score = 100 * value(next(row["Overall"] for row in rows if row["split"] == split))
    else:
        score = 100 * value(rows[0]["Overall"])
    if not 0 <= score <= 100:
        raise ValueError(f"Invalid accuracy in {path}")
    return {"benchmark": dataset, "score_percent": score, "file": str(path.resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Merged local Hugging Face checkpoint")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory with VLMEvalKit TSVs")
    parser.add_argument("--output-dir", type=Path, required=True, help="A new output directory")
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--math-judge", default="gpt-4o-mini", help="Judge for the first six benchmarks")
    parser.add_argument("--choice-judge", default="gpt-3.5-turbo", help="Judge for the last four benchmarks")
    parser.add_argument("--api-nproc", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print protocol")
    args = parser.parse_args()
    gpu_ids = args.gpus.split(",")
    if not re.fullmatch(r"\d+(?:,\d+)*", args.gpus) or len(set(gpu_ids)) != len(gpu_ids):
        parser.error("--gpus must be a comma-separated list of unique numeric device IDs")
    if args.api_nproc < 1:
        parser.error("--api-nproc must be positive")
    model = args.model.resolve()
    data_root = args.data_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("--output-dir must be a new path")
    config_path = model / "config.json"
    if not config_path.is_file() or not any(
        p.is_file() and p.stat().st_size for pattern in ("*.safetensors", "pytorch_model*.bin")
        for p in model.glob(pattern)
    ):
        parser.error("--model must contain config.json and merged model weights")
    if json.loads(config_path.read_text()).get("model_type") != "qwen2_5_vl":
        parser.error("--model must be a Qwen2.5-VL checkpoint")
    datasets = list(dict.fromkeys(args.benchmarks))
    paths = {dataset: data_root / f"{dataset}.tsv" for dataset in datasets}
    missing = [str(path) for path in paths.values() if not path.is_file() or not path.stat().st_size]
    if missing:
        parser.error("Missing dataset TSVs: " + ", ".join(missing))
    hashes = {name: sha256(path) for name, path in paths.items()}
    use_mmmu_1020 = hashes.get("MMMU_DEV_VAL") == MMMU_1020_SHA256
    judges = {b: args.choice_judge if b in JUDGE_35_DATASETS else args.math_judge for b in datasets}
    model_config = {
        "class": "Qwen2VLChat", "model_path": str(model),
        "min_pixels": 1003520, "max_pixels": 12845056, "max_new_tokens": 2048,
        "top_p": .001, "top_k": 1, "temperature": .01,
        "repetition_penalty": 1, "use_custom_prompt": False,
    }
    protocol = {
        "model": model_config, "gpus": gpu_ids, "dataset_sha256": hashes,
        "judges": judges, "mmmu_reported_split": "validation",
        "mmmu_1020_terminal_answer_priority": use_mmmu_1020,
    }
    print(json.dumps(protocol, indent=2), flush=True)
    if args.dry_run:
        return 0
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in your environment for judge scoring")
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    env = os.environ.copy()
    env.update(
        LMUData=str(data_root), CAPPO_LOCAL_DATA_ONLY="1", CUDA_VISIBLE_DEVICES=args.gpus,
        MMMU_TERMINAL_ANSWER_PRIORITY=str(int(use_mmmu_1020)),
        TOKENIZERS_PARALLELISM="false", DIST_TIMEOUT="21600",
        PYTHONPATH=os.pathsep.join(filter(None, (str(KIT), env.get("PYTHONPATH")))),
    )
    if env.get("OPENAI_API_BASE"):
        base = env["OPENAI_API_BASE"].rstrip("/")
        env["OPENAI_API_BASE"] = base if base.endswith("/chat/completions") else base + "/chat/completions"
    results = {}
    for index, judge in enumerate(dict.fromkeys(judges.values()), 1):
        group_path = output / f"judge_{index}"
        group_path.mkdir()
        subset = [dataset for dataset in datasets if judges[dataset] == judge]
        config = {
            "model": {"CapPO": model_config},
            "data": {dataset: {"class": BENCHMARKS[dataset], "dataset": dataset} for dataset in subset},
        }
        config_file = group_path / "config.json"
        config_file.write_text(json.dumps(config, indent=2) + "\n")
        command = [sys.executable]
        if len(gpu_ids) > 1:
            command += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={len(gpu_ids)}"]
        command += [str(KIT / "run.py"), "--config", str(config_file), "--judge", judge,
                    "--work-dir", str(group_path / "outputs"), "--api-nproc", str(args.api_nproc)]
        subprocess.run(command, cwd=KIT, env=dict(env, LOCAL_LLM=judge), check=True)
        for dataset in subset:
            results[dataset] = read_score(group_path / "outputs", dataset)
        (output / "scores.json").write_text(json.dumps([results[b] for b in datasets if b in results], indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
