import argparse
import csv
import json
import os
import os.path as osp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    official_name: str
    config_path: str


ORDERED_MODELS: List[ModelSpec] = [
    ModelSpec("Gemini-3-Pro", "configs/ssi_bench/gemini_3_pro.json"),
    ModelSpec("Gemini-3-Flash", "configs/ssi_bench/gemini_3_flash.json"),
    ModelSpec("Gemini-2.5-Pro", "configs/ssi_bench/gemini_2_5_pro.json"),
    ModelSpec("Gemini-2.5-Flash", "configs/ssi_bench/gemini_2_5_flash.json"),
    ModelSpec("GPT-5.2", "configs/ssi_bench/gpt_5_2.json"),
    ModelSpec("GPT-5 mini", "configs/ssi_bench/gpt_5_mini.json"),
    ModelSpec("GPT-4.1", "configs/ssi_bench/gpt_4_1.json"),
    ModelSpec("GPT-4o", "configs/ssi_bench/gpt_4o.json"),
    ModelSpec("Claude-Sonnet-4.5", "configs/ssi_bench/claude_sonnet_4_5.json"),
    ModelSpec("Seed-1.8", "configs/ssi_bench/seed_1_8.json"),
    ModelSpec("GLM-4.6V", "configs/ssi_bench/glm_4_6v.json"),
    ModelSpec("GLM-4.6V-Flash", "configs/ssi_bench/glm_4_6v_flash.json"),
    ModelSpec("GLM-4.5V", "configs/ssi_bench/glm_4_5v.json"),
    ModelSpec("Qwen3-VL-235B-A22B", "configs/ssi_bench/qwen3_vl_235b_a22b.json"),
    ModelSpec("Qwen3-VL-30B-A3B", "configs/ssi_bench/qwen3_vl_30b_a3b.json"),
    ModelSpec("Qwen3-VL-8B", "configs/ssi_bench/qwen3_vl_8b.json"),
    ModelSpec("Qwen3-VL-4B", "configs/ssi_bench/qwen3_vl_4b.json"),
    ModelSpec("Qwen3-VL-2B", "configs/ssi_bench/qwen3_vl_2b.json"),
    ModelSpec("InternVL3.5-241B-A28B", "configs/ssi_bench/internvl3_5_241b_a28b.json"),
    ModelSpec("InternVL3.5-30B-A3B", "configs/ssi_bench/internvl3_5_30b_a3b.json"),
    ModelSpec("InternVL3.5-38B", "configs/ssi_bench/internvl3_5_38b.json"),
    ModelSpec("InternVL3.5-14B", "configs/ssi_bench/internvl3_5_14b.json"),
    ModelSpec("InternVL3.5-8B", "configs/ssi_bench/internvl3_5_8b.json"),
    ModelSpec("InternVL3.5-4B", "configs/ssi_bench/internvl3_5_4b.json"),
    ModelSpec("InternVL3.5-2B", "configs/ssi_bench/internvl3_5_2b.json"),
    ModelSpec("Llama-4-Scout-17B-16E", "configs/ssi_bench/llama_4_scout_17b_16e.json"),
    ModelSpec("Gemma-3-27B", "configs/ssi_bench/gemma_3_27b.json"),
    ModelSpec("Gemma-3-12B", "configs/ssi_bench/gemma_3_12b.json"),
    ModelSpec("Gemma-3-4B", "configs/ssi_bench/gemma_3_4b.json"),
    ModelSpec("LLaVA-Onevision-72B", "configs/ssi_bench/llava_onevision_72b.json"),
    ModelSpec("LLaVA-Onevision-7B", "configs/ssi_bench/llava_onevision_7b.json"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", type=str, default="outputs", help="Same as run.py --work-dir (default: outputs).")
    p.add_argument("--dataset", type=str, default="SSI_Bench", help="Dataset name (default: SSI_Bench).")
    p.add_argument(
        "--out-dir",
        type=str,
        default=osp.join("outputs", "ssi_bench_summary"),
        help="Directory to write summary csv files (default: outputs/ssi_bench_summary).",
    )
    p.add_argument(
        "--eval-id",
        type=str,
        default=None,
        help="Optional eval id (e.g. T20260107_G95ac9aab). If unset, uses the latest result per model.",
    )
    return p.parse_args()


def _load_model_name_from_config(config_path: str) -> Optional[str]:
    if not osp.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, dict) or not model_cfg:
        return None
    return next(iter(model_cfg.keys()))


def _find_acc_files(work_dir: str, model_name: str, dataset: str) -> List[Path]:
    root = Path(work_dir) / model_name
    if not root.exists():
        return []

    expected = f"{model_name}_{dataset}_acc.csv"
    matches: List[Path] = []

    # New layout: outputs/<model>/<eval_id>/<model>_<dataset>_acc.csv
    matches.extend(root.glob(f"*/{expected}"))
    # Legacy layout: outputs/<model>/<model>_<dataset>_acc.csv
    matches.extend(root.glob(expected))
    return [p for p in matches if p.is_file()]


def _pick_acc_file(work_dir: str, model_name: str, dataset: str, eval_id: Optional[str]) -> Optional[Path]:
    files = _find_acc_files(work_dir, model_name, dataset)
    if not files:
        return None
    if eval_id is not None:
        for p in files:
            if p.parent.name == eval_id:
                return p
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _parse_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().strip('"').strip("'")
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _normalize_row_to_percent(row: Dict[str, str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    metric_vals: List[float] = []
    for k, v in row.items():
        if k == "acc_type":
            continue
        fv = _parse_float(v)
        if fv is None:
            continue
        metric_vals.append(fv)
        out[k] = fv

    mx = max(metric_vals) if metric_vals else 0.0
    if mx <= 1.0:
        for k in list(out.keys()):
            out[k] = out[k] * 100.0
    return out


def _extract_metric_row(rows: List[Dict[str, str]], acc_type: str) -> Optional[Dict[str, float]]:
    if not rows:
        return None
    has_acc_type = "acc_type" in rows[0]
    if has_acc_type:
        for r in rows:
            if str(r.get("acc_type", "")).strip() == acc_type:
                return _normalize_row_to_percent(r)
        return None

    if acc_type != "task_acc":
        return None
    return _normalize_row_to_percent(rows[0])


def _metric_cols_in_order(rows: Iterable[Dict]) -> List[str]:
    seen = set()
    out: List[str] = []
    for r in rows:
        for k in r.keys():
            if k in {"official_name", "model", "acc_type"}:
                continue
            if k not in seen:
                out.append(k)
                seen.add(k)
    return out


def _build_table(work_dir: str, dataset: str, specs: List[ModelSpec], acc_type: str, eval_id: Optional[str]) -> List[Dict]:
    rows: List[Dict] = []
    for spec in specs:
        model_name = _load_model_name_from_config(spec.config_path)
        if not model_name:
            continue
        acc_path = _pick_acc_file(work_dir, model_name, dataset, eval_id=eval_id)
        if acc_path is None:
            continue

        raw_rows = _read_csv_rows(acc_path)
        metrics = _extract_metric_row(raw_rows, acc_type=acc_type)
        if metrics is None:
            continue

        record = {"official_name": spec.official_name, "model": model_name}
        record.update(metrics)
        rows.append(record)

    metric_cols = _metric_cols_in_order(rows)
    ordered = []
    for r in rows:
        ordered.append({k: r.get(k) for k in ["official_name", "model"] + metric_cols})
    return ordered


def _write_table(path: str, rows: List[Dict], metric_cols: List[str]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["official_name", "model"] + metric_cols)
        writer.writeheader()
        for r in rows:
            out = {"official_name": r.get("official_name"), "model": r.get("model")}
            for c in metric_cols:
                v = r.get(c)
                out[c] = "" if v is None else f"{float(v):.6f}"
            writer.writerow(out)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    task_rows = _build_table(
        work_dir=args.work_dir, dataset=args.dataset, specs=ORDERED_MODELS, acc_type="task_acc", eval_id=args.eval_id
    )
    pairwise_rows = _build_table(
        work_dir=args.work_dir,
        dataset=args.dataset,
        specs=ORDERED_MODELS,
        acc_type="pairwise_acc",
        eval_id=args.eval_id,
    )

    task_out = osp.join(args.out_dir, f"{args.dataset}_task_acc.csv")
    pairwise_out = osp.join(args.out_dir, f"{args.dataset}_pairwise_acc.csv")

    task_cols = _metric_cols_in_order(task_rows)
    pairwise_cols = _metric_cols_in_order(pairwise_rows)
    _write_table(task_out, task_rows, task_cols)
    _write_table(pairwise_out, pairwise_rows, pairwise_cols)
    print(f"Wrote: {task_out}")
    print(f"Wrote: {pairwise_out}")


if __name__ == "__main__":
    main()
