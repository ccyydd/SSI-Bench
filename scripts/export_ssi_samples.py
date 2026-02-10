import argparse
import base64
import json
import os.path as osp
from datetime import datetime


TASKS = [
    "area",
    "cycle_length",
    "dimension",
    "ground_angle",
    "ground_height",
    "hop_distance",
    "multi_view_geometric",
    "multi_view_topological",
    "relative_distance",
    "volume",
]


def _b64_of_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-file", type=str, required=True, help="The prediction xlsx/csv produced by running SSI_Bench.")
    p.add_argument("--n", type=int, default=3, help="Samples per task (default: 3).")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling (default: 0).")
    p.add_argument("--out", type=str, default=None, help="Output json path. Default: next to eval file.")
    p.add_argument(
        "--embed-images",
        action="store_true",
        help="Embed decoded images as base64 in the JSON (large). Default: store file paths only.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    from vlmeval.dataset import build_dataset
    from vlmeval.smp import load
    from vlmeval.dataset.ssi_bench import _parse_python_list_from_text

    eval_df = load(args.eval_file)
    if "task" not in eval_df.columns or "index" not in eval_df.columns:
        raise ValueError("`eval-file` must contain at least `index` and `task` columns.")

    dataset = build_dataset("SSI_Bench")
    base_map = {str(r["index"]): r for r in dataset.data.to_dict(orient="records")}

    rng = __import__("random").Random(args.seed)

    samples = []
    for task in TASKS:
        sub = eval_df[eval_df["task"].astype(str) == task].copy()
        if "prediction" in sub.columns:
            sub = sub[~sub["prediction"].isna()]
        sub = sub[sub["index"].apply(lambda x: str(x) in base_map)]
        records = sub.to_dict(orient="records")
        rng.shuffle(records)
        picked = records[: args.n]

        for row in picked:
            base = base_map[str(row["index"])]
            msgs = dataset.build_prompt(base)

            input_msgs = []
            for msg in msgs:
                if msg.get("type") == "image":
                    item = {"type": "image", "value": msg.get("value")}
                    if args.embed_images and msg.get("value") and osp.exists(msg["value"]):
                        item["base64"] = _b64_of_file(msg["value"])
                    input_msgs.append(item)
                else:
                    input_msgs.append({"type": "text", "value": msg.get("value")})

            gt = _parse_python_list_from_text(row.get("answer", None))
            pred_list = _parse_python_list_from_text(row.get("prediction", None))

            samples.append(
                {
                    "task": task,
                    "index": row.get("index"),
                    "category": row.get("category"),
                    "sample": row.get("sample"),
                    "annotation_color": row.get("annotation_color"),
                    "template_name": row.get("question"),
                    "input": input_msgs,
                    "gt_answer": gt,
                    "output_raw": row.get("prediction"),
                    "output_list": pred_list,
                }
            )

    if args.out is None:
        base, _ = osp.splitext(args.eval_file)
        args.out = base + f"_samples_{args.n}x{len(TASKS)}.json"

    out_obj = {
        "dataset": "SSI_Bench",
        "eval_file": args.eval_file,
        "tasks": TASKS,
        "n_per_task": args.n,
        "seed": args.seed,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(samples),
        "samples": samples,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(samples)} samples to {args.out}")


if __name__ == "__main__":
    main()

