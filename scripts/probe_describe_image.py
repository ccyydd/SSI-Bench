import argparse
import copy as cp
import json
import os
import os.path as osp
from datetime import datetime
from pathlib import Path


def _build_model_from_config(cfg, model_name, use_vllm: bool = False):
    import vlmeval.api
    import vlmeval.vlm
    from vlmeval.config import supported_VLM

    config = cp.deepcopy(cfg[model_name])
    if use_vllm:
        config["use_vllm"] = use_vllm
    if "class" not in config:
        return supported_VLM[model_name](**config)
    cls_name = config.pop("class")
    if hasattr(vlmeval.api, cls_name):
        return getattr(vlmeval.api, cls_name)(**config)
    if hasattr(vlmeval.vlm, cls_name):
        return getattr(vlmeval.vlm, cls_name)(**config)
    raise ValueError(f"Class {cls_name} is not supported in `vlmeval.api` or `vlmeval.vlm`")


def _build_dataset_from_config(cfg, dataset_name):
    import inspect
    import vlmeval.dataset
    from vlmeval.dataset.video_dataset_config import supported_video_datasets

    config = cp.deepcopy(cfg[dataset_name])
    if config == {}:
        return supported_video_datasets[dataset_name]()
    if "class" not in config:
        raise ValueError(f"Invalid dataset config for {dataset_name}: missing `class` field.")
    cls_name = config.pop("class")
    if not hasattr(vlmeval.dataset, cls_name):
        raise ValueError(f"Class {cls_name} is not supported in `vlmeval.dataset`")

    cls = getattr(vlmeval.dataset, cls_name)
    sig = inspect.signature(cls.__init__)
    valid_params = {k: v for k, v in config.items() if k in sig.parameters}
    if getattr(cls, "MODALITY", None) == "VIDEO":
        if valid_params.get("fps", 0) > 0 and valid_params.get("nframe", 0) > 0:
            raise ValueError("fps and nframe should not be set at the same time")
        if valid_params.get("fps", 0) <= 0 and valid_params.get("nframe", 0) <= 0:
            raise ValueError("fps and nframe should be set at least one valid value")
    return cls(**valid_params)


def _override_text_prompt(msgs, new_prompt: str):
    msgs = cp.deepcopy(msgs)
    last_text_idx = None
    for i, msg in enumerate(msgs):
        if isinstance(msg, dict) and msg.get("type") == "text":
            last_text_idx = i

    if last_text_idx is None:
        msgs.append({"type": "text", "value": new_prompt})
    else:
        msgs[last_text_idx]["value"] = new_prompt
    return msgs


def _json_default(obj):
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass

    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            return obj.hex()
    return str(obj)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to the same config JSON you pass to `run.py --config`.")
    p.add_argument("--n", type=int, default=1, help="Number of samples to export per dataset (default: 1).")
    p.add_argument("--work-dir", type=str, default="./outputs", help="Output root (default: ./outputs).")
    p.add_argument("--prompt", type=str, default="描述图片", help="Prompt to replace the original text prompt.")
    p.add_argument("--models", type=str, nargs="*", default=None, help="Optional: only run these model keys in config.")
    p.add_argument("--datasets", type=str, nargs="*", default=None, help="Optional: only run these dataset keys in config.")
    p.add_argument("--use-vllm", action="store_true", help="Pass through to model builder (if supported).")
    p.add_argument("--dry-run", action="store_true", help="Only dump inputs; do not call model.generate.")
    return p.parse_args()


def main():
    args = parse_args()

    from vlmeval.smp import githash, timestr

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_names = list(cfg.get("model", {}).keys())
    dataset_names = list(cfg.get("data", {}).keys())
    if args.models is not None and len(args.models):
        model_names = [m for m in model_names if m in set(args.models)]
    if args.datasets is not None and len(args.datasets):
        dataset_names = [d for d in dataset_names if d in set(args.datasets)]

    if not model_names:
        raise ValueError("No models selected (check `--models` / config).")
    if not dataset_names:
        raise ValueError("No datasets selected (check `--datasets` / config).")

    date, commit_id = timestr("day"), githash(digits=8)
    eval_id = f"T{date}_G{commit_id}_probe_describe_image"

    for model_name in model_names:
        pred_root = osp.join(args.work_dir, model_name, eval_id)
        os.makedirs(pred_root, exist_ok=True)

        model = _build_model_from_config(cfg["model"], model_name, use_vllm=args.use_vllm)

        for dataset_name in dataset_names:
            dataset = _build_dataset_from_config(cfg["data"], dataset_name)

            if hasattr(model, "set_dump_image"):
                model.set_dump_image(dataset.dump_image)

            data_df = dataset.data
            picked = data_df.iloc[: max(args.n, 0)]

            samples = []
            for i in range(len(picked)):
                row = picked.iloc[i]
                idx = row.get("index", None)
                try:
                    import numpy as np

                    if isinstance(idx, np.generic):
                        idx = idx.item()
                except Exception:
                    pass

                if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(dataset_name):
                    struct = model.build_prompt(row, dataset=dataset_name)
                else:
                    struct = dataset.build_prompt(row)

                original_text = None
                for msg in struct:
                    if isinstance(msg, dict) and msg.get("type") == "text":
                        original_text = msg.get("value")

                struct2 = _override_text_prompt(struct, args.prompt)

                output = None
                if not args.dry_run:
                    output = model.generate(message=struct2, dataset=dataset_name)

                samples.append(
                    {
                        "row_i": int(i),
                        "index": int(idx) if isinstance(idx, int) else idx,
                        "original_text_prompt": original_text,
                        "text_prompt": args.prompt,
                        "input": struct2,
                        "output": output,
                    }
                )

            out_path = osp.join(pred_root, f"probe_{dataset_name}_n{args.n}.json")
            out_obj = {
                "config": osp.abspath(args.config),
                "model_name": model_name,
                "dataset_name": dataset_name,
                "eval_id": eval_id,
                "prompt": args.prompt,
                "n": args.n,
                "dry_run": bool(args.dry_run),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "samples": samples,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out_obj, f, ensure_ascii=False, indent=2, default=_json_default)
            print(f"Wrote {len(samples)} samples to {out_path}")


if __name__ == "__main__":
    main()
