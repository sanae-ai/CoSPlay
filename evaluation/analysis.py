

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from bon_valid_ut_eval import _get_valid_mask, _select_best_code


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


class TfidfEncoder:
    def __init__(self, docs: List[str]) -> None:
        df = Counter()
        for doc in docs:
            tokens = set(_tokenize(doc))
            df.update(tokens)
        n_docs = max(len(docs), 1)
        self.idf = {t: math.log((1 + n_docs) / (1 + c)) + 1.0 for t, c in df.items()}
        self.cache: Dict[str, Tuple[Dict[str, float], float]] = {}

    def encode(self, text: str) -> Tuple[Dict[str, float], float]:
        if text in self.cache:
            return self.cache[text]
        tokens = _tokenize(text)
        if not tokens:
            self.cache[text] = ({}, 0.0)
            return self.cache[text]
        tf = Counter(tokens)
        total = len(tokens)
        vec = {}
        for t, c in tf.items():
            idf = self.idf.get(t)
            if idf is None:
                continue
            vec[t] = (c / total) * idf
        norm = math.sqrt(sum(v * v for v in vec.values()))
        self.cache[text] = (vec, norm)
        return vec, norm

    @staticmethod
    def cosine(a: Tuple[Dict[str, float], float], b: Tuple[Dict[str, float], float]) -> float:
        vec_a, norm_a = a
        vec_b, norm_b = b
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a
        dot = sum(v * vec_b.get(k, 0.0) for k, v in vec_a.items())
        return dot / (norm_a * norm_b)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute similarity between problem and codes (correct & selected)."
    )
    parser.add_argument("--input", default=None, help="Path to the evaluation JSON file.")
    parser.add_argument("--input-dir", default=None, help="Directory containing evaluation JSON files.")
    parser.add_argument(
        "--recursive",
        action="store_const",
        const=True,
        default=None,
        help="Search JSON files recursively under input-dir.",
    )
    parser.add_argument("--config", default=None, help="Path to a JSON config file.")
    parser.add_argument("--max-codes", type=int, default=None, help="Use only the first N codes.")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Use only the first M valid UT cases after masking.",
    )
    parser.add_argument(
        "--min-ut-pass-rate",
        type=float,
        default=None,
        help="Drop valid UT columns with pass rate below this threshold.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Comma-separated or JSON list of thresholds, e.g. \"[null,0.1,0.2]\".",
    )
    parser.add_argument("--output-dir", default=None, help="Override output base directory.")
    parser.add_argument(
        "--no-valid-mask",
        action="store_const",
        const=True,
        default=None,
        help="Ignore valid-UT mask and use all UT columns.",
    )
    return parser.parse_args()


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must be a JSON object.")
    return cfg


def _parse_thresholds(value: Any) -> Optional[List[Optional[float]]]:
    if value is None:
        return None
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
            raw = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            raw = [v.strip() for v in value.split(",") if v.strip()]
    else:
        raw = [value]

    out: List[Optional[float]] = []
    for item in raw:
        if item is None:
            out.append(None)
            continue
        if isinstance(item, str):
            if item.lower() in ("none", "null"):
                out.append(None)
                continue
            out.append(float(item))
            continue
        out.append(float(item))
    return out


def _merge_args_with_config(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    merged = argparse.Namespace()
    merged.config = args.config
    merged.input = args.input if args.input is not None else cfg.get("input")
    merged.input_dir = args.input_dir if args.input_dir is not None else cfg.get("input_dir")
    if args.recursive is not None:
        merged.recursive = args.recursive
    else:
        merged.recursive = bool(cfg.get("recursive", False))
    merged.max_codes = args.max_codes if args.max_codes is not None else cfg.get("max_codes")
    merged.max_cases = args.max_cases if args.max_cases is not None else cfg.get("max_cases")
    merged.min_ut_pass_rate = (
        args.min_ut_pass_rate
        if args.min_ut_pass_rate is not None
        else cfg.get("min_ut_pass_rate")
    )
    merged.thresholds = (
        _parse_thresholds(args.thresholds)
        if args.thresholds is not None
        else _parse_thresholds(cfg.get("thresholds"))
    )
    if args.no_valid_mask is not None:
        merged.no_valid_mask = args.no_valid_mask
    else:
        merged.no_valid_mask = bool(cfg.get("no_valid_mask", False))
    merged.output_dir = args.output_dir if args.output_dir is not None else cfg.get("output_dir")
    return merged


def _format_threshold_tag(value: Optional[float]) -> str:
    if value is None:
        return "thr_none"
    return f"thr_{value}".replace(".", "p")


def _resolve_path(path: Optional[str], base_dir: str) -> Optional[str]:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _iter_json_files(input_dir: str, recursive: bool) -> Iterable[str]:
    if recursive:
        for root, _, files in os.walk(input_dir):
            for name in files:
                if name.endswith(".json"):
                    yield os.path.join(root, name)
    else:
        for name in os.listdir(input_dir):
            if name.endswith(".json"):
                yield os.path.join(input_dir, name)


def _build_output_path(input_path: str, output_dir: Optional[str], threshold: Optional[float]) -> str:
    input_tag = os.path.splitext(os.path.basename(input_path))[0]
    if output_dir:
        base_dir = os.path.join(output_dir, input_tag)
    else:
        base_dir = os.path.join(os.path.dirname(input_path), "result", input_tag)
    os.makedirs(base_dir, exist_ok=True)
    threshold_tag = _format_threshold_tag(threshold)
    return os.path.join(base_dir, f"{input_tag}_{threshold_tag}_sim.txt")


def _prepare_encoder(data: List[dict]) -> TfidfEncoder:
    docs: List[str] = []
    for record in data:
        question = record.get("question", "")
        if isinstance(question, str):
            docs.append(question)
        codes = record.get("generated_code", [])
        if isinstance(codes, list):
            docs.extend([c for c in codes if isinstance(c, str)])
    return TfidfEncoder(docs)


def _evaluate_one(
    input_path: str,
    min_ut_pass_rate: Optional[float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    with open(input_path, "r") as f:
        data = json.load(f)

    encoder = _prepare_encoder(data)

    correct_sim_sum = 0.0
    correct_sim_cnt = 0
    correct_task_sum = 0.0
    correct_task_cnt = 0

    selected_sim_sum = 0.0
    selected_task_cnt = 0
    bon_score = 0
    bon_num = 0
    bon_acc_score = 0
    bon_acc_num = 0
    sim_bon_score = 0
    sim_bon_num = 0
    sim_bon_acc_score = 0
    sim_bon_acc_num = 0

    skipped = 0

    for record in data:
        test_table = record.get("test_bool_table")
        case_table = record.get("case_bool_table")
        codes = record.get("generated_code")
        question = record.get("question", "")

        if test_table is None or case_table is None or not isinstance(codes, list):
            skipped += 1
            continue

        test_arr = np.array(test_table, dtype=bool)
        if args.max_codes is not None:
            test_arr = test_arr[: args.max_codes, :]
            codes = codes[: args.max_codes]

        correct_idx = np.where(test_arr.all(axis=1))[0].tolist()
        if correct_idx:
            per_task_vals = []
            q_vec = encoder.encode(question)
            for idx in correct_idx:
                if idx >= len(codes):
                    continue
                code_text = codes[idx]
                if not isinstance(code_text, str):
                    continue
                sim = encoder.cosine(q_vec, encoder.encode(code_text))
                correct_sim_sum += sim
                correct_sim_cnt += 1
                per_task_vals.append(sim)
            if per_task_vals:
                correct_task_sum += sum(per_task_vals) / len(per_task_vals)
                correct_task_cnt += 1

        valid_mask = None if args.no_valid_mask else _get_valid_mask(record)
        best_idx, _ = _select_best_code(
            case_table,
            valid_mask,
            args.max_codes,
            args.max_cases,
            min_ut_pass_rate,
        )
        if best_idx < len(codes) and isinstance(codes[best_idx], str):
            q_vec = encoder.encode(question)
            sim = encoder.cosine(q_vec, encoder.encode(codes[best_idx]))
            selected_sim_sum += sim
            selected_task_cnt += 1
            best_row = test_arr[best_idx, :]
            bon_score += int(best_row.all())
            bon_num += 1
            bon_acc_score += int(best_row.sum())
            bon_acc_num += int(len(best_row))


        if codes:
            case_arr = np.array(case_table, dtype=bool)
            if args.max_codes is not None:
                case_arr = case_arr[: args.max_codes, :]
            if valid_mask is not None:
                valid_indices = [i for i, v in enumerate(valid_mask) if v]
                case_arr = case_arr[:, valid_indices] if valid_indices else case_arr[:, :0]
            if args.max_cases is not None:
                case_arr = case_arr[:, : args.max_cases]
            if min_ut_pass_rate is not None and case_arr.shape[1] > 0:
                ut_pass_rates = case_arr.mean(axis=0)
                keep_mask = ut_pass_rates >= min_ut_pass_rate
                case_arr = case_arr[:, keep_mask] if keep_mask.any() else case_arr[:, :0]

            if case_arr.shape[0] > 0:
                if case_arr.shape[1] == 0:
                    top_candidates = list(range(case_arr.shape[0]))
                else:
                    pass_counts = case_arr.sum(axis=1)
                    max_pass = pass_counts.max()
                    top_candidates = np.where(pass_counts == max_pass)[0].tolist()

                if top_candidates:
                    if len(top_candidates) == 1:
                        sim_best_idx = int(top_candidates[0])
                    else:
                        q_vec = encoder.encode(question)
                        best_sim = float("-inf")
                        sim_best_idx = int(top_candidates[0])
                        for idx in top_candidates:
                            if idx >= len(codes) or not isinstance(codes[idx], str):
                                continue
                            sim = encoder.cosine(q_vec, encoder.encode(codes[idx]))
                            if sim > best_sim:
                                best_sim = sim
                                sim_best_idx = int(idx)
                    sim_best_row = test_arr[sim_best_idx, :]
                    sim_bon_score += int(sim_best_row.all())
                    sim_bon_num += 1
                    sim_bon_acc_score += int(sim_best_row.sum())
                    sim_bon_acc_num += int(len(sim_best_row))

    correct_avg_codes = correct_sim_sum / correct_sim_cnt if correct_sim_cnt else 0.0
    correct_avg_tasks = correct_task_sum / correct_task_cnt if correct_task_cnt else 0.0
    selected_avg = selected_sim_sum / selected_task_cnt if selected_task_cnt else 0.0
    bon_acc = bon_score / bon_num if bon_num else 0.0
    bon_acc_acc = bon_acc_score / bon_acc_num if bon_acc_num else 0.0
    sim_bon_acc = sim_bon_score / sim_bon_num if sim_bon_num else 0.0
    sim_bon_acc_acc = sim_bon_acc_score / sim_bon_acc_num if sim_bon_acc_num else 0.0

    report_lines = [
        "=== Similarity Report ===",
        f"correct code sim avg (over codes): {correct_avg_codes}",
        f"correct code sim avg (over tasks): {correct_avg_tasks}",
        f"selected code sim avg (over tasks): {selected_avg}",
        f"bon acc (case-selected): {bon_acc}",
        f"bon accumulate acc (case-selected): {bon_acc_acc}",
        f"bon acc (case-tie-sim): {sim_bon_acc}",
        f"bon accumulate acc (case-tie-sim): {sim_bon_acc_acc}",
        f"tasks with correct codes: {correct_task_cnt}",
        f"selected tasks: {selected_task_cnt}",
        f"skipped tasks: {skipped}",
    ]
    if min_ut_pass_rate is not None:
        report_lines.append(f"min UT pass rate threshold: {min_ut_pass_rate}")

    for line in report_lines:
        print(line)

    out_path = _build_output_path(input_path, args.output_dir, min_ut_pass_rate)
    with open(out_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"saved report: {out_path}")

    return {
        "input": input_path,
        "threshold": min_ut_pass_rate,
        "correct_sim_sum": correct_sim_sum,
        "correct_sim_cnt": correct_sim_cnt,
        "correct_task_sum": correct_task_sum,
        "correct_task_cnt": correct_task_cnt,
        "selected_sim_sum": selected_sim_sum,
        "selected_task_cnt": selected_task_cnt,
        "bon_score": bon_score,
        "bon_num": bon_num,
        "bon_acc_score": bon_acc_score,
        "bon_acc_num": bon_acc_num,
        "sim_bon_score": sim_bon_score,
        "sim_bon_num": sim_bon_num,
        "sim_bon_acc_score": sim_bon_acc_score,
        "sim_bon_acc_num": sim_bon_acc_num,
    }


def _write_threshold_csv(
    results: List[Dict[str, Any]],
    thresholds: List[Optional[float]],
    output_path: str,
) -> None:
    agg = {
        thr: {
            "correct_sim_sum": 0.0,
            "correct_sim_cnt": 0,
            "correct_task_sum": 0.0,
            "correct_task_cnt": 0,
            "selected_sim_sum": 0.0,
            "selected_task_cnt": 0,
            "bon_score": 0,
            "bon_num": 0,
            "bon_acc_score": 0,
            "bon_acc_num": 0,
            "sim_bon_score": 0,
            "sim_bon_num": 0,
            "sim_bon_acc_score": 0,
            "sim_bon_acc_num": 0,
        }
        for thr in thresholds
    }
    for r in results:
        thr = r["threshold"]
        bucket = agg[thr]
        for k in bucket:
            bucket[k] += r[k]

    lines = [
        "threshold,correct_sim_avg_codes,correct_sim_avg_tasks,selected_sim_avg_tasks,bon_acc_case,bon_acc_acc_case,bon_acc_case_tie_sim,bon_acc_acc_case_tie_sim,correct_code_cnt,correct_task_cnt,selected_task_cnt"
    ]
    for t in thresholds:
        d = agg[t]
        correct_avg_codes = d["correct_sim_sum"] / d["correct_sim_cnt"] if d["correct_sim_cnt"] else 0.0
        correct_avg_tasks = d["correct_task_sum"] / d["correct_task_cnt"] if d["correct_task_cnt"] else 0.0
        selected_avg = d["selected_sim_sum"] / d["selected_task_cnt"] if d["selected_task_cnt"] else 0.0
        bon_acc = d["bon_score"] / d["bon_num"] if d["bon_num"] else 0.0
        bon_acc_acc = d["bon_acc_score"] / d["bon_acc_num"] if d["bon_acc_num"] else 0.0
        sim_bon_acc = d["sim_bon_score"] / d["sim_bon_num"] if d["sim_bon_num"] else 0.0
        sim_bon_acc_acc = d["sim_bon_acc_score"] / d["sim_bon_acc_num"] if d["sim_bon_acc_num"] else 0.0
        label = "none" if t is None else str(t)
        lines.append(
            f"{label},{correct_avg_codes},{correct_avg_tasks},{selected_avg},"
            f"{bon_acc},{bon_acc_acc},{sim_bon_acc},{sim_bon_acc_acc},"
            f"{d['correct_sim_cnt']},{d['correct_task_cnt']},{d['selected_task_cnt']}"
        )
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = _parse_args()
    if args.config is None:
        args.config = os.path.join(os.path.dirname(__file__), "bon_similarity_eval_config.json")
    cfg = _load_config(args.config)
    args = _merge_args_with_config(args, cfg)
    config_dir = os.path.dirname(os.path.abspath(args.config))
    args.input = _resolve_path(args.input, config_dir)
    args.input_dir = _resolve_path(args.input_dir, config_dir)
    args.output_dir = _resolve_path(args.output_dir, config_dir)

    if args.input_dir:
        if os.path.isfile(args.input_dir):
            input_files = [args.input_dir]
        else:
            input_files = sorted(_iter_json_files(args.input_dir, args.recursive))
    elif args.input:
        input_files = [args.input]
    else:
        raise ValueError("Missing input path. Set input or input_dir in config.")

    thresholds = args.thresholds if args.thresholds is not None else [args.min_ut_pass_rate]
    if thresholds is None:
        thresholds = [None]

    all_results: List[Dict[str, Any]] = []
    for input_path in input_files:
        for thr in thresholds:
            all_results.append(_evaluate_one(input_path, thr, args))

    base_dir = args.output_dir if args.output_dir else os.path.join(os.path.dirname(input_files[0]), "result")
    os.makedirs(base_dir, exist_ok=True)
    csv_path = os.path.join(base_dir, "threshold_sweep_similarity.csv")
    _write_threshold_csv(all_results, thresholds, csv_path)
    print(f"saved sweep csv: {csv_path}")


if __name__ == "__main__":
    main()
