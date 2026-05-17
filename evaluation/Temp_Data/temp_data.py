"""Recompute CoSPlay metrics from downloaded temp_data JSON matrices.

The Hugging Face `temp_data/{main,generalization,scaling,tts}` artifacts already
store execution matrices. This script reads those matrices directly; it does not
call a model and does not execute generated code again.

Typical usage:

    python evaluation/Temp_Data/temp_data.py \
        --root /path/to/CosPlay/temp_data/scaling \
        --out-dir outputs/temp_data_scaling

The core fields consumed from each JSON row are:

* generated_code[i]: code candidate i.
* case_bool_table[i][j]: whether code i passes generated unit test j.
* case_is_valid[j]: optional mask for valid generated unit tests.
* test_bool_table[i][t]: whether code i passes official/held-out test t.
* new_bon_cluster_info: cached output-consensus selections when present.
  The stored key `new_bon` is reported as `cluster_*`, matching the paper's
  output-consensus clustering selector. The stored `new_bon_front/back` variants
  are emitted only as `debug_cluster_front/back_*` diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PASS_AT_K = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_AUTO_SCALES = (1, 2, 4, 8, 16, 32, 64)
SPLIT_NAMES = {"main", "generalization", "scaling", "tts"}
CLUSTER_OUTPUT_NAMES = {
    "new_bon": "cluster",
    "new_bon_front": "debug_cluster_front",
    "new_bon_back": "debug_cluster_back",
}


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y"}:
            return True
        if text in {"0", "false", "f", "no", "n", ""}:
            return False
    return bool(value)


def _bool_row(row: Any) -> list[bool]:
    if row is None:
        return []
    if isinstance(row, str):
        return [_as_bool(ch) for ch in row.strip()]
    if isinstance(row, Iterable):
        return [_as_bool(x) for x in row]
    return [_as_bool(row)]


def _bool_matrix(value: Any) -> list[list[bool]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_bool_row(row) for row in value]
    return []


def _filter_columns(matrix: list[list[bool]], valid_mask: Any) -> list[list[bool]]:
    if not matrix or not isinstance(valid_mask, list):
        return matrix
    n_cols = max((len(row) for row in matrix), default=0)
    if len(valid_mask) != n_cols:
        return matrix
    keep = [i for i, flag in enumerate(valid_mask) if _as_bool(flag)]
    return [[row[i] for i in keep if i < len(row)] for row in matrix]


def _filter_list(items: Any, valid_mask: Any) -> list[Any]:
    if not isinstance(items, list):
        return []
    if not isinstance(valid_mask, list) or len(valid_mask) != len(items):
        return list(items)
    return [item for item, flag in zip(items, valid_mask) if _as_bool(flag)]


def _sum_matrix(matrix: list[list[bool]]) -> int:
    return sum(sum(1 for x in row if x) for row in matrix)


def _pass_at_k_for_task(test_table: list[list[bool]], k: int) -> float:
    n = len(test_table)
    if n == 0:
        return 0.0
    c = sum(1 for row in test_table if all(row))
    k_eff = min(k, n)
    if c == 0 or k_eff == 0:
        return 0.0
    if n - c >= k_eff:
        prob_all_wrong = math.comb(n - c, k_eff) / math.comb(n, k_eff)
    else:
        prob_all_wrong = 0.0
    return 1.0 - prob_all_wrong


def _parse_int_list(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _parse_scales(text: str) -> str | list[tuple[int, int]]:
    if text.lower() == "auto":
        return "auto"
    scales: list[tuple[int, int]] = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if "x" in part:
            left, right = part.split("x", 1)
            scales.append((int(left), int(right)))
        else:
            value = int(part)
            scales.append((value, value))
    return scales


def _auto_scales(max_code: int, max_case: int) -> list[tuple[int, int]]:
    limit = max(1, min(max_code, max_case))
    values = [k for k in DEFAULT_AUTO_SCALES if k <= limit]
    if limit not in values:
        values.append(limit)
    return [(k, k) for k in sorted(set(values))]


def _scale_name(scale: tuple[int, int]) -> str:
    code_k, case_k = scale
    return str(code_k) if code_k == case_k else f"{code_k}x{case_k}"


def _select_bon_row(
    case_table: list[list[bool]],
    test_table: list[list[bool]],
    scale: tuple[int, int],
) -> tuple[int | None, list[bool], list[int]]:
    code_k, case_k = scale
    case_slice = [row[:case_k] for row in case_table[:code_k]]
    test_slice = test_table[:code_k]
    if not test_slice:
        return None, [], []
    if case_slice and case_slice[0]:
        pass_counts = [sum(1 for x in row if x) for row in case_slice]
        best_idx = max(range(len(pass_counts)), key=lambda i: (pass_counts[i], -i))
    else:
        pass_counts = [0 for _ in test_slice]
        best_idx = 0
    if best_idx >= len(test_slice):
        return None, [], pass_counts
    return best_idx, test_slice[best_idx], pass_counts


def _select_oracle_tie_row(
    test_table: list[list[bool]],
    pass_counts: list[int],
) -> tuple[int | None, list[bool], int]:
    if not test_table or not pass_counts:
        return None, [], 0
    max_pass = max(pass_counts)
    top = [i for i, count in enumerate(pass_counts) if count == max_pass]
    if not top:
        top = [0]
    best_idx = max(
        top,
        key=lambda i: (
            _safe_div(sum(1 for x in test_table[i] if x), len(test_table[i])),
            -i,
        ),
    )
    return best_idx, test_table[best_idx], len(top)


class Accumulator:
    def __init__(self, pass_at_k: list[int], scales: str | list[tuple[int, int]]):
        self.pass_at_k = pass_at_k
        self.scales_arg = scales
        self.stats: dict[str, float] = defaultdict(float)
        self.pass_sums: dict[int, float] = defaultdict(float)
        self.pass_nums: dict[int, int] = defaultdict(int)
        self.bon: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.bon_all: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.new_bon: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def add_file(self, path: Path, rows: list[dict[str, Any]]) -> None:
        self.stats["files"] += 1
        for row in rows:
            self.add_task(path, row)

    def add_task(self, path: Path, row: dict[str, Any]) -> None:
        test_table = _bool_matrix(row.get("test_bool_table") or row.get("test_bool_table_rows"))
        case_table_raw = _bool_matrix(row.get("case_bool_table") or row.get("case_bool_table_rows"))
        if not test_table:
            self.stats["skipped_no_test_table"] += 1
            return

        valid_mask = row.get("case_is_valid")
        case_table = _filter_columns(case_table_raw, valid_mask)

        n_code = len(test_table)
        n_test = max((len(r) for r in test_table), default=0)
        n_case = max((len(r) for r in case_table), default=0)
        correct_code_indices = [i for i, r in enumerate(test_table) if all(r)]

        self.stats["tasks"] += 1
        self.stats["code_candidates"] += n_code
        self.stats["generated_tests"] += n_case
        self.stats["correct_code_candidates"] += len(correct_code_indices)
        self.stats["test_cells"] += n_code * n_test
        self.stats["test_true_cells"] += _sum_matrix(test_table)

        for k in self.pass_at_k:
            self.pass_sums[k] += _pass_at_k_for_task(test_table, k)
            self.pass_nums[k] += 1

        case_inputs = _filter_list(row.get("case_input"), valid_mask)
        if case_inputs:
            self.stats["ut_rank_sum"] += len(set(str(x) for x in case_inputs))
            self.stats["ut_rank_norm_sum"] += len(set(str(x) for x in case_inputs)) / len(case_inputs)
            self.stats["ut_rank_num"] += 1

        if correct_code_indices and n_case:
            correct_rows = [
                case_table[i]
                for i in correct_code_indices
                if i < len(case_table)
            ]
            if correct_rows:
                correct_case_indices = [
                    j
                    for j in range(n_case)
                    if all(j < len(r) and r[j] for r in correct_rows)
                ]
                self.stats["ut_num"] += n_case
                self.stats["ut_correct"] += len(correct_case_indices)
                self.stats["ut_cells"] += len(correct_rows) * n_case
                self.stats["ut_true_cells"] += _sum_matrix(correct_rows)

                wrong_code_indices = [
                    i for i in range(len(case_table)) if i not in correct_code_indices
                ]
                wrong_case_indices = [
                    j for j in range(n_case) if j not in correct_case_indices
                ]
                if wrong_code_indices and correct_case_indices:
                    cells = 0
                    rejects = 0
                    for i in wrong_code_indices:
                        for j in correct_case_indices:
                            if i < len(case_table) and j < len(case_table[i]):
                                cells += 1
                                rejects += 0 if case_table[i][j] else 1
                    self.stats["p01_reject_cells"] += rejects
                    self.stats["p01_cells"] += cells
                if wrong_code_indices and wrong_case_indices:
                    cells = 0
                    accepts = 0
                    for i in wrong_code_indices:
                        for j in wrong_case_indices:
                            if i < len(case_table) and j < len(case_table[i]):
                                cells += 1
                                accepts += 1 if case_table[i][j] else 0
                    self.stats["p00_accept_cells"] += accepts
                    self.stats["p00_cells"] += cells

        scales = self.scales_arg
        if scales == "auto":
            scales = _auto_scales(n_code, n_case)
        for scale in scales:
            name = _scale_name(scale)
            selected_idx, selected_test_row, pass_counts = _select_bon_row(
                case_table, test_table, scale
            )
            if selected_idx is not None and selected_test_row:
                st = self.bon[name]
                st["num"] += 1
                st["score"] += 1 if all(selected_test_row) else 0
                st["acc_num"] += len(selected_test_row)
                st["acc_score"] += sum(1 for x in selected_test_row if x)

            oracle_idx, oracle_row, tie_count = _select_oracle_tie_row(
                test_table[: scale[0]], pass_counts
            )
            if oracle_idx is not None and oracle_row:
                st = self.bon_all[name]
                st["num"] += 1
                st["score"] += 1 if all(oracle_row) else 0
                st["acc_num"] += len(oracle_row)
                st["acc_score"] += sum(1 for x in oracle_row if x)
                st["tie_count_sum"] += tie_count
                st["ties_gt1"] += 1 if tie_count > 1 else 0

        info = row.get("new_bon_cluster_info")
        if isinstance(info, dict):
            for key in ("new_bon", "new_bon_front", "new_bon_back"):
                selected = info.get(key, {})
                if not isinstance(selected, dict):
                    continue
                idx = selected.get("selected_index")
                if not isinstance(idx, int) or idx < 0 or idx >= len(test_table):
                    continue
                test_row = test_table[idx]
                st = self.new_bon[key]
                st["num"] += 1
                st["score"] += 1 if all(test_row) else 0
                st["acc_num"] += len(test_row)
                st["acc_score"] += sum(1 for x in test_row if x)

    def finalize(self) -> dict[str, float]:
        out: dict[str, float] = {
            "files": self.stats["files"],
            "tasks": self.stats["tasks"],
            "code_candidates": self.stats["code_candidates"],
            "generated_tests": self.stats["generated_tests"],
            "code_acc": _safe_div(
                self.stats["correct_code_candidates"],
                self.stats["code_candidates"],
            ),
            "code_accumulate": _safe_div(
                self.stats["test_true_cells"],
                self.stats["test_cells"],
            ),
            "ut_acc": _safe_div(self.stats["ut_correct"], self.stats["ut_num"]),
            "ut_accumulate": _safe_div(
                self.stats["ut_true_cells"],
                self.stats["ut_cells"],
            ),
            "ut_rank_avg": _safe_div(
                self.stats["ut_rank_sum"],
                self.stats["ut_rank_num"],
            ),
            "ut_rank_norm_avg": _safe_div(
                self.stats["ut_rank_norm_sum"],
                self.stats["ut_rank_num"],
            ),
            "p01_reject_rate": _safe_div(
                self.stats["p01_reject_cells"],
                self.stats["p01_cells"],
            ),
            "estimated_p01_false_rejection": 1
            - _safe_div(self.stats["p01_reject_cells"], self.stats["p01_cells"]),
            "p00_false_acceptance": _safe_div(
                self.stats["p00_accept_cells"],
                self.stats["p00_cells"],
            ),
        }

        for k in self.pass_at_k:
            out[f"pass@{k}"] = _safe_div(self.pass_sums[k], self.pass_nums[k])

        for name, st in sorted(self.bon.items()):
            out[f"bon_{name}_acc"] = _safe_div(st["score"], st["num"])
            out[f"bon_{name}_accumulate"] = _safe_div(st["acc_score"], st["acc_num"])
        for name, st in sorted(self.bon_all.items()):
            out[f"bon_all_{name}_acc"] = _safe_div(st["score"], st["num"])
            out[f"bon_all_{name}_accumulate"] = _safe_div(st["acc_score"], st["acc_num"])
            out[f"bon_all_{name}_avg_top_candidates"] = _safe_div(
                st["tie_count_sum"], st["num"]
            )
            out[f"bon_all_{name}_ties_gt1"] = st["ties_gt1"]
        for key, st in sorted(self.new_bon.items()):
            output_name = CLUSTER_OUTPUT_NAMES.get(key, key)
            out[f"{output_name}_acc"] = _safe_div(st["score"], st["num"])
            out[f"{output_name}_accumulate"] = _safe_div(
                st["acc_score"], st["acc_num"]
            )
        return out


def _read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return [x for x in data["data"] if isinstance(x, dict)]
        return [data]
    return []


def _iter_json_files(root: Path, kind: str, round_id: str | None) -> list[Path]:
    if root.is_file():
        return [root]
    patterns: list[str] = []
    if kind in {"final", "both"}:
        patterns.append("**/outputs_results_eval_*.json")
    if kind in {"round", "both"}:
        round_glob = "*" if round_id is None else round_id.zfill(2)
        patterns.append(f"**/self_play_v2_rounds/round_{round_glob}_results_eval_*.json")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    return sorted(set(files))


def _metadata(path: Path, root: Path) -> dict[str, str]:
    parts = path.parts
    split = ""
    setting = ""
    run = ""
    if "temp_data" in parts:
        for i, part in enumerate(parts):
            if part in SPLIT_NAMES:
                split = part
                setting = parts[i + 1] if i + 1 < len(parts) else ""
                run = parts[i + 2] if i + 2 < len(parts) else ""
                break
    if not split:
        rel = path.relative_to(root if root.is_dir() else root.parent)
        rel_parts = rel.parts
        if rel_parts and rel_parts[0] in SPLIT_NAMES:
            split = rel_parts[0]
            setting = rel_parts[1] if len(rel_parts) > 1 else ""
            run = rel_parts[2] if len(rel_parts) > 2 else ""
        else:
            setting = rel_parts[0] if rel_parts else ""
            run = rel_parts[1] if len(rel_parts) > 1 else ""

    filename = path.name
    round_match = re.search(r"round_(\d+)_results_eval", filename)
    dataset_match = re.search(
        r"(LB_LCB_CC_CF_200(?:_seed_\d+)?_chunk_\d+|"
        r"(?:CodeContests|CodeForces|LiveBench|LiveCodeBench)(?:_chunk_\d+)?)",
        filename,
    )
    return {
        "split": split,
        "setting": setting,
        "run": run,
        "dataset": dataset_match.group(1) if dataset_match else "",
        "round": round_match.group(1) if round_match else "final",
        "path": str(path),
    }


def _new_acc(pass_at_k: list[int], scales: str | list[tuple[int, int]]) -> Accumulator:
    return Accumulator(pass_at_k=pass_at_k, scales=scales)


def _row_with_meta(meta: dict[str, str], metrics: dict[str, float]) -> dict[str, Any]:
    row: dict[str, Any] = dict(meta)
    row.update(metrics)
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = ["split", "setting", "run", "dataset", "round", "path"]
    fieldnames = list(preferred)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_table(rows: list[dict[str, Any]], keys: list[str], limit: int = 30) -> None:
    if not rows:
        print("No rows.")
        return
    widths = {k: len(k) for k in keys}
    for row in rows[:limit]:
        for key in keys:
            widths[key] = max(widths[key], len(str(row.get(key, ""))))
    print("  ".join(k.ljust(widths[k]) for k in keys))
    print("  ".join("-" * widths[k] for k in keys))
    for row in rows[:limit]:
        print("  ".join(str(row.get(k, "")).ljust(widths[k]) for k in keys))
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more rows")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute CoSPlay metrics from downloaded temp_data JSON matrices."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Path to temp_data, temp_data/main, temp_data/scaling, or one JSON file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("temp_data_metrics"),
        help="Directory for CSV/JSON summaries.",
    )
    parser.add_argument(
        "--kind",
        choices=["final", "round", "both"],
        default="final",
        help="Use final outputs JSONs, saved round JSONs, or both.",
    )
    parser.add_argument(
        "--round",
        dest="round_id",
        default=None,
        help="Round id for --kind round, e.g. 05. Omit to include all rounds.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=sorted(SPLIT_NAMES),
        help="Optional split filter. Can be repeated.",
    )
    parser.add_argument(
        "--setting",
        action="append",
        help="Optional setting directory filter. Can be repeated.",
    )
    parser.add_argument(
        "--pass-at-k",
        default=",".join(str(k) for k in DEFAULT_PASS_AT_K),
        help="Comma-separated pass@k values.",
    )
    parser.add_argument(
        "--scales",
        default="auto",
        help="Comma-separated BoN scales, e.g. auto, 16, 2,4,8,16, or 16x32.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Debug limit on number of JSON files to process.",
    )
    args = parser.parse_args()

    root = args.root
    files = _iter_json_files(root, args.kind, args.round_id)
    pass_at_k = _parse_int_list(args.pass_at_k)
    scales = _parse_scales(args.scales)

    selected_files = []
    for path in files:
        meta = _metadata(path, root)
        if args.split and meta["split"] not in set(args.split):
            continue
        if args.setting and meta["setting"] not in set(args.setting):
            continue
        selected_files.append(path)
    if args.max_files:
        selected_files = selected_files[: args.max_files]

    if not selected_files:
        raise SystemExit(f"No JSON files found under {root}")

    per_file_rows: list[dict[str, Any]] = []
    group_acc: dict[tuple[str, str, str], Accumulator] = {}
    setting_acc: dict[tuple[str, str], Accumulator] = {}
    all_acc = _new_acc(pass_at_k, scales)

    for idx, path in enumerate(selected_files, start=1):
        print(f"[{idx}/{len(selected_files)}] {path}")
        rows = _read_json(path)
        meta = _metadata(path, root)
        file_acc = _new_acc(pass_at_k, scales)
        file_acc.add_file(path, rows)
        all_acc.add_file(path, rows)
        per_file_rows.append(_row_with_meta(meta, file_acc.finalize()))

        run_key = (meta["split"], meta["setting"], meta["run"])
        group_acc.setdefault(run_key, _new_acc(pass_at_k, scales)).add_file(path, rows)
        setting_key = (meta["split"], meta["setting"])
        setting_acc.setdefault(setting_key, _new_acc(pass_at_k, scales)).add_file(path, rows)

    per_run_rows = []
    for (split, setting, run), acc in sorted(group_acc.items()):
        per_run_rows.append(
            _row_with_meta(
                {"split": split, "setting": setting, "run": run, "dataset": "", "round": ""},
                acc.finalize(),
            )
        )

    per_setting_rows = []
    for (split, setting), acc in sorted(setting_acc.items()):
        per_setting_rows.append(
            _row_with_meta(
                {"split": split, "setting": setting, "run": "", "dataset": "", "round": ""},
                acc.finalize(),
            )
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "per_file_metrics.csv", per_file_rows)
    _write_csv(args.out_dir / "per_run_metrics.csv", per_run_rows)
    _write_csv(args.out_dir / "per_setting_metrics.csv", per_setting_rows)
    with (args.out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "root": str(root),
                "kind": args.kind,
                "round": args.round_id,
                "num_files": len(selected_files),
                "overall": all_acc.finalize(),
                "per_setting": per_setting_rows,
            },
            f,
            indent=2,
        )

    print("\nPer-setting summary:")
    summary_keys = [
        "split",
        "setting",
        "files",
        "tasks",
        "pass@1",
        "bon_16_acc",
        "cluster_acc",
        "ut_acc",
    ]
    _print_table(per_setting_rows, summary_keys)
    print(f"\nWrote summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
