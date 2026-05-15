"""Recompute the paper's Signal metric from generated code and UT outputs.

This file intentionally lives under evaluation/Signal/. Because its filename is
signal.py, remove this directory from sys.path before importing evaluation
modules; otherwise Python can shadow the standard-library signal module used by
multiprocessing.
"""

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Optional

# Reuse interfaces from the evaluation directory.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(THIS_DIR)
if THIS_DIR in sys.path:
    sys.path.remove(THIS_DIR)
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from execution import run_all_executions
from metrics import compute_and_log_metrics
import evaluation_config


def str2bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


# ---------------- JSON IO ----------------
def _load_json_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict) and isinstance(obj.get("data"), list):
        items = obj["data"]
    elif isinstance(obj, dict):
        items = [obj]
    else:
        raise ValueError(f"Unsupported JSON structure in {path}: {type(obj)}")

    out = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"Item {i} in {path} is not a dict: {type(it)}")
        out.append(it)
    return out


def _dump_json_items(path: str, items: List[Dict[str, Any]], *, wrap_data: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {"data": items} if wrap_data else items
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _truncate_list_field(item: Dict[str, Any], field: str, keep: Optional[int]) -> None:
    if keep is None:
        return
    value = item.get(field)
    if isinstance(value, list):
        item[field] = value[:keep]


def _truncate_merged_item(
    case_item: Dict[str, Any],
    code_item: Dict[str, Any],
    *,
    code_field: str,
    max_codes: Optional[int] = None,
    max_cases: Optional[int] = None,
) -> None:
    _truncate_list_field(code_item, code_field, max_codes)

    case_fields = [
        "case_input",
        "case_output",
        "case_is_valid",
        "case_text",
        "attack_ideas",
        "full_case_generation",
        "case_input_original",
        "case_output_original",
        "case_output_samples",
        "case_output_samples_extracted",
        "case_code_voting_results",
    ]
    for field in case_fields:
        _truncate_list_field(case_item, field, max_cases)


# ---------------- Core merge: replace case generated_code with code outputs ----------------
def merge_replace_generated_code(
    code_json_path: str,
    case_json_path: str,
    out_json_path: str,
    *,
    id_key: str = "task_id",
    question_key: str = "question",
    code_field: str = "generated_code",
    max_codes: Optional[int] = None,
    max_cases: Optional[int] = None,
    strict: bool = True,
    question_check: bool = True,
    question_normalize_ws: bool = True,
    wrap_data: bool = False,
    verbose: bool = True,
    allow_question_fallback: bool = True,   # NEW
) -> Dict[str, Any]:
    """
    1) Match by task_id first.
    2) If a case item lacks task_id, fall back to question matching when enabled.
    3) Replace the case file's generated_code with the code file's generated_code.
    """
    code_items = _load_json_items(code_json_path)
    case_items = _load_json_items(case_json_path)

    def norm_q(s: Any) -> str:
        if s is None:
            return ""
        s = str(s)
        if question_normalize_ws:
            s = " ".join(s.split())
        return s

    # --- code side index ---
    code_map_tid: Dict[str, Dict[str, Any]] = {}
    code_map_q: Dict[str, List[Dict[str, Any]]] = {}

    dup_ids = 0
    missing_id_in_code = 0
    for it in code_items:
        tid = it.get(id_key, None)
        if tid is None:
            missing_id_in_code += 1
        else:
            tid = str(tid)
            if tid in code_map_tid:
                dup_ids += 1
            else:
                code_map_tid[tid] = it

        q = norm_q(it.get(question_key))
        if q:
            code_map_q.setdefault(q, []).append(it)

    replaced = 0
    missing_in_code = 0
    question_mismatch = 0
    fallback_by_question = 0
    ambiguous_question = 0
    # Track the occurrence index already matched for each case-side question.
    case_q_seen: Dict[str, int] = {}

    for it in case_items:
        tid = it.get(id_key, None)
        q_case_norm = norm_q(it.get(question_key))

        # -------- choose src code item --------
        src = None

        if tid is not None:
            tid = str(tid)
            src = code_map_tid.get(tid)
            if src is None:
                missing_in_code += 1
                if strict:
                    raise KeyError(f"task_id={tid} not found in code file: {code_json_path}")
                if verbose:
                    print(f"[WARN] task_id={tid} not found in code file, skip")
                continue
        else:
            # case missing task_id
            if not allow_question_fallback:
                missing_in_code += 1
                if strict:
                    raise KeyError(f"Case item missing '{id_key}': keys={list(it.keys())[:30]}")
                if verbose:
                    print(f"[WARN] Case missing {id_key}, skip. question={q_case_norm[:120]}")
                continue

            if not q_case_norm:
                missing_in_code += 1
                if strict:
                    raise KeyError(f"Case item missing '{id_key}' and empty '{question_key}'")
                if verbose:
                    print("[WARN] Case missing task_id and empty question, skip.")
                continue

            cands = code_map_q.get(q_case_norm, [])
            # Align by the k-th occurrence for the same question.
            k = case_q_seen.get(q_case_norm, 0)
            case_q_seen[q_case_norm] = k + 1

            if k < len(cands):
                src = cands[k]
                fallback_by_question += 1
                # Fill task_id on the case item to keep downstream processing consistent.
                if id_key in src and src.get(id_key) is not None:
                    it[id_key] = src[id_key]
            else:
                # The case side has more occurrences of this question than the code side.
                missing_in_code += 1
                ambiguous_question += 1
                msg = (
                    f"Case item missing '{id_key}', and question occurrence out of range.\n"
                    f"  question={q_case_norm[:200]}\n"
                    f"  occurrence_index={k}, matched_code_items={len(cands)}"
                )
                if strict:
                    raise KeyError(msg)
                if verbose:
                    print("[WARN]", msg)
                continue

        # -------- optional question consistency check --------
        if question_check:
            q_code_norm = norm_q(src.get(question_key))
            if q_case_norm and q_code_norm and (q_case_norm != q_code_norm):
                question_mismatch += 1
                msg = (
                    f"[WARN] question mismatch\n"
                    f"  case: {q_case_norm[:200]}\n"
                    f"  code: {q_code_norm[:200]}"
                )
                if strict:
                    raise ValueError(msg)
                if verbose:
                    print(msg)

        # -------- replace code field --------
        if code_field not in src:
            if strict:
                raise KeyError(f"Matched code item missing field '{code_field}'")
            if verbose:
                print(f"[WARN] Matched code item missing '{code_field}', skip")
            continue

        _truncate_merged_item(it, src, code_field=code_field, max_codes=max_codes, max_cases=max_cases)
        it[code_field] = src[code_field]
        replaced += 1

    _dump_json_items(out_json_path, case_items, wrap_data=wrap_data)

    report = {
        "code_items": len(code_items),
        "case_items": len(case_items),
        "replaced": replaced,
        "missing_in_code": missing_in_code,
        "question_mismatch": question_mismatch,
        "missing_id_in_code": missing_id_in_code,
        "dup_task_id_in_code": dup_ids,
        "fallback_by_question": fallback_by_question,
        "ambiguous_question": ambiguous_question,
        "out_json_path": out_json_path,
    }
    if verbose:
        print("[merge_replace_generated_code]", report)
    return report


# ---------------- Evaluation args ----------------
def _build_eval_args(**overrides) -> SimpleNamespace:
    ec = evaluation_config
    args = SimpleNamespace(
        k_case=getattr(ec, "k_case", 0),
        max_test=getattr(ec, "max_test", 64),
        num_chunks=getattr(ec, "num_chunks", 1),
        exe_verbose=getattr(ec, "exe_verbose", False),
        generation_mode=getattr(ec, "generation_mode", "original"),

        # metrics
        single_eval=False,
        eval_pass_at_k_only=False,
        eval_bon=True,
        pass_at_k_list=deepcopy(getattr(ec, "pass_at_k_list", [])),
        scale_tuple_list=deepcopy(getattr(ec, "scale_tuple_list", [])),

        is_final_eval=getattr(ec, "is_final_eval", False),
        mode=getattr(ec, "mode", "debug_score_ut"),

        use_random_ut_cluster=getattr(ec, "use_random_ut_cluster", True),
        random_ut_batch=getattr(ec, "random_ut_batch", 16),
        random_ut_min_top_count=getattr(ec, "random_ut_min_top_count", 2),
        random_ut_cluster_max_diff=getattr(ec, "random_ut_cluster_max_diff", 0),
        random_ut_placeholder=getattr(
            ec, "random_ut_placeholder", "We can not extract the input in the output. "
        ),
        skip_bon_log=False,
    )

    for k, v in overrides.items():
        setattr(args, k, v)

    # Ensure mutually exclusive evaluation modes.
    if getattr(args, "single_eval", False):
        args.eval_pass_at_k_only = False
        args.eval_bon = False
    elif getattr(args, "eval_pass_at_k_only", False):
        args.single_eval = False
        args.eval_bon = False
    else:
        args.single_eval = False
        args.eval_pass_at_k_only = False
        args.eval_bon = True

    return args


def _row_all_true(rows: List[Any], idx: int) -> bool:
    if idx < 0 or idx >= len(rows):
        return False
    row = rows[idx]
    if row is None:
        return False
    return all(bool(x) for x in row)


def _case_pass_counts(item: Dict[str, Any], args: SimpleNamespace) -> List[int]:
    table = item.get("case_bool_table")
    if not table:
        return []

    rows = table.tolist() if hasattr(table, "tolist") else table
    if getattr(args, "generation_mode", None) == "plansearch":
        case_is_valid = item.get("case_is_valid", [])
        valid_indices = [j for j, valid in enumerate(case_is_valid) if valid]
        rows = [[row[j] for j in valid_indices if j < len(row)] for row in rows]

    return [sum(1 for value in row if bool(value)) for row in rows if isinstance(row, list)]


def _selected_new_bon_index(item: Dict[str, Any], key: str, pass_counts: List[int]) -> int:
    info = item.get("new_bon_cluster_info") or {}
    selected = (info.get(key) or {}).get("selected_index")
    if selected is not None:
        return int(selected)
    if not pass_counts:
        return -1
    return int(pass_counts.index(max(pass_counts)))


def _cache_new_bon_from_items(data: List[Dict[str, Any]], args: SimpleNamespace) -> None:
    scores = {"new_bon": 0, "new_bon_front": 0, "new_bon_back": 0}
    total = 0

    for item in data:
        test_table = item.get("test_bool_table")
        if not test_table or not item.get("case_bool_table"):
            continue

        pass_counts = _case_pass_counts(item, args)
        if not pass_counts:
            continue

        test_rows_current = test_table.tolist() if hasattr(test_table, "tolist") else test_table
        test_rows = list(test_rows_current) + list(item.get("history_test_bool_rows_1") or [])

        for key in scores:
            selected_idx = _selected_new_bon_index(item, key, pass_counts)
            if _row_all_true(test_rows, selected_idx):
                scores[key] += 1
        total += 1

    if total > 0:
        args._new_bon_cached = {
            "new_bon": scores["new_bon"] / total,
            "new_bon_front": scores["new_bon_front"] / total,
            "new_bon_back": scores["new_bon_back"] / total,
        }
    elif hasattr(args, "_new_bon_cached"):
        delattr(args, "_new_bon_cached")


def run_exec_and_metrics(
    merged_json_path: str,
    *,
    outputs_name: str,
    out_with_exec_path: str,
    args: SimpleNamespace,
    compute_new_bon: bool,
) -> Dict[str, Any]:
    data = _load_json_items(merged_json_path)

    data = run_all_executions(
        data,
        args,
        skip_random_ut=not compute_new_bon,
        compute_new_bon=compute_new_bon,
    )

    old_use_random_ut_cluster = getattr(args, "use_random_ut_cluster", False)
    args.use_random_ut_cluster = False
    data = compute_and_log_metrics(data, outputs_name, mean_code=0, mean_case=0, args=args)
    args.use_random_ut_cluster = old_use_random_ut_cluster

    _dump_json_items(out_with_exec_path, data, wrap_data=False)
    rep = {
        "input": merged_json_path,
        "output": out_with_exec_path,
        "outputs_name": outputs_name,
        "num_tasks": len(data),
    }
    print("[run_exec_and_metrics]", rep)
    return rep


# ---------------- Folder scanning and chunk pairing ----------------
_CHUNK_RE = re.compile(r"(?:^|_|\b)chunk_(\d+)\.json$", re.IGNORECASE)

def _extract_chunk_id(filename: str) -> Optional[int]:
    m = _CHUNK_RE.search(filename)
    if not m:
        return None
    return int(m.group(1))


def _list_json_by_chunk(folder: str, contains: Optional[str] = None) -> Dict[int, str]:
    """
    Return: chunk_id -> filepath.
    contains: optionally keep only JSON files whose filename contains this substring.
    """
    out: Dict[int, str] = {}
    for fn in os.listdir(folder):
        if not fn.endswith(".json"):
            continue
        if contains and (contains not in fn):
            continue
        cid = _extract_chunk_id(fn)
        if cid is None:
            continue
        path = os.path.join(folder, fn)
        # If a chunk appears more than once, keep the lexicographically newer filename.
        if cid in out:
            if os.path.basename(path) > os.path.basename(out[cid]):
                out[cid] = path
        else:
            out[cid] = path
    return out


def batch_merge_and_eval_dir(
    *,
    code_dir: str,
    case_dir: str,
    out_dir: str,
    outputs_prefix: str,
    id_key: str = "task_id",
    question_key: str = "question",
    code_field: str = "generated_code",
    strict: bool = True,
    question_check: bool = True,
    code_contains: Optional[str] = None,
    case_contains: Optional[str] = None,
    mode: str = "debug_score_ut",
    result_subdir: Optional[str] = None,
    generation_mode: Optional[str] = None,
    max_test: Optional[int] = None,
    k_code: Optional[int] = None,
    k_case: Optional[int] = None,
    start_chunk: Optional[int] = None,
    end_chunk: Optional[int] = None,
    compute_new_bon: bool = False,
) -> Dict[str, Any]:
    # Archive each configuration into its own subdirectory to avoid mixed outputs.
    config_out_dir = os.path.join(out_dir, outputs_prefix)
    os.makedirs(config_out_dir, exist_ok=True)

    code_map = _list_json_by_chunk(code_dir, contains=code_contains)
    case_map = _list_json_by_chunk(case_dir, contains=case_contains)
    all_common = sorted(set(code_map.keys()) & set(case_map.keys()))

    if start_chunk is not None and end_chunk is not None and start_chunk > end_chunk:
        raise ValueError(f"Invalid chunk range: start_chunk={start_chunk} > end_chunk={end_chunk}")

    def _in_range(cid: int) -> bool:
        if start_chunk is not None and cid < start_chunk:
            return False
        if end_chunk is not None and cid > end_chunk:
            return False
        return True

    code_keys = {cid for cid in code_map.keys() if _in_range(cid)}
    case_keys = {cid for cid in case_map.keys() if _in_range(cid)}

    common = sorted(code_keys & case_keys)
    only_code = sorted(code_keys - case_keys)
    only_case = sorted(case_keys - code_keys)

    if not common:
        raise RuntimeError(
            f"No common chunks found.\n"
            f"code_dir={code_dir} has {len(code_keys)} chunks(in range), case_dir={case_dir} has {len(case_keys)} chunks(in range).\n"
            f"Try adjusting --code_contains/--case_contains or filenames."
        )

    print(
        f"[scan] chunk_range=[{start_chunk},{end_chunk}] "
        f"run_common_chunks={len(common)} all_common_chunks={len(all_common)} "
        f"only_code={only_code[:10]} only_case={only_case[:10]}"
    )
    if strict and (only_code or only_case):
        raise RuntimeError(
            f"Strict mode: chunk mismatch.\n"
            f"Only in code: {only_code}\n"
            f"Only in case: {only_case}"
        )

    metric_mode = os.path.join(result_subdir, mode) if result_subdir else mode

    # eval args
    overrides = {"mode": metric_mode, "eval_bon": True}
    if generation_mode is not None:
        overrides["generation_mode"] = generation_mode
    if max_test is not None:
        overrides["max_test"] = max_test
    if k_case is not None:
        overrides["k_case"] = k_case
    eval_args = _build_eval_args(**overrides)

    merge_reports = []
    exec_reports = []
    with_exec_files = []

    for cid in common:
        code_path = code_map[cid]
        case_path = case_map[cid]

        merged_path = os.path.join(config_out_dir, f"{outputs_prefix}_chunk_{cid}_merged.json")
        with_exec_path = os.path.join(config_out_dir, f"{outputs_prefix}_chunk_{cid}_merged_with_exec.json")

        rep_m = merge_replace_generated_code(
            code_path,
            case_path,
            merged_path,
            id_key=id_key,
            question_key=question_key,
            code_field=code_field,
            max_codes=k_code,
            max_cases=k_case,
            strict=strict,
            question_check=question_check,
            verbose=True,
        )
        merge_reports.append(rep_m)

        rep_e = run_exec_and_metrics(
            merged_json_path=merged_path,
            outputs_name=f"{outputs_prefix}_chunk_{cid}",
            out_with_exec_path=with_exec_path,
            args=eval_args,
            compute_new_bon=compute_new_bon,
        )
        exec_reports.append(rep_e)
        with_exec_files.append(with_exec_path)

    # Aggregate all chunk-level with_exec files and recompute the full result.
    # Even when this run only covers part of the chunks, ALL requires every chunk.
    all_with_exec_files: List[str] = []
    missing_all_chunks: List[int] = []
    for cid in all_common:
        p = os.path.join(config_out_dir, f"{outputs_prefix}_chunk_{cid}_merged_with_exec.json")
        if os.path.isfile(p):
            all_with_exec_files.append(p)
        else:
            missing_all_chunks.append(cid)

    if missing_all_chunks:
        raise RuntimeError(
            "Cannot compute ALL over all chunks because some chunk outputs are missing.\n"
            f"missing_chunks={missing_all_chunks}\n"
            "Please run the missing chunks first (can use --start_chunk/--end_chunk to resume)."
        )

    all_data: List[Dict[str, Any]] = []
    for p in all_with_exec_files:
        all_data.extend(_load_json_items(p))

    final_outputs_name = f"{outputs_prefix}_ALL"
    old_use_random_ut_cluster = getattr(eval_args, "use_random_ut_cluster", False)
    if compute_new_bon:
        _cache_new_bon_from_items(all_data, eval_args)
    elif hasattr(eval_args, "_new_bon_cached"):
        delattr(eval_args, "_new_bon_cached")
    eval_args.use_random_ut_cluster = False
    all_data = compute_and_log_metrics(all_data, final_outputs_name, mean_code=0, mean_case=0, args=eval_args)
    eval_args.use_random_ut_cluster = old_use_random_ut_cluster

    final_path = os.path.join(config_out_dir, f"{outputs_prefix}_ALL_merged_with_exec.json")
    _dump_json_items(final_path, all_data, wrap_data=False)

    summary = {
        "code_dir": code_dir,
        "case_dir": case_dir,
        "out_dir": out_dir,
        "config_out_dir": config_out_dir,
        "outputs_prefix": outputs_prefix,
        "num_run_common_chunks": len(common),
        "num_all_common_chunks": len(all_common),
        "num_total_tasks": len(all_data),
        "final_output_path": final_path,
        "only_code_chunks": only_code,
        "only_case_chunks": only_case,
        "missing_all_chunks": missing_all_chunks,
        "start_chunk": start_chunk,
        "end_chunk": end_chunk,
    }
    print("[DONE summary]", summary)
    return {"summary": summary, "merge_reports": merge_reports, "exec_reports": exec_reports}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_dir", type=str, required=True, help="folder of code jsons (has generated_code), split by chunk")
    ap.add_argument("--case_dir", type=str, required=True, help="folder of case jsons (has case_input/case_output), split by chunk")
    ap.add_argument("--out_dir", type=str, required=True, help="output folder")
    ap.add_argument("--outputs_prefix", type=str, default="results_merged", help="prefix for output names/files")

    ap.add_argument("--strict", action="store_true", help="strict matching for chunks and task_id")
    ap.add_argument("--no_strict", dest="strict", action="store_false")
    ap.set_defaults(strict=True)

    ap.add_argument("--id_key", type=str, default="task_id")
    ap.add_argument("--question_key", type=str, default="question")
    ap.add_argument("--code_field", type=str, default="generated_code")

    ap.add_argument("--question_check", action="store_true")
    ap.add_argument("--no_question_check", dest="question_check", action="store_false")
    ap.set_defaults(question_check=True)

    ap.add_argument("--code_contains", type=str, default=None, help="optional: only use code files whose name contains this substring")
    ap.add_argument("--case_contains", type=str, default=None, help="optional: only use case files whose name contains this substring")

    ap.add_argument("--mode", type=str, default="debug_score_ut")
    ap.add_argument("--result_subdir", type=str, default=None, help="optional: insert an extra subdirectory before mode for txt metric outputs")
    ap.add_argument("--generation_mode", type=str, default="original_resample", help="optional: override evaluation_config.generation_mode for this run")
    ap.add_argument("--max_test", type=int, default=None)
    ap.add_argument("--k_code", type=int, default=None)
    ap.add_argument("--k_case", type=int, default=None)
    ap.add_argument("--start_chunk", type=int, default=None, help="optional: start chunk id (inclusive)")
    ap.add_argument("--end_chunk", type=int, default=None, help="optional: end chunk id (inclusive)")
    ap.add_argument("--compute_new_bon", type=str2bool, default=False, help="run New_BoN execution during each chunk")

    args = ap.parse_args()

    batch_merge_and_eval_dir(
        code_dir=args.code_dir,
        case_dir=args.case_dir,
        out_dir=args.out_dir,
        outputs_prefix=args.outputs_prefix,
        id_key=args.id_key,
        question_key=args.question_key,
        code_field=args.code_field,
        strict=args.strict,
        question_check=args.question_check,
        code_contains=args.code_contains,
        case_contains=args.case_contains,
        mode=args.mode,
        result_subdir=args.result_subdir,
        generation_mode=args.generation_mode,
        max_test=args.max_test,
        k_code=args.k_code,
        k_case=args.k_case,
        start_chunk=args.start_chunk,
        end_chunk=args.end_chunk,
        compute_new_bon=args.compute_new_bon,
    )


if __name__ == "__main__":
    main()
