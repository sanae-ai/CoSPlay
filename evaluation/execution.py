# Sandboxed execution helpers for generated programs and unit tests.
import os
import io
import sys
import time
import typing
import multiprocessing as mp
import tempfile
from collections import Counter
from contextlib import contextmanager

import numpy as np
from termcolor import cprint


SANDBOX_ROOT = os.path.join(os.path.dirname(__file__), "temp_exec")
os.makedirs(SANDBOX_ROOT, exist_ok=True)


def _limit_case_fields(data_item, max_cases):
    if not max_cases or max_cases <= 0:
        return

    def _consistency_score(samples):
        if not samples or not isinstance(samples, list):
            return 0, 0.0, 0
        flat = [s for s in samples if s is not None]
        if not flat:
            return 0, 0.0, len(samples)
        counts = Counter(flat)
        top = counts.most_common(1)[0][1]
        total = len(flat)
        return top, top / total if total > 0 else 0.0, total

    def _pad_case_fields(target_len: int):
        pad_map = {
            "case_input": "",
            "case_output": "",
            "case_is_valid": False,
            "case_text": "",
            "attack_ideas": "Unknown",
            "full_case_generation": "",
            "case_input_original": "",
            "case_output_original": "",
            "case_output_samples": [],
            "case_output_samples_extracted": [],
            "case_code_voting_results": None,
        }
        for field, fill in pad_map.items():
            lst = data_item.get(field)
            if not isinstance(lst, list):
                continue
            while len(lst) < target_len:

                lst.append(list(fill) if isinstance(fill, list) else fill)


    case_input = data_item.get("case_input", [])
    case_output = data_item.get("case_output", [])
    base_len = len(case_input)
    if base_len > 0:
        candidates = []
        case_is_valid = data_item.get("case_is_valid", [])
        samples_extracted = data_item.get("case_output_samples_extracted", [])
        for i in range(base_len):
            if i >= len(case_input) or i >= len(case_output):
                continue
            if case_input[i] is None or case_output[i] is None:
                continue
            top_count, ratio, total_len = _consistency_score(
                samples_extracted[i] if i < len(samples_extracted) else []
            )
            is_valid = case_is_valid[i] if i < len(case_is_valid) else True
            candidates.append((i, is_valid, top_count, ratio, total_len))

        if not candidates:

            _pad_case_fields(max_cases)
            return


        candidates.sort(key=lambda t: (bool(t[1]), t[2], t[3], t[4], -t[0]), reverse=True)
        selected_indices = [t[0] for t in candidates[:max_cases]]

        def reorder(field):
            lst = data_item.get(field)
            if isinstance(lst, list) and len(lst) > 0:
                data_item[field] = [lst[i] for i in selected_indices if i < len(lst)]

        for field in [
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
        ]:
            reorder(field)
        _pad_case_fields(max_cases)
    else:

        _pad_case_fields(max_cases)


@contextmanager
def sandbox_env(stdin_payload: str):
    prev_cwd = os.getcwd()
    payload = stdin_payload if isinstance(stdin_payload, str) else str(stdin_payload)
    with tempfile.TemporaryDirectory(prefix="eval_exec_", dir=SANDBOX_ROOT) as tmp_dir:
        os.chdir(tmp_dir)
        for fname in ("input.txt", "stdin.txt"):
            with open(fname, "w", encoding="utf-8") as fh:
                fh.write(payload)
        try:
            yield tmp_dir
        finally:
            os.chdir(prev_cwd)

def worker(script, input_val, output_queue):
    stdin_payload = input_val if isinstance(input_val, str) else str(input_val)
    input_lines = iter(stdin_payload.splitlines())

    def fake_input(prompt=""):
        try:
            return next(input_lines)
        except StopIteration:
            raise EOFError("No more input")


    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)


    tfile = tempfile.TemporaryFile(mode='w+b')

    try:

        sys.stdout.flush()
        sys.stderr.flush()


        os.dup2(tfile.fileno(), 1)
        os.dup2(tfile.fileno(), 2)

        context = {
            "__name__": "__main__",
            "input": fake_input,
            "List": typing.List,
            "Tuple": typing.Tuple,
            "Optional": typing.Optional,

        }

        try:
            with sandbox_env(stdin_payload):
                exec(script, context)
        except SystemExit:
            pass
        except Exception:


            import traceback
            traceback.print_exc()

    except Exception as e:


        try:
            tfile.write(f"\nSystem Error: {e}\n".encode('utf-8'))
        except:
            pass
    
    finally:


        sys.stdout.flush()
        sys.stderr.flush()


        os.dup2(original_stdout_fd, 1)
        os.dup2(original_stderr_fd, 2)
        

        os.close(original_stdout_fd)
        os.close(original_stderr_fd)


        tfile.flush()
        tfile.seek(0)
        try:

            content = tfile.read().decode('utf-8', errors='replace')
            output_queue.put(content)
        except Exception as read_e:
            output_queue.put(f"Error reading output: {read_e}")
        
        tfile.close()


def run_scripts_with_timeout(scripts, inputs, time_limits):
    results = [None] * len(scripts)
    processes = []
    queues = []
    deadlines = []

    for i in range(len(scripts)):
        q = mp.Queue()
        p = mp.Process(target=worker, args=(scripts[i], inputs[i], q))
        processes.append(p)
        queues.append(q)
        p.start()
        deadlines.append(time.time() + time_limits[i])

    while any(p.is_alive() for p in processes):
        now = time.time()
        for i, p in enumerate(processes):
            if p.is_alive() and now >= deadlines[i]:
                p.terminate()
                results[i] = "Timeout Error"
        time.sleep(0.001)

    for i, p in enumerate(processes):
        if results[i] is None:
            try:
                results[i] = queues[i].get_nowait()
            except Exception as e:
                results[i] = f"Execution Error: {e}"

    return results


def test_if_eq(x, y):
    return " ".join(x.split()) == " ".join(y.split())


def get_chunk_indices(n, num_chunks):
    chunk_size = max(1, n // num_chunks)
    remainder = n % num_chunks
    indices = []
    start = 0
    for i in range(num_chunks):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        if start >= n:
            break
        indices.append((start, min(end, n)))
        start = end
    return indices


def run_scripts_with_chunk(code_list, test_input_list, time_limit_list, num_chunks, exe_verbose):
    chunks = get_chunk_indices(len(code_list), num_chunks)
    exe_results = []
    for idx, (start, end) in enumerate(chunks):
        if exe_verbose:
            print(f"process {idx}/{len(chunks)}")
        sub_code_list = code_list[start:end]
        sub_test_input_list = test_input_list[start:end]
        sub_time_limit_list = time_limit_list[start:end]
        sub_exe_results = run_scripts_with_timeout(
            sub_code_list, sub_test_input_list, sub_time_limit_list
        )
        exe_results.extend(sub_exe_results)
    return exe_results


# Full execution path used for final evaluation.
def run_all_executions(data, args, *, skip_random_ut=False, compute_new_bon=True):
    print(f"✓ Loaded generation results: {len(data)} records", flush=True)

    case_index_list = []
    case_position_list = []
    case_code_list = []
    case_input_list = []
    case_output_list = []
    case_time_limit_list = []

    test_index_list = []
    test_position_list = []
    test_code_list = []
    test_input_list = []
    test_output_list = []
    test_time_limit_list = []

    max_cases = getattr(args, "k_case", None)

    for i in range(len(data)):
        data[i].setdefault("test_input", [])
        data[i].setdefault("test_output", [])
        data[i].setdefault("test_time_limit", 1)
        _limit_case_fields(data[i], max_cases)

        if len(data[i]["case_input"]) * len(data[i]["generated_code"]) == 0:
            data[i]["case_exe_results"] = None
            data[i]["case_bool_table"] = None
        else:
            n_row = len(data[i]["generated_code"])
            n_col = len(data[i]["case_input"])
            data[i]["case_exe_results"] = [["" for _ in range(n_col)] for _ in range(n_row)]
            data[i]["case_bool_table"] = np.full((n_row, n_col), False, dtype=bool)


        if len(data[i]["test_input"]) * len(data[i]["generated_code"]) == 0:
            data[i]["test_exe_results"] = None
            data[i]["test_bool_table"] = None
        else:
            n_row = len(data[i]["generated_code"])
            n_col = min(len(data[i]["test_input"]), args.max_test)
            data[i]["test_exe_results"] = [["" for _ in range(n_col)] for _ in range(n_row)]
            data[i]["test_bool_table"] = np.full((n_row, n_col), False, dtype=bool)

        data_i = data[i].copy()


        for j in range(len(data_i["generated_code"])):
            for k in range(len(data_i["case_input"])):
                code = data_i["generated_code"][j]
                case_input = data_i["case_input"][k]
                case_output = data_i["case_output"][k]
                


                if args.generation_mode == "exp-atk":
                    if case_input is None or case_output is None:
                        continue
                    
                case_code_list.append(code)
                case_input_list.append(case_input)
                case_output_list.append(case_output)
                if "test_time_limit" in data_i.keys():
                    case_time_limit_list.append(data_i["test_time_limit"])

                else:
                    case_time_limit_list.append(1)
                    print("No time limit provided!")
                case_index_list.append(i)
                case_position_list.append((j, k))


        max_k = min(len(data_i["test_input"]), args.max_test)
        for j in range(len(data_i["generated_code"])):
            for k in range(max_k):
                code = data_i["generated_code"][j]
                test_input = data_i["test_input"][k]
                test_output = data_i["test_output"][k]
                test_code_list.append(code)
                test_input_list.append(test_input)
                test_output_list.append(test_output)
                if "test_time_limit" in data_i.keys():
                    test_time_limit_list.append(data_i["test_time_limit"])

                else:
                    test_time_limit_list.append(1)
                    print("No time limit provided!")
                test_index_list.append(i)
                test_position_list.append((j, k))

    print(f"Preparing to execute generated tests: {len(case_code_list)} jobs", flush=True)


    if not args.single_eval:
        cprint("start execution for generated unit tests", "green")
        case_exe_results = run_scripts_with_chunk(
            case_code_list,
            case_input_list,
            case_time_limit_list,
            args.num_chunks,
            args.exe_verbose,
        )
        cprint("execution job done!", "green")
    else:
        case_exe_results = []


    cprint("start execution for ground-truth unit tests", "green")
    test_exe_results = run_scripts_with_chunk(
        test_code_list,
        test_input_list,
        test_time_limit_list,
        args.num_chunks,
        args.exe_verbose,
    )
    cprint("execution job done!", "green")


    for i in range(len(case_index_list)):
        index_i = case_index_list[i]
        j, k = case_position_list[i]
        if data[index_i]["case_exe_results"] is not None:
            data[index_i]["case_exe_results"][j][k] = case_exe_results[i]
            data[index_i]["case_bool_table"][j][k] = test_if_eq(
                case_exe_results[i], case_output_list[i]
            )


    for i in range(len(test_index_list)):
        index_i = test_index_list[i]
        j, k = test_position_list[i]
        data[index_i]["test_exe_results"][j][k] = test_exe_results[i]
        data[index_i]["test_bool_table"][j][k] = test_if_eq(
            test_exe_results[i], test_output_list[i]
        )


    if (not skip_random_ut) and compute_new_bon:
        _compute_new_bon_with_history(data, args, print_results=True)
    return data


# Random-UT BoN with current and historical code pools.
def _compute_new_bon_with_history(data, args, *, print_results=True):

    try:
        from evaluation import metrics as metrics_mod
    except Exception:
        import metrics as metrics_mod


    use_random = bool(getattr(args, "use_random_ut_cluster", True))
    if not use_random:
        return None
    random_ut_batch = int(getattr(args, "random_ut_batch", 16))
    random_ut_select_count = random_ut_batch
    random_ut_min_top_count = int(getattr(args, "random_ut_min_top_count", 2))
    random_ut_cluster_max_diff = int(getattr(args, "random_ut_cluster_max_diff", 0))
    random_ut_placeholder = getattr(
        args, "random_ut_placeholder", "We can not extract the input in the output. "
    )


    if not hasattr(args, "_random_ut_history_front") or args._random_ut_history_front is None:
        args._random_ut_history_front = [dict() for _ in range(len(data))]
    if not hasattr(args, "_random_ut_history_back") or args._random_ut_history_back is None:
        args._random_ut_history_back = [dict() for _ in range(len(data))]

    if len(args._random_ut_history_front) != len(data):
        args._random_ut_history_front = [dict() for _ in range(len(data))]
    if len(args._random_ut_history_back) != len(data):
        args._random_ut_history_back = [dict() for _ in range(len(data))]


    for i in range(len(data)):
        raw_inputs = data[i].get("random_case_input", []) or []
        data[i]["random_case_input_selected"] = metrics_mod._select_random_inputs(
            raw_inputs, random_ut_select_count, random_ut_placeholder
        )


    combined_codes_by_task = []
    history_codes_by_task = []
    exec_matrix_by_task = []

    code_list = []
    input_list = []
    time_limit_list = []
    position_list = []

    for i in range(len(data)):
        current_codes = data[i].get("generated_code", []) or []
        hist_front_codes = list(args._random_ut_history_front[i].keys())
        hist_back_codes = list(args._random_ut_history_back[i].keys())
        history_codes = []
        for code in hist_front_codes + hist_back_codes:
            if code:
                history_codes.append(code)
        combined_codes = list(current_codes) + history_codes

        random_inputs = data[i].get("random_case_input_selected", []) or []
        time_limit = data[i].get("test_time_limit", 1)

        combined_codes_by_task.append(combined_codes)
        history_codes_by_task.append(history_codes)
        exec_matrix_by_task.append([["" for _ in range(len(random_inputs))] for _ in range(len(combined_codes))])

        data[i]["random_ut_history_codes_front"] = list(args._random_ut_history_front[i].keys())
        data[i]["random_ut_history_codes_back"] = list(args._random_ut_history_back[i].keys())
        data[i]["random_ut_history_codes_used"] = list(history_codes)


        for code_idx, code in enumerate(combined_codes):
            for ut_idx, ut_input in enumerate(random_inputs):
                code_list.append(code)
                input_list.append(ut_input)
                time_limit_list.append(time_limit)
                position_list.append((i, code_idx, ut_idx))

    if code_list:
        outputs = run_scripts_with_chunk(
            code_list,
            input_list,
            time_limit_list,
            args.num_chunks,
            args.exe_verbose,
        )
        for out, (task_i, code_idx, ut_idx) in zip(outputs, position_list):
            exec_matrix_by_task[task_i][code_idx][ut_idx] = out


    for i in range(len(data)):
        current_len = len(data[i].get("generated_code", []) or [])
        all_rows = exec_matrix_by_task[i]

        data[i]["random_case_exe_results_1"] = all_rows[:current_len]
        data[i]["random_case_bool_table_rows_1"] = [
            "".join("0" if metrics_mod._is_error_output(x) else "1" for x in row)
            for row in all_rows[:current_len]
        ]

        data[i]["random_case_exe_results_history_1"] = all_rows[current_len:]
        data[i]["random_case_bool_table_rows_history_1"] = [
            "".join("0" if metrics_mod._is_error_output(x) else "1" for x in row)
            for row in all_rows[current_len:]
        ]


    history_test_bool_rows_by_task = [[] for _ in range(len(data))]
    hist_code_list = []
    hist_input_list = []
    hist_time_limit_list = []
    hist_position_list = []
    for i in range(len(data)):
        history_codes = history_codes_by_task[i]
        if not history_codes:
            continue
        test_inputs = data[i].get("test_input", []) or []
        time_limit = data[i].get("test_time_limit", 1)
        for code_idx, code in enumerate(history_codes):
            cached = args._random_ut_history_back[i].get(code) or args._random_ut_history_front[i].get(code)
            cached_row = cached.get("test_row") if cached else None
            if cached_row is not None:
                history_test_bool_rows_by_task[i].append(cached_row)
                continue
            history_test_bool_rows_by_task[i].append(None)
            for ut_idx, ut_input in enumerate(test_inputs):
                hist_code_list.append(code)
                hist_input_list.append(ut_input)
                hist_time_limit_list.append(time_limit)
                hist_position_list.append((i, code_idx, ut_idx))

    if hist_code_list:
        hist_outputs = run_scripts_with_chunk(
            hist_code_list,
            hist_input_list,
            hist_time_limit_list,
            args.num_chunks,
            args.exe_verbose,
        )

        hist_rows_raw = {}
        for out, (task_i, code_idx, ut_idx) in zip(hist_outputs, hist_position_list):
            hist_rows_raw.setdefault((task_i, code_idx), []).append(out)
        for (task_i, code_idx), row in hist_rows_raw.items():
            test_outputs = data[task_i].get("test_output", []) or []
            bool_row = []
            for out, gold in zip(row, test_outputs):
                if out is None or gold is None:
                    bool_row.append(False)
                else:
                    bool_row.append(test_if_eq(out, gold))
            history_test_bool_rows_by_task[task_i][code_idx] = bool_row

    for i in range(len(data)):
        data[i]["history_test_bool_rows_1"] = history_test_bool_rows_by_task[i]


    def _dedupe_indices(indices):
        seen = set()
        out = []
        for idx in indices:
            if idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
        return out

    def _candidate_stats(outputs, cluster_indices, selected_local, pass_counts, error_masks, candidate_indices):

        match_sum = 0
        for j in cluster_indices:
            if j == selected_local:
                continue
            match_sum += metrics_mod._non_error_match_count(
                outputs[selected_local],
                outputs[j],
                None if error_masks is None else error_masks[selected_local],
                None if error_masks is None else error_masks[j],
            )
        err_count = sum(error_masks[selected_local]) if error_masks is not None else 0
        global_idx = candidate_indices[selected_local]
        pass_cnt = pass_counts[global_idx] if global_idx < len(pass_counts) else 0
        return match_sum, err_count, pass_cnt

    def _pick_with_priority(outputs, candidate_indices, pass_counts, prefer_set):
        if not outputs:
            return None, [], None
        error_masks = metrics_mod._build_error_masks(outputs)
        clusters = metrics_mod._cluster_outputs(outputs, random_ut_cluster_max_diff, error_masks=error_masks)
        if not clusters:
            return None, [], None

        def cluster_match_score(cluster):
            score = 0
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    ii = cluster[i]
                    jj = cluster[j]
                    score += metrics_mod._non_error_match_count(
                        outputs[ii],
                        outputs[jj],
                        None if error_masks is None else error_masks[ii],
                        None if error_masks is None else error_masks[jj],
                    )
            return score

        def cluster_key(cluster):
            match_score = cluster_match_score(cluster)
            pass_sum = sum(pass_counts[candidate_indices[i]] for i in cluster)
            min_idx = min(candidate_indices[i] for i in cluster)
            return (match_score, pass_sum, -min_idx)

        best_cluster = max(clusters, key=cluster_key)

        best_center = None
        best_key = None
        for i in best_cluster:
            idx = candidate_indices[i]
            err_count = sum(error_masks[i]) if error_masks is not None else 0
            key = (0 if idx in prefer_set else 1, err_count, -pass_counts[idx], idx)
            if best_key is None or key < best_key:
                best_key = key
                best_center = i
        if best_center is None:
            return None, best_cluster, None
        return candidate_indices[best_center], best_cluster, error_masks


    new_score = 0
    new_front_score = 0
    new_back_score = 0
    total = 0

    for i in range(len(data)):
        if data[i].get("test_bool_table") is None:
            continue
        table = data[i].get("case_bool_table")
        if table is None:
            continue
        case_arr = np.array(table, dtype=bool)
        if case_arr.ndim != 2 or case_arr.size == 0:
            continue
        if getattr(args, "generation_mode", None) == "exp-atk":
            case_is_valid = data[i].get("case_is_valid", [True] * case_arr.shape[1])
            valid_indices = [j for j, v in enumerate(case_is_valid) if v]
            case_arr = case_arr[:, valid_indices].copy() if valid_indices else case_arr[:, :0]

        current_codes = data[i].get("generated_code", []) or []
        combined_codes = combined_codes_by_task[i]
        exe_matrix = exec_matrix_by_task[i]

        pass_counts_current = case_arr.sum(axis=1).astype(int).tolist() if case_arr.shape[0] > 0 else []
        pass_counts_extra = []
        for code in combined_codes[len(current_codes):]:
            info = args._random_ut_history_back[i].get(code) or args._random_ut_history_front[i].get(code)
            pass_counts_extra.append(int(info.get("pass_count", 0)) if info else 0)
        pass_counts = pass_counts_current + pass_counts_extra

        test_table = np.array(data[i].get("test_bool_table"), dtype=bool)
        test_rows_current = [test_table[j].tolist() for j in range(min(len(current_codes), test_table.shape[0]))]
        test_rows_extra = data[i].get("history_test_bool_rows_1", [])
        test_rows = test_rows_current + test_rows_extra

        if not pass_counts_current:
            continue
        max_pass = max(pass_counts_current)
        top_candidates = [idx for idx, cnt in enumerate(pass_counts_current) if cnt == max_pass]
        best_code_index = int(pass_counts_current.index(max_pass)) if pass_counts_current else 0
        new_best_index = best_code_index
        new_front_index = best_code_index
        new_back_index = best_code_index


        new_info = {
            "source": "top_pass",
            "top_candidates": top_candidates,
            "history_indices": [],
        }
        if exe_matrix and len(top_candidates) >= random_ut_min_top_count:
            active_outputs = []
            active_indices = []
            for code_idx in top_candidates:
                if code_idx < len(exe_matrix):
                    row = exe_matrix[code_idx]
                    active_outputs.append([metrics_mod._normalize_output(x) for x in row])
                    active_indices.append(code_idx)
            if active_outputs:
                clusters = metrics_mod._cluster_outputs(
                    active_outputs,
                    random_ut_cluster_max_diff,
                    error_masks=metrics_mod._build_error_masks(active_outputs),
                )
                best_cluster, center_idx, _ = metrics_mod._pick_cluster_and_center(
                    clusters,
                    active_outputs,
                    active_indices,
                    pass_counts,
                    error_masks=metrics_mod._build_error_masks(active_outputs),
                )
                if best_cluster:
                    new_best_index = active_indices[center_idx]
                    error_masks = metrics_mod._build_error_masks(active_outputs)
                    match_sum, err_count, pass_cnt = _candidate_stats(
                        active_outputs, best_cluster, center_idx, pass_counts, error_masks, active_indices
                    )
                    new_info.update({
                        "selected_index": new_best_index,
                        "cluster_size": len(best_cluster),
                        "match_sum": match_sum,
                        "err_count": err_count,
                        "pass_count": pass_cnt,
                    })

        hist_front_indices = [idx for idx, code in enumerate(combined_codes) if code in args._random_ut_history_front[i]]
        hist_back_indices = [idx for idx, code in enumerate(combined_codes) if code in args._random_ut_history_back[i]]


        front_info = {
            "source": "history_first",
            "top_candidates": top_candidates,
            "history_indices": hist_front_indices,
        }
        if exe_matrix:
            union_front = _dedupe_indices(hist_front_indices + top_candidates)
            outputs_front = []
            indices_front = []
            for code_idx in union_front:
                if code_idx < len(exe_matrix):
                    row = exe_matrix[code_idx]
                    outputs_front.append([metrics_mod._normalize_output(x) for x in row])
                    indices_front.append(code_idx)
            if outputs_front:
                picked, best_cluster, error_masks = _pick_with_priority(
                    outputs_front,
                    indices_front,
                    pass_counts,
                    prefer_set=set(hist_front_indices),
                )
                if picked is not None:
                    new_front_index = picked
                    local_idx = indices_front.index(picked)
                    match_sum, err_count, pass_cnt = _candidate_stats(
                        outputs_front, best_cluster, local_idx, pass_counts, error_masks, indices_front
                    )
                    front_info.update({
                        "selected_index": new_front_index,
                        "cluster_size": len(best_cluster),
                        "match_sum": match_sum,
                        "err_count": err_count,
                        "pass_count": pass_cnt,
                    })


        back_info = {
            "source": "current_first",
            "top_candidates": top_candidates,
            "history_indices": hist_back_indices,
        }
        if exe_matrix:
            union_back = _dedupe_indices(top_candidates + hist_back_indices)
            outputs_back = []
            indices_back = []
            for code_idx in union_back:
                if code_idx < len(exe_matrix):
                    row = exe_matrix[code_idx]
                    outputs_back.append([metrics_mod._normalize_output(x) for x in row])
                    indices_back.append(code_idx)
            if outputs_back:
                picked, best_cluster, error_masks = _pick_with_priority(
                    outputs_back,
                    indices_back,
                    pass_counts,
                    prefer_set=set(top_candidates),
                )
                if picked is not None:
                    new_back_index = picked
                    local_idx = indices_back.index(picked)
                    match_sum, err_count, pass_cnt = _candidate_stats(
                        outputs_back, best_cluster, local_idx, pass_counts, error_masks, indices_back
                    )
                    back_info.update({
                        "selected_index": new_back_index,
                        "cluster_size": len(best_cluster),
                        "match_sum": match_sum,
                        "err_count": err_count,
                        "pass_count": pass_cnt,
                    })

        data[i]["new_bon_cluster_info"] = {
            "new_bon": new_info,
            "new_bon_front": front_info,
            "new_bon_back": back_info,
        }

        def _row_all_true(rows, idx):
            if idx < 0 or idx >= len(rows):
                return False
            row = rows[idx]
            if row is None:
                return False
            return all(row)

        if _row_all_true(test_rows, new_best_index):
            new_score += 1
        if _row_all_true(test_rows, new_front_index):
            new_front_score += 1
        if _row_all_true(test_rows, new_back_index):
            new_back_score += 1
        total += 1


        if new_front_index < len(combined_codes):
            code = combined_codes[new_front_index]
            if code:
                args._random_ut_history_front[i][code] = {
                    "test_row": test_rows[new_front_index],
                    "pass_count": pass_counts[new_front_index] if new_front_index < len(pass_counts) else 0,
                }
        if new_back_index < len(combined_codes):
            code = combined_codes[new_back_index]
            if code:
                args._random_ut_history_back[i][code] = {
                    "test_row": test_rows[new_back_index],
                    "pass_count": pass_counts[new_back_index] if new_back_index < len(pass_counts) else 0,
                }

    if total > 0:
        new_bon = new_score / total
        new_bon_front = new_front_score / total
        new_bon_back = new_back_score / total
        if print_results:
            print(
                f"New_BoN (current): {new_bon:.4f} | New_BoN_front: {new_bon_front:.4f} | New_BoN_back: {new_bon_back:.4f}",
                flush=True,
            )

        args._new_bon_cached = {
            "new_bon": new_bon,
            "new_bon_front": new_bon_front,
            "new_bon_back": new_bon_back,
        }
        return {
            "new_bon": new_bon,
            "new_bon_front": new_bon_front,
            "new_bon_back": new_bon_back,
            "total": total,
        }
    return None

# Case-only execution path used inside generation and self-play rounds.
def run_all_executions_generate(data, args):
    print(f"✓ Loaded generation results: {len(data)} records", flush=True)

    case_index_list = []
    case_position_list = []
    case_code_list = []
    case_input_list = []
    case_output_list = []
    case_time_limit_list = []

    test_index_list = []
    test_position_list = []
    test_code_list = []
    test_input_list = []
    test_output_list = []
    test_time_limit_list = []

    max_cases = getattr(args, "k_case", None)

    for i in range(len(data)):
        data[i].setdefault("test_input", [])
        data[i].setdefault("test_output", [])
        data[i].setdefault("test_time_limit", 1)
        _limit_case_fields(data[i], max_cases)

        if len(data[i]["case_input"]) * len(data[i]["generated_code"]) == 0:
            data[i]["case_exe_results"] = None
            data[i]["case_bool_table"] = None
        else:
            n_row = len(data[i]["generated_code"])
            n_col = len(data[i]["case_input"])
            data[i]["case_exe_results"] = [["" for _ in range(n_col)] for _ in range(n_row)]
            data[i]["case_bool_table"] = np.full((n_row, n_col), False, dtype=bool)


        if len(data[i]["test_input"]) * len(data[i]["generated_code"]) == 0:
            data[i]["test_exe_results"] = None
            data[i]["test_bool_table"] = None
        else:
            n_row = len(data[i]["generated_code"])
            n_col = min(len(data[i]["test_input"]), args.max_test)
            data[i]["test_exe_results"] = [["" for _ in range(n_col)] for _ in range(n_row)]
            data[i]["test_bool_table"] = np.full((n_row, n_col), False, dtype=bool)

        data_i = data[i].copy()


        for j in range(len(data_i["generated_code"])):
            for k in range(len(data_i["case_input"])):
                code = data_i["generated_code"][j]
                case_input = data_i["case_input"][k]
                case_output = data_i["case_output"][k]
                


                if args.generation_mode == "exp-atk":
                    if case_input is None or case_output is None:
                        continue
                    
                case_code_list.append(code)
                case_input_list.append(case_input)
                case_output_list.append(case_output)
                if "test_time_limit" in data_i.keys():
                    case_time_limit_list.append(data_i["test_time_limit"])

                else:
                    case_time_limit_list.append(1)
                    print("No time limit provided!")
                case_index_list.append(i)
                case_position_list.append((j, k))


    print(f"Preparing to execute generated tests: {len(case_code_list)} jobs", flush=True)


    if not args.single_eval:
        cprint("start execution for generated unit tests", "green")
        case_exe_results = run_scripts_with_chunk(
            case_code_list,
            case_input_list,
            case_time_limit_list,
            args.num_chunks,
            args.exe_verbose,
        )
        cprint("execution job done!", "green")
    else:
        case_exe_results = []


    for i in range(len(case_index_list)):
        index_i = case_index_list[i]
        j, k = case_position_list[i]
        if data[index_i]["case_exe_results"] is not None:
            data[index_i]["case_exe_results"][j][k] = case_exe_results[i]
            data[index_i]["case_bool_table"][j][k] = test_if_eq(
                case_exe_results[i], case_output_list[i]
            )


    return data
