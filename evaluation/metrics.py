
# Metric computation for one-shot, pass@k, and BoN evaluation modes.
import os
import math
import json

import numpy as np
from termcolor import cprint

try:
    from evaluation import execution
except Exception:
    import execution


def compute_pass_at_k_for_task(all_test_table_i, pass_at_k_values):
    results = {}
    n_samples = all_test_table_i.shape[0]
    if n_samples == 0:
        return {k: 0.0 for k in pass_at_k_values}

    is_correct_codes = all_test_table_i.all(axis=1)
    c = int(is_correct_codes.sum())

    for k in pass_at_k_values:
        k_eff = min(k, n_samples)
        if c == 0 or k_eff == 0:
            pass_k_i = 0.0
        else:
            if n_samples - c >= k_eff:
                prob_all_wrong = math.comb(n_samples - c, k_eff) / math.comb(
                    n_samples, k_eff
                )
            else:
                prob_all_wrong = 0.0
            pass_k_i = 1.0 - prob_all_wrong
        results[k] = pass_k_i
    return results


def _is_error_output(text):
    if text is None:
        return False
    raw = str(text).lower()
    return "traceback" in raw or "error" in raw or "exception" in raw


def _normalize_output(text):
    if text is None:
        return ""
    return " ".join(str(text).split())


def _build_error_masks(outputs):
    return [[_is_error_output(item) for item in row] for row in outputs]


def _hamming_distance(a, b, mask_a=None, mask_b=None):
    count = 0
    for i, (x, y) in enumerate(zip(a, b)):
        if mask_a is not None and mask_b is not None:
            if (i < len(mask_a) and mask_a[i]) or (i < len(mask_b) and mask_b[i]):
                continue
        if x != y:
            count += 1
    return count


def _non_error_match_count(a, b, mask_a=None, mask_b=None):
    count = 0
    for i, (x, y) in enumerate(zip(a, b)):
        if mask_a is not None and mask_b is not None:
            if (i < len(mask_a) and mask_a[i]) or (i < len(mask_b) and mask_b[i]):
                continue
        if x == y:
            count += 1
    return count


def _cluster_outputs(outputs, max_diff, error_masks=None):
    if not outputs:
        return []


    if max_diff <= 0:
        clusters = []
        for idx, out in enumerate(outputs):
            mask_idx = None if error_masks is None else (error_masks[idx] if idx < len(error_masks) else [])
            placed = False
            for cluster in clusters:
                ok = True
                for j in cluster:
                    mask_j = None if error_masks is None else (error_masks[j] if j < len(error_masks) else [])
                    if _hamming_distance(out, outputs[j], mask_idx, mask_j) > 0:
                        ok = False
                        break
                if ok:
                    cluster.append(idx)
                    placed = True
                    break
            if not placed:
                clusters.append([idx])
        return clusters
    parent = list(range(len(outputs)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            if (
                _hamming_distance(
                    outputs[i],
                    outputs[j],
                    None if error_masks is None else error_masks[i],
                    None if error_masks is None else error_masks[j],
                )
                <= max_diff
            ):
                union(i, j)
    clusters = {}
    for idx in range(len(outputs)):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)
    return list(clusters.values())


def _pick_cluster_and_center(clusters, outputs, candidate_indices, pass_counts, error_masks=None):
    if not clusters:
        return [], -1, 0
    def cluster_match_score(cluster):
        score = 0
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                ii = cluster[i]
                jj = cluster[j]
                score += _non_error_match_count(
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
    if len(best_cluster) == 1:
        return best_cluster, best_cluster[0], 0
    best_center = best_cluster[0]
    best_key = None
    best_dist = 0
    for i in best_cluster:
        dist_sum = sum(
            _hamming_distance(
                outputs[i],
                outputs[j],
                None if error_masks is None else error_masks[i],
                None if error_masks is None else error_masks[j],
            )
            for j in best_cluster
        )
        match_sum = sum(
            _non_error_match_count(
                outputs[i],
                outputs[j],
                None if error_masks is None else error_masks[i],
                None if error_masks is None else error_masks[j],
            )
            for j in best_cluster
            if j != i
        )
        err_count = sum(error_masks[i]) if error_masks is not None else 0
        key = (-match_sum, err_count, -pass_counts[candidate_indices[i]], candidate_indices[i])
        if best_key is None or key < best_key:
            best_key = key
            best_center = i
            best_dist = dist_sum
    return best_cluster, best_center, best_dist


def _format_bool_matrix_rows(mat):
    if mat is None:
        return []
    rows = []
    for row in mat:
        rows.append("".join("1" if bool(x) else "0" for x in row))
    return rows


def _run_random_ut_batch(code_list, ut_inputs, time_limit, args):
    if not code_list or not ut_inputs:
        return [], []
    code_list_expanded = []
    input_list = []
    time_limit_list = []
    for code in code_list:
        for ut_input in ut_inputs:
            code_list_expanded.append(code)
            input_list.append(ut_input)
            time_limit_list.append(time_limit)
    outputs = execution.run_scripts_with_chunk(
        code_list_expanded,
        input_list,
        time_limit_list,
        args.num_chunks,
        args.exe_verbose,
    )
    n_col = len(ut_inputs)
    exe_matrix = []
    bool_matrix = []
    for i in range(len(code_list)):
        row = outputs[i * n_col : (i + 1) * n_col]
        exe_matrix.append(row)
        bool_matrix.append([not _is_error_output(x) for x in row])
    return exe_matrix, bool_matrix


def _select_random_inputs(inputs, target_count, placeholder):
    if target_count <= 0:
        return []
    if len(inputs) >= target_count:
        return list(inputs[:target_count])
    if not inputs:
        return [placeholder] * target_count
    padded = list(inputs)
    while len(padded) < target_count:
        padded.append(placeholder)
    return padded


def _execute_random_ut_batches(
    data,
    args,
    random_ut_max_attempts,
    random_ut_batch,
    random_ut_total,
    max_scale_code,
    random_inputs_key="random_case_input",
):
    random_exec_cache = {}
    if random_ut_max_attempts <= 0 or random_ut_batch <= 0:
        return random_exec_cache
    for attempt in range(random_ut_max_attempts):
        code_list = []
        input_list = []
        time_limit_list = []
        index_list = []
        position_list = []
        batch_len_by_task = {}
        for i in range(len(data)):
            if data[i].get("case_exe_results") is None or data[i].get("test_exe_results") is None:
                continue
            random_inputs = data[i].get(random_inputs_key, [])
            if random_ut_total > 0:
                random_inputs = random_inputs[:random_ut_total]
            start = attempt * random_ut_batch
            end = start + random_ut_batch
            batch_inputs = random_inputs[start:end]
            if not batch_inputs:
                continue
            max_code = min(max_scale_code, len(data[i].get("generated_code", [])))
            if max_code <= 0:
                continue
            data[i][f"random_case_exe_results_{attempt + 1}"] = [
                ["" for _ in range(len(batch_inputs))]
                for _ in range(max_code)
            ]
            batch_len_by_task[i] = len(batch_inputs)
            for code_idx in range(max_code):
                code = data[i]["generated_code"][code_idx]
                for ut_idx, ut_input in enumerate(batch_inputs):
                    code_list.append(code)
                    input_list.append(ut_input)
                    time_limit_list.append(data[i].get("test_time_limit", 1))
                    index_list.append(i)
                    position_list.append((code_idx, ut_idx))
        if code_list:
            outputs = execution.run_scripts_with_chunk(
                code_list,
                input_list,
                time_limit_list,
                args.num_chunks,
                args.exe_verbose,
            )
            for idx, output in enumerate(outputs):
                task_i = index_list[idx]
                code_idx, ut_idx = position_list[idx]
                data[task_i][f"random_case_exe_results_{attempt + 1}"][code_idx][ut_idx] = output
        for task_i, _ in batch_len_by_task.items():
            exe_matrix = data[task_i].get(f"random_case_exe_results_{attempt + 1}", [])
            if not exe_matrix:
                continue
            bool_matrix = [
                [not _is_error_output(x) for x in row]
                for row in exe_matrix
            ]
            bool_rows = _format_bool_matrix_rows(bool_matrix)
            data[task_i][f"random_case_bool_table_rows_{attempt + 1}"] = bool_rows
            random_exec_cache[(task_i, attempt)] = (exe_matrix, bool_rows)
    return random_exec_cache


def _get_outputs_result_name(args, outputs_name):
    if args.is_final_eval:
        outputs_result_name = (
            f"../CURE_results/results_final_eval/{args.mode}/"
            + outputs_name 
            + "_final_eval.txt"
        )
    else:
        outputs_result_name = (
            f"../CURE_results/results_optimization_eval/{args.mode}/"
            + outputs_name 
            + ".txt"
        )
    return outputs_result_name


def _open_result_file(args, outputs_name):
    outputs_result_name = _get_outputs_result_name(args, outputs_name)
    os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
    f = open(outputs_result_name, "a")
    return f, outputs_result_name


def _safe_divide(d1, d2):
    return d1 / d2 if d2 != 0 else 0


def _ut_input_rank(case_inputs):
    if not case_inputs:
        return 0, 0
    n = len(case_inputs)
    eq = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            eq[i, j] = 1.0 if case_inputs[i] == case_inputs[j] else 0.0
    rank = int(np.linalg.matrix_rank(eq))
    return rank, n


# Evaluation modes.
def _evaluate_single_eval_mode(data, outputs_name, mean_code, mean_case, args):
    code_score = 0
    code_num = 0
    code_acc_score = 0
    code_acc_num = 0

    for i in range(len(data)):
        if data[i]["test_exe_results"] is None or data[i]["test_bool_table"] is None:
            continue
        all_test_table_i = np.array(data[i]["test_bool_table"]).copy()


        correct_code_list = np.where(all_test_table_i.all(axis=1))[0].tolist()
        code_score += len(correct_code_list)
        code_num += all_test_table_i.shape[0]

        code_acc_score += np.sum(all_test_table_i).item()
        code_acc_num += all_test_table_i.shape[0] * all_test_table_i.shape[1]

    code_acc = _safe_divide(code_score, code_num)
    code_acc_acc = _safe_divide(code_acc_score, code_acc_num)

    f, outputs_result_name = _open_result_file(args, outputs_name)
    with f:

        def save_and_print(text):
            cprint(text, color="green")
            f.write(text + "\n")

        save_and_print(
            f"code acc (average proportion of tasks the generated code can pass): {code_acc}\n"
            f"code accumulate acc (average proportion of unit tests the generated code can pass): {code_acc_acc}"
        )

        save_and_print(f"code average response length: {mean_code}")

    return data


def _evaluate_passatk_mode(data, outputs_name, mean_code, mean_case, args):
    pass_at_k_values = sorted(set(args.pass_at_k_list)) if args.pass_at_k_list else []

    pass_at_k_scores = {k: [] for k in pass_at_k_values}

    code_score = 0
    code_num = 0
    code_acc_score = 0
    code_acc_num = 0

    for i in range(len(data)):
        if data[i]["test_exe_results"] is None or data[i]["test_bool_table"] is None:
            continue
        all_test_table_i = np.array(data[i]["test_bool_table"]).copy()


        correct_code_list = np.where(all_test_table_i.all(axis=1))[0].tolist()
        code_score += len(correct_code_list)
        code_num += all_test_table_i.shape[0]

        code_acc_score += np.sum(all_test_table_i).item()
        code_acc_num += all_test_table_i.shape[0] * all_test_table_i.shape[1]


        if pass_at_k_values:
            task_passk = compute_pass_at_k_for_task(all_test_table_i, pass_at_k_values)
            for k, v in task_passk.items():
                pass_at_k_scores[k].append(v)

    code_acc = _safe_divide(code_score, code_num)
    code_acc_acc = _safe_divide(code_acc_score, code_acc_num)


    overall_pass_at_k = {}
    for k in pass_at_k_values:
        vals = pass_at_k_scores.get(k, [])
        overall_pass_at_k[k] = sum(vals) / len(vals) if vals else 0.0

    f, outputs_result_name = _open_result_file(args, outputs_name)
    with f:

        def save_and_print(text):
            cprint(text, color="green")
            f.write(text + "\n")

        save_and_print(
            f"code acc (average proportion of tasks the generated code can pass): {code_acc}\n"
            f"code accumulate acc (average proportion of unit tests the generated code can pass): {code_acc_acc}"
        )

        for k in sorted(overall_pass_at_k.keys()):
            save_and_print(f"pass@{k}: {overall_pass_at_k[k]}")


        save_and_print(f"code average response length: {mean_code}")

    return data


def _evaluate_bon_mode(data, outputs_name, mean_code, mean_case, args):
    

    pass_at_k_values = sorted(set(args.pass_at_k_list)) if args.pass_at_k_list else []
    pass_at_k_scores = {k: [] for k in pass_at_k_values}


    stats_single = {
        "BoN_score": 0,
        "BoN_num": 0,
        "BoN_acc_score": 0,
        "BoN_acc_num": 0,
    }

    stats = []
    stats_all = []
    for tpl in (args.scale_tuple_list or []):
        stats_i = {
            "tuple": tpl,
            "BoN_score": 0,
            "BoN_num": 0,
            "BoN_acc_score": 0,
            "BoN_acc_num": 0,
            "passed_tasks": [],
        }
        stats.append(stats_i)
        stats_all_i = {
            "tuple": tpl,
            "BoN_all_score": 0,
            "BoN_all_num": 0,
            "BoN_all_acc_score": 0,
            "BoN_all_acc_num": 0,
            "tie_counts": [],
        }
        stats_all.append(stats_all_i)

    skip_bon_log = bool(getattr(args, "skip_bon_log", False))

    code_score = 0; code_num = 0
    code_acc_score = 0; code_acc_num = 0
    case_score = 0; case_num = 0
    case_acc_score = 0; case_acc_num = 0
    p_01_score = 0; p_01_num = 0
    p_00_score = 0; p_00_num = 0
    ut_rank_sum = 0; ut_rank_norm_sum = 0; ut_rank_num = 0
    bon_debug_records = []
    random_exec_cache = {}
    use_random_ut_cluster = bool(getattr(args, "use_random_ut_cluster", True))
    random_ut_total = int(getattr(args, "random_ut_total", 80))
    random_ut_batch = int(getattr(args, "random_ut_batch", 16))
    random_ut_max_attempts = int(getattr(args, "random_ut_max_attempts", 5))
    random_ut_min_top_count = int(getattr(args, "random_ut_min_top_count", 2))
    random_ut_cluster_max_diff = int(getattr(args, "random_ut_cluster_max_diff", 0))
    random_ut_placeholder = getattr(
        args, "random_ut_placeholder", "We can not extract the input in the output. "
    )
    random_ut_select_count = random_ut_batch
    random_ut_exec_rounds = 1
    max_scale_code = max([tpl[0] for tpl in args.scale_tuple_list], default=0)
    if use_random_ut_cluster:
        for i in range(len(data)):
            for attempt in range(random_ut_max_attempts):
                data[i].setdefault(f"random_case_bool_table_rows_{attempt + 1}", [])
                data[i].setdefault(f"random_case_exe_results_{attempt + 1}", [])
            raw_inputs = data[i].get("random_case_input", [])
            if random_ut_total > 0:
                raw_inputs = raw_inputs[:random_ut_total]
            data[i]["random_case_input_selected"] = _select_random_inputs(
                raw_inputs,
                random_ut_select_count,
                random_ut_placeholder,
            )
        if random_ut_select_count > 0 and max_scale_code > 0:
            random_exec_cache = _execute_random_ut_batches(
                data,
                args,
                random_ut_exec_rounds,
                random_ut_select_count,
                random_ut_select_count,
                max_scale_code,
                random_inputs_key="random_case_input_selected",
            )

    for i in range(len(data)):

        if data[i]["case_exe_results"] is None or data[i]["test_exe_results"] is None:
            continue

        all_case_table_i = np.array(data[i]["case_bool_table"]).copy()
        all_test_table_i = np.array(data[i]["test_bool_table"]).copy()
        case_is_valid = None
        valid_indices = None


        if args.generation_mode == "plansearch":
            case_is_valid = data[i].get("case_is_valid", [True] * all_case_table_i.shape[1])
            valid_indices = [j for j, valid in enumerate(case_is_valid) if valid]

            all_case_table_i = all_case_table_i[:, valid_indices].copy()


        case_inputs = data[i].get("case_input", [])
        if valid_indices is not None:
            case_inputs = [case_inputs[j] for j in valid_indices if j < len(case_inputs)]
        ut_rank, ut_n = _ut_input_rank(case_inputs)
        if ut_n > 0:
            ut_rank_sum += ut_rank
            ut_rank_norm_sum += (ut_rank / ut_n)
            ut_rank_num += 1

        debug_entry = {
            "idx": i,
            "case_shape": list(all_case_table_i.shape),
            "test_shape": list(all_test_table_i.shape),
            "case_is_valid": case_is_valid,
            "valid_indices": valid_indices,
            "case_table": all_case_table_i.tolist(),
            "test_table": all_test_table_i.tolist(),
            "bon_slices": [],
        }


        if pass_at_k_values:
            task_passk = compute_pass_at_k_for_task(all_test_table_i, pass_at_k_values)
            for k, v in task_passk.items():
                pass_at_k_scores[k].append(v)


        correct_code_list = np.where(all_test_table_i.all(axis=1))[0].tolist()
        code_score += len(correct_code_list)
        code_num += all_test_table_i.shape[0]
        code_acc_score += np.sum(all_test_table_i).item()
        code_acc_num += all_test_table_i.shape[0] * all_test_table_i.shape[1]


        sub_case_table_i = all_case_table_i[correct_code_list, :].copy()
        
        if len(correct_code_list) > 0 and all_case_table_i.shape[1] > 0:
            correct_case_list = np.where(sub_case_table_i.all(axis=0))[0].tolist()
            case_score += len(correct_case_list)
            case_num += sub_case_table_i.shape[1]
            case_acc_score += np.sum(sub_case_table_i).item()
            case_acc_num += sub_case_table_i.shape[0] * sub_case_table_i.shape[1]


            wrong_code_list = [
                j for j in range(all_case_table_i.shape[0]) if j not in correct_code_list
            ]
            wrong_case_list = [
                j for j in range(all_case_table_i.shape[1]) if j not in correct_case_list
            ]
            if len(wrong_code_list) > 0:
                if len(correct_case_list) > 0:
                    wrong_code_correct_case_table_i = all_case_table_i[wrong_code_list, :][
                        :, correct_case_list
                    ].copy()
                    p_01_score += np.sum(~wrong_code_correct_case_table_i).item()
                    p_01_num += (
                        wrong_code_correct_case_table_i.shape[0]
                        * wrong_code_correct_case_table_i.shape[1]
                    )
                if len(wrong_case_list) > 0:
                    wrong_code_wrong_case_table_i = all_case_table_i[wrong_code_list, :][
                        :, wrong_case_list
                    ].copy()
                    p_00_score += np.sum(wrong_code_wrong_case_table_i).item()
                    p_00_num += (
                        wrong_code_wrong_case_table_i.shape[0]
                        * wrong_code_wrong_case_table_i.shape[1]
                    )


        if len(args.scale_tuple_list) > 0:
            index_id = 0
            for scale_num_code, scale_num_case in args.scale_tuple_list:


                actual_num_case = min(scale_num_case, all_case_table_i.shape[1])
                case_table_i = all_case_table_i[:scale_num_code, :actual_num_case].copy()
                test_table_i = all_test_table_i[:scale_num_code, :].copy()
                
                pass_counts = np.sum(case_table_i, 1) if case_table_i.shape[0] > 0 else np.array([])

                if case_table_i.shape[1] > 0 and pass_counts.size > 0:
                    best_code_index = int(pass_counts.argmax())
                else:
                    best_code_index = 0

                sub_test_table_i = test_table_i[best_code_index, :].copy()
                stats[index_id]["BoN_score"] += int(all(sub_test_table_i))
                stats[index_id]["BoN_num"] += 1
                stats[index_id]["BoN_acc_score"] += np.sum(sub_test_table_i).item()
                stats[index_id]["BoN_acc_num"] += len(sub_test_table_i)
                if all(sub_test_table_i):
                    stats[index_id]["passed_tasks"].append(i)

                if pass_counts.size > 0:
                    max_pass = pass_counts.max()
                    top_candidates = np.where(pass_counts == max_pass)[0].tolist()
                    tie_count = len(top_candidates)
                    if tie_count == 0:
                        tie_count = 1
                        top_candidates = [best_code_index]
                    best_top_idx = top_candidates[0]
                    best_top_score = -1
                    for cand in top_candidates:
                        cand_row = test_table_i[cand, :].copy()
                        cand_score = cand_row.mean() if len(cand_row) > 0 else 0
                        if cand_score > best_top_score:
                            best_top_score = cand_score
                            best_top_idx = cand
                    best_row = test_table_i[best_top_idx, :].copy()
                    stats_all[index_id]["BoN_all_score"] += int(all(best_row))
                    stats_all[index_id]["BoN_all_num"] += 1
                    stats_all[index_id]["BoN_all_acc_score"] += np.sum(best_row).item()
                    stats_all[index_id]["BoN_all_acc_num"] += len(best_row)
                    stats_all[index_id]["tie_counts"].append(tie_count)
                debug_entry["bon_slices"].append(
                    {
                        "tuple": [scale_num_code, scale_num_case],
                        "case_slice_shape": list(case_table_i.shape),
                        "test_slice_shape": list(test_table_i[:scale_num_code, :].shape),
                        "case_slice": case_table_i.tolist(),
                        "test_slice": test_table_i[:scale_num_code, :].tolist(),
                        "best_code_index": int(best_code_index),
                        "best_code_test_row": sub_test_table_i.tolist(),
                    }
                )
                index_id += 1
        bon_debug_records.append(debug_entry)


    code_acc = _safe_divide(code_score, code_num)
    code_acc_acc = _safe_divide(code_acc_score, code_acc_num)
    

    case_acc = _safe_divide(case_score, case_num)
    case_acc_acc = _safe_divide(case_acc_score, case_acc_num)
    p_01 = _safe_divide(p_01_score, p_01_num)
    p_00 = _safe_divide(p_00_score, p_00_num)


    overall_pass_at_k = {}
    for k in pass_at_k_values:
        vals = pass_at_k_scores.get(k, [])
        overall_pass_at_k[k] = sum(vals) / len(vals) if vals else 0.0


    f, outputs_result_name = _open_result_file(args, outputs_name)
    with f:
        def save_and_print(text):
            cprint(text, color="green")
            f.write(text + "\n")

        save_and_print("=== Evaluation Report ===")
        

        save_and_print(
            f"code acc (average proportion of tasks the generated code can pass): {code_acc}\n"
            f"code accumulate acc (average proportion of unit tests the generated code can pass): {code_acc_acc}"
        )
        

        correct_task_indices = [i for i, d in enumerate(data) if d.get("is_correct", False)]
        save_and_print(f"Correct task indices: {correct_task_indices}")
        

        if overall_pass_at_k:
            save_and_print("--- Pass@k Metrics ---")
            for k in sorted(overall_pass_at_k.keys()):
                save_and_print(f"pass@{k}: {overall_pass_at_k[k]}")


        save_and_print("--- Generated Unit Test Quality ---")
        save_and_print(
            f"estimated unit test acc (average proportion of tasks that the generated unit test can pass all correct code): {case_acc}\n"
            f"estimated unit test accumulate acc (average proportion of correct code that the generated unit test can pass): {case_acc_acc}"
        )
        save_and_print(f"estimated p_01 (False Rejection Rate): {1 - p_01}")
        save_and_print(f"estimated p_00 (False Acceptance Rate): {p_00}")


        if ut_rank_num > 0:
            ut_rank_avg = ut_rank_sum / ut_rank_num
            ut_rank_norm_avg = ut_rank_norm_sum / ut_rank_num
            save_and_print("--- UT Input Redundancy ---")
            save_and_print(f"ut input rank avg: {ut_rank_avg}")
            save_and_print(f"ut input rank norm avg: {ut_rank_norm_avg}")


        if (not skip_bon_log) and (args.scale_tuple_list or []):
            save_and_print("--- Best-of-N Metrics ---")

            hist = getattr(args, "_new_bon_cached", None)
            for st, st_all in zip(stats, stats_all):
                tuple_name = st["tuple"]
                if st["BoN_num"] == 0 or st["BoN_acc_num"] == 0:
                    continue
                acc = st["BoN_score"] / st["BoN_num"]
                acc_acc = st["BoN_acc_score"] / st["BoN_acc_num"]
                save_and_print(f"BoN setting {tuple_name}:")
                save_and_print(f"acc: {acc}, accumulate acc: {acc_acc}")
                passed = st.get("passed_tasks", [])
                passed_str = ", ".join(str(x) for x in passed) if passed else "None"
                save_and_print(f"passed task indices: {passed_str}")

                tuple_name = st_all.get("tuple", st.get("tuple"))
                if st_all.get("BoN_all_num", 0) == 0 or st_all.get("BoN_all_acc_num", 0) == 0:
                    pass
                else:
                    acc_all = st_all["BoN_all_score"] / st_all["BoN_all_num"]
                    acc_acc_all = st_all["BoN_all_acc_score"] / st_all["BoN_all_acc_num"]
                    tie_counts = st_all.get("tie_counts", [])
                    tie_info = (
                        f"max_top_candidates={max(tie_counts)}, "
                        f"avg_top_candidates={sum(tie_counts)/len(tie_counts):.2f}, "
                        f"ties_gt1={sum(1 for t in tie_counts if t>1)}"
                    ) if tie_counts else "no data"
                    save_and_print(f"BoN_all setting {tuple_name}:")
                    save_and_print(f"acc: {acc_all}, accumulate acc: {acc_acc_all}")
                    save_and_print(f"top-candidate tie summary: {tie_info}")


                if hist:
                    save_and_print(f"New_BoN setting {tuple_name}:")
                    save_and_print(
                        f"acc: {hist['new_bon']}, front: {hist['new_bon_front']}, back: {hist['new_bon_back']}"
                    )
                else:
                    save_and_print(f"New_BoN setting {tuple_name}:")
                    save_and_print("acc: N/A")


        save_and_print(
            f"code average response length: {mean_code}, unit test average response length: {mean_case}"
        )


    if not skip_bon_log:
        debug_outputs_result_name = _get_outputs_result_name(args, outputs_name)
        debug_json_path = os.path.splitext(debug_outputs_result_name)[0] + "_bon_tables.json"
        os.makedirs(os.path.dirname(debug_json_path), exist_ok=True)
        with open(debug_json_path, "w") as df:
            json.dump(bon_debug_records, df, indent=2)
        cprint(f"[BoN Debug] Saved BoN boolean tables to {debug_json_path}", color="yellow")

    return data


# Public metrics entrypoint.
def compute_and_log_metrics(data, outputs_name, mean_code, mean_case, args):

    if args.single_eval:

        data = _evaluate_single_eval_mode(data, outputs_name, mean_code, mean_case, args)

    elif args.eval_pass_at_k_only:

        data = _evaluate_passatk_mode(data, outputs_name, mean_code, mean_case, args)

    elif args.eval_bon:

        data = _evaluate_bon_mode(data, outputs_name, mean_code, mean_case, args)

    else:

        raise ValueError(
            "No evaluation mode selected. "
            "Please set exactly ONE of {single_eval, eval_pass_at_k_only, eval_bon} to True."
        )


    for i in range(len(data)):
        if data[i]["case_exe_results"] is not None and isinstance(
            data[i]["case_bool_table"], (np.ndarray, list)
        ):
            if isinstance(data[i]["case_bool_table"], np.ndarray):
                data[i]["case_bool_table"] = data[i]["case_bool_table"].tolist()

        if data[i]["test_exe_results"] is not None and isinstance(
            data[i]["test_bool_table"], (np.ndarray, list)
        ):
            if isinstance(data[i]["test_bool_table"], np.ndarray):
                data[i]["test_bool_table"] = data[i]["test_bool_table"].tolist()

    return data
