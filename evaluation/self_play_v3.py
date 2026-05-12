# Self-play loop for refining generated code and unit tests.
import json
import os
import copy

from execution import run_all_executions_generate, run_all_executions, _compute_new_bon_with_history
from inference import ModelRunner
from UT_config import (
    get_ut_input_generation_prompt,
    get_ut_input_random_generation_prompt,
    get_ut_output_generation_prompt,
    get_ut_output_refine_prompt,
    get_full_prompt,
    get_fix_code_prompt_with_ut,
)
from generator_v3 import extract_code, extract_ut_input, extract_ut_output
from generator_v3 import parse_random_case_inputs, _strip_case_prefix
from collections import Counter
import math
import random
import hashlib
from prompts import get_stage2_prompt
from metrics import compute_and_log_metrics
from usage_tracking import (
    consume_item_usage,
    format_usage_round_summary,
    record_direct_usage,
    self_play_round_key,
    sync_usage_to_data,
)


def convert_ndarray(obj):
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)

def _format_bool_matrix(matrix, max_rows=8, max_cols=16):
    if matrix is None:
        return "[empty bool matrix]"
    try:
        import numpy as np

        if isinstance(matrix, np.ndarray):
            matrix = matrix.tolist()
    except Exception:
        pass
    if not matrix:
        return "[empty bool matrix]"
    first_row = matrix[0]
    if isinstance(first_row, (list, tuple)):
        cols = len(first_row)
    else:
        cols = len(matrix)
        matrix = [matrix]
    preview = []
    row_count = min(len(matrix), max_rows)
    col_count = min(cols, max_cols)
    for r in range(row_count):
        row = matrix[r][:col_count]
        preview.append(" ".join("1" if cell else "0" for cell in row))
    suffix = ""
    if len(matrix) > max_rows or len(matrix[0]) > max_cols:
        suffix = f" ... (showing {row_count}x{col_count} of {len(matrix)}x{len(matrix[0])})"
    return "\n        " + "\n        ".join(preview) + suffix


def _log_problem_bool_matrix(data, problem_idx, label):
    if not data or problem_idx >= len(data):
        print(f"    - {label}: problem data unavailable.", flush=True)
        return
    table = data[problem_idx].get("case_bool_table")
    if table is None:
        table = data[problem_idx].get("refined_bool_table")
    if table is None:
        print(f"    - {label}: bool matrix missing.", flush=True)
        return
    print(f"    - {label}: bool matrix preview:\n{_format_bool_matrix(table)}", flush=True)


def _record_round_bool_history(data, round_idx):
    for prob_idx, item in enumerate(data):
        bool_table = item.get("case_bool_table")
        if bool_table is None:
            continue
        history = item.setdefault("round_bool_history", [])
        history.append(
            {
                "round": round_idx,
                "problem_index": prob_idx,
                "bool_matrix": bool_table,
            }
        )


def _clear_round_transient_fields(data):
    transient_fields = [
        "ut_refine_round_trace",
    ]
    for item in data:
        for field in transient_fields:
            item.pop(field, None)


def _pick_new_attack_idea(data_i, idea_pool, ut_idx, fallback="Unknown"):
    if not isinstance(idea_pool, list) or not idea_pool:
        return fallback or "Unknown"

    pool = []
    for idea in idea_pool:
        if idea is None:
            continue
        idea_text = str(idea).strip()
        if idea_text:
            pool.append(idea_text)
    if not pool:
        return fallback or "Unknown"

    used = data_i.get("attack_ideas_used")
    if not isinstance(used, list):
        used = []
        data_i["attack_ideas_used"] = used
        existing = data_i.get("attack_ideas", [])
        pool_set = set(pool)
        if isinstance(existing, list):
            for idea in existing:
                if idea is None:
                    continue
                idea_text = str(idea).strip()
                if idea_text and idea_text in pool_set and idea_text not in used:
                    used.append(idea_text)

    used_set = set(used)
    if len(used_set) >= len(set(pool)):
        used.clear()
        used_set = set()

    start = ut_idx % len(pool) if pool else 0
    for offset in range(len(pool)):
        cand = pool[(start + offset) % len(pool)]
        if cand not in used_set:
            used.append(cand)
            return cand

    cand = pool[start]
    used.append(cand)
    return cand


# Step 0: resample codes that currently fail all generated tests.
def resample_code(data, runner, tokenizer, args, round_num=None):
    print("\n=== Step 0.5: Resample Codes with 0% Accuracy ===", flush=True)
    round_key = self_play_round_key(round_num)

    def _get_valid_indices(item, num_cases: int) -> list[int]:
        case_is_valid = item.get("case_is_valid")
        if not isinstance(case_is_valid, list) or len(case_is_valid) == 0:
            return list(range(num_cases))
        limit = min(len(case_is_valid), num_cases)
        valid = [j for j in range(limit) if bool(case_is_valid[j])]
        return valid if valid else list(range(num_cases))

    def _normalize_stage2_obs_list(item) -> list[str]:
        obs_list = item.get("stage2_observations_list")
        if isinstance(obs_list, list) and obs_list:
            return [str(x).strip() for x in obs_list if str(x).strip()]
        stage2_text = item.get("stage2_observations", "")
        if not isinstance(stage2_text, str) or not stage2_text.strip():
            return []
        lines = []
        for line in stage2_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("-"):
                line = line[1:].strip()
            if line:
                lines.append(line)
        return lines

    def _build_code_prompt(item: dict, problem: str, second_order_obs: str) -> str:
        problem_template = problem
        cg = item.get("code_generation_prompts")
        if isinstance(cg, dict) and cg.get("stage2_template"):
            problem_template = cg["stage2_template"]

        obs_text = second_order_obs.strip()
        if obs_text and not obs_text.startswith("-"):
            obs_text = f"- {obs_text}"
        obs_text = obs_text + "\n"

        tmpl = getattr(args, "system_prompts_stage4_ablation_only_stage", None)
        if not isinstance(tmpl, str) or not tmpl.strip():
            tmpl = getattr(args, "system_prompts", "")
        return get_stage2_prompt(problem_template, obs_text, tmpl)

    resample_tasks = []
    resample_prompts = []
    

    for idx, data_i in enumerate(data):
        data_i.setdefault("code_resample_stats", {"self_play": 0})

    for idx, data_i in enumerate(data):

        if data_i.get("skip_self_play", False):
            continue
        
        bool_table = data_i.get("case_bool_table")
        if bool_table is None or len(bool_table) == 0:
            print(f"Error: - Problem {idx}: No execution results.", flush=True)
            continue

        num_codes = len(bool_table)
        num_cases = len(bool_table[0]) if num_codes > 0 else 0
        if num_codes == 0 or num_cases == 0:
            print(f"Error: - Problem {idx}: Empty bool table.", flush=True)
            continue

        valid_indices = _get_valid_indices(data_i, num_cases)
        if not valid_indices:
            print(f"Error: - Problem {idx}: No valid UT indices.", flush=True)
            continue

        stage2_obs_list = _normalize_stage2_obs_list(data_i)
        if not stage2_obs_list:
            print(
                f"Warning: - Problem {idx}: No stage2 observations available, skip resample.",
                flush=True,
            )
            continue

        generated_codes = data_i.get("generated_code", [])
        if not isinstance(generated_codes, list) or not generated_codes:
            continue


        used_map = data_i.setdefault("resample_used_stage2_obs", {})
        used_global = set(used_map.get("_global", []))
        start_idx = int(getattr(args, "k_code", 16))
        cursor = int(data_i.get("resample_stage2_cursor", start_idx))
        cursor = max(cursor, start_idx)
        resampled_code_indices = []

        for c_idx in range(min(num_codes, len(generated_codes))):
            row = bool_table[c_idx] if c_idx < len(bool_table) else []
            passed = 0
            for j in valid_indices:
                if j < len(row) and row[j]:
                    passed += 1

            if passed != 0:
                continue


            def _pick_next_obs():
                nonlocal cursor
                for scan_range in (
                    range(cursor, len(stage2_obs_list)),
                    range(start_idx, len(stage2_obs_list)),
                    range(0, len(stage2_obs_list)),
                ):
                    for j in scan_range:
                        obs = stage2_obs_list[j]
                        if obs and obs not in used_global:
                            cursor = j + 1
                            return obs
                return None

            chosen_obs = _pick_next_obs()
            if not chosen_obs:
                continue

            used_global.add(chosen_obs)
            used_map["_global"] = list(used_global)
            data_i["resample_stage2_cursor"] = cursor

            prompt = _build_code_prompt(data_i, data_i.get("question", ""), chosen_obs)
            resample_prompts.append(prompt)
            resample_tasks.append((idx, c_idx, chosen_obs))
            resampled_code_indices.append(c_idx)
            consume_item_usage(args, idx, "stage2_observation", [chosen_obs], round_key)

        if resampled_code_indices:
            round_prefix = f"Round {round_num}: " if round_num is not None else ""
            print(
                f"{round_prefix}Problem {idx}: Resample 0-accuracy codes: {sorted(resampled_code_indices)}",
                flush=True,
            )

    if not resample_prompts:
        return data, False

    outputs = runner.generate(resample_prompts)
    record_direct_usage(
        args,
        [prob_idx for prob_idx, _, _ in resample_tasks],
        resample_prompts,
        outputs,
        "self_play_code_resample",
        round_key,
    )
    for (prob_idx, code_idx, obs), full_output in zip(resample_tasks, outputs):
        new_code = extract_code(full_output)
        if not new_code or "We can not extract" in new_code:
            continue
        data[prob_idx]["generated_code"][code_idx] = new_code
        data[prob_idx]["full_code_generation"][code_idx] = full_output
        data[prob_idx].setdefault("resample_code_records", []).append(
            {
                "round": round_num,
                "code_idx": code_idx,
                "stage2_observation": obs,
            }
        )

        data[prob_idx]["code_resample_stats"]["self_play"] += 1

    return data, True

# Step 1: use discriminative UTs and their ideas to repair weak code candidates.
def ut_attack_code(data, runner, tokenizer, args, round_num=None):
    print("\n=== Step 1: UT Attack + 50% Filter + Code Fix ===", flush=True)
    round_key = self_play_round_key(round_num)


    target = 1.0
    skip_all_pass = getattr(args, "skip_attack_when_all_pass", True)
    refine_prompts = []
    refine_info = []

    for idx, data_i in enumerate(data):

        if data_i.get("skip_self_play", False):
            continue
        
        data_i["attack_ut_indices"] = []
        data_i["attack_ut_pass_rates"] = {}


        bool_table = data_i.get("case_bool_table")
        if bool_table is None or len(bool_table) == 0:
            print(f"Error: - Problem {idx}: No execution results.", flush=True)  
            continue  
        

        num_codes = len(bool_table)
        num_cases = len(bool_table[0]) if num_codes > 0 else 0
        if num_codes == 0 or num_cases == 0:
            print(f"Error: - Problem {idx}: Empty bool table.", flush=True)  
            continue  
        

        case_is_valid = data_i.get("case_is_valid", [True] * num_cases)
        valid_indices = [j for j, valid in enumerate(case_is_valid) if valid]
        if not valid_indices:
            print(f"Error: - Problem {idx}: No valid UTs.", flush=True)  
            continue  


        ut_stats = []
        for j in valid_indices:
            passed = 0
            for c in range(num_codes):
                if j < len(bool_table[c]) and bool_table[c][j]:
                    passed += 1
            rate = passed / num_codes if num_codes > 0 else 0.0
            ut_stats.append((j, rate))

        if not ut_stats:
            print(f"Error: - Problem {idx}: No UT stats to select.", flush=True)
            continue

        failing_ut = [(j, r) for (j, r) in ut_stats if r < 1.0]
        if failing_ut:
            selected, selected_rate = max(failing_ut, key=lambda x: x[1])
        else:
            if skip_all_pass:
                print(f"    - Problem {idx}: all UTs passed by all codes, skip attack.", flush=True)
                continue
            selected, selected_rate = max(ut_stats, key=lambda x: x[1])

        if selected_rate <= 0.0:
            print(f"    - Problem {idx}: selected UT pass rate is 0, skip attack.", flush=True)
            continue

        data_i["attack_ut_indices"].append(selected)
        data_i["attack_ut_pass_rates"] = {j: rate for j, rate in ut_stats}

        selected_input = ""
        selected_output = ""
        if selected < len(data_i.get("case_input", [])):
            selected_input = data_i["case_input"][selected]
        if selected < len(data_i.get("case_output", [])):
            selected_output = data_i["case_output"][selected]

        round_prefix = f"Round {round_num}: " if round_num is not None else ""
        print(
            f"{round_prefix}Problem {idx}: Selected UT index: {selected}, Pass Rate: {selected_rate:.2f}, "
            f"Content: Input: {selected_input}, Output: {selected_output}",
            flush=True,
        )

        if selected is None:
            continue

        problem = data_i.get("question", "")
        attack_ideas = data_i.get("attack_ideas", ["Unknown"] * num_cases)
        case_inputs = data_i.get("case_input", [])
        case_outputs = data_i.get("case_output", [])
        generated_codes = data_i.get("generated_code", [])

        for c_idx in range(num_codes):
            if c_idx >= len(generated_codes):
                continue  


            if selected < len(bool_table[c_idx]) and not bool_table[c_idx][selected]:
                failed_code = generated_codes[c_idx]
                attack_idea = attack_ideas[selected] if selected < len(attack_ideas) else "Unknown"
                attack_ut_input = case_inputs[selected] if selected < len(case_inputs) else ""
                attack_ut_output = case_outputs[selected] if selected < len(case_outputs) else ""
                exe_output = ""
                case_exe = data_i.get("case_exe_results")
                if case_exe and c_idx < len(case_exe) and selected < len(case_exe[c_idx]):
                    exe_output = case_exe[c_idx][selected]


                prompt_list = get_fix_code_prompt_with_ut(
                    problem=problem,
                    failed_code=failed_code,
                    attack_ut_input=[attack_ut_input],
                    attack_ut_output=[attack_ut_output],
                    exe_output=[exe_output],
                    num_to_include=1,
                )
                prompt = get_full_prompt(prompt_list)
                refine_prompts.append(prompt)
                refine_info.append((idx, c_idx))


    if not refine_prompts:
        for data_i in data:
            data_i["refined_codes"] = data_i.get("generated_code", []).copy()
            data_i["refined_codes_full"] = data_i.get("full_code_generation", []).copy()
        return data

    print(f"    - Generating {len(refine_prompts)} refined codes...", flush=True)
    refined_outputs = runner.generate(refine_prompts)
    record_direct_usage(
        args,
        [idx for idx, _ in refine_info],
        refine_prompts,
        refined_outputs,
        "self_play_code_fix",
        round_key,
    )


    for data_i in data:
        data_i["refined_codes"] = data_i.get("generated_code", []).copy()
        data_i["refined_codes_full"] = data_i.get("full_code_generation", []).copy()

    round_label = f"Round {round_num}" if round_num is not None else "Round ?"
    for (idx, c_idx), output in zip(refine_info, refined_outputs):
        new_code = extract_code(output)
        data[idx]["refined_codes"][c_idx] = new_code
        data[idx]["refined_codes_full"][c_idx] = output


    print("    - Step 1 refine complete.", flush=True)
    return data

# Step 2: refine low-agreement UT outputs before code repair.
def ut_refine(data, runner, tokenizer, args, round_num=None):


    print("\n=== Step 0.5: UT Refine (lowest non-0/1 pass-rate) ===", flush=True)
    round_key = self_play_round_key(round_num)
    max_attempts = getattr(args, "ut_regen_max_attempts", 5)
    sc_num = getattr(args, "self_consistency_num", 4)
    min_consistency = max(1, math.ceil(sc_num * 0.75)) if sc_num > 1 else 1
    did_refine = False
    ut_states = {}
    for data_i in data:
        data_i.pop("ut_refine_round_trace", None)
    for idx, data_i in enumerate(data):
        if data_i.get("skip_self_play", False):
            continue
        bool_table = data_i.get("case_bool_table")
        if bool_table is None or len(bool_table) == 0:
            print(f"    - Problem {idx}: No execution results.", flush=True)
            continue
        num_codes = len(bool_table)
        num_cases = len(bool_table[0]) if num_codes > 0 else 0
        if num_codes == 0 or num_cases == 0:
            print(f"    - Problem {idx}: Empty bool table.", flush=True)
            continue
        case_is_valid = data_i.get("case_is_valid", [True] * num_cases)
        valid_indices = [j for j, valid in enumerate(case_is_valid) if valid and j < num_cases]
        if not valid_indices:
            print(f"    - Problem {idx}: No valid UTs.", flush=True)
            continue
        ut_stats = []
        for j in valid_indices:
            passed = 0
            for c in range(num_codes):
                if j < len(bool_table[c]) and bool_table[c][j]:
                    passed += 1
            rate = passed / num_codes if num_codes > 0 else 0.0
            ut_stats.append((j, rate))
        candidates = [(j, r) for (j, r) in ut_stats if 0.0 < r < 1.0]
        if not candidates:
            continue
        selected, selected_rate = min(candidates, key=lambda x: x[1])
        problem = data_i.get("question", "")
        case_inputs = data_i.get("case_input", [])
        case_outputs = data_i.get("case_output", [])
        case_output_original = data_i.get("case_output_original", [])
        case_input_original = data_i.get("case_input_original", [])
        case_sources = data_i.get("case_source", [])
        full_case_generation = data_i.get("full_case_generation", [])
        case_text = data_i.get("case_text", [])
        attack_ideas = data_i.get("attack_ideas", [])
        attack_ideas_candidates = data_i.get("attack_ideas_candidates", [])
        k_code = len(data_i.get("generated_code", []))
        def _ensure_len(lst, length, fill_val):
            while len(lst) < length:
                lst.append(fill_val)
        _ensure_len(case_inputs, num_cases, "")
        _ensure_len(case_outputs, num_cases, "")
        _ensure_len(case_output_original, num_cases, "")
        _ensure_len(case_input_original, num_cases, "")
        _ensure_len(case_sources, num_cases, "unknown")
        _ensure_len(full_case_generation, num_cases, "")
        _ensure_len(case_text, num_cases, "")
        _ensure_len(attack_ideas, num_cases, "Unknown")
        _ensure_len(case_is_valid, num_cases, True)
        data_i["case_input"] = case_inputs
        data_i["case_output"] = case_outputs
        data_i["case_output_original"] = case_output_original
        data_i["case_input_original"] = case_input_original
        data_i["case_source"] = case_sources
        data_i["full_case_generation"] = full_case_generation
        data_i["case_text"] = case_text
        data_i["attack_ideas"] = attack_ideas
        data_i["case_is_valid"] = case_is_valid

        current_idea = attack_ideas[selected] if selected < len(attack_ideas) else "Unknown"
        idea_pool = attack_ideas_candidates if attack_ideas_candidates else attack_ideas
        new_idea = _pick_new_attack_idea(data_i, idea_pool, selected, fallback=current_idea)
        consume_item_usage(args, idx, "attack_idea", [new_idea], round_key)
        previous_ut = ""
        if selected < len(case_inputs):
            previous_ut += f"Input:\n{case_inputs[selected]}\n"
        if selected < len(case_outputs):
            previous_ut += f"Expected Output:\n{case_outputs[selected]}\n"
        previous_code = ""
        generated_codes = data_i.get("generated_code", [])
        for c_idx in range(num_codes):
            if selected < len(bool_table[c_idx]) and bool_table[c_idx][selected]:
                if c_idx < len(generated_codes):
                    previous_code = generated_codes[c_idx]
                break
        
        input_prompt = get_full_prompt(
            get_ut_input_generation_prompt(problem, new_idea)
        )
        ut_states[(idx, selected)] = {
            "prob_idx": idx,
            "ut_idx": selected,
            "problem": problem,
            "idea": new_idea,
            "input_prompt": input_prompt,
            "input_output_raw": "",
            "input": "",
            "base_output_prompt": "",
            "previous_ut": previous_ut,
            "previous_code": previous_code,
            "input_ok": False,
            "last_failed_output_raw": "",
            "selected_rate": selected_rate,
            "attempts": 0,
            "success": False,
        }

    if not ut_states:
        print("    - UT refine: no eligible UTs, skip.", flush=True)
        return data, did_refine

    print(
        f"    - UT refine: scheduling {len(ut_states)} UTs (sc={sc_num}, max_attempts={max_attempts})",
        flush=True,
    )

    input_tasks = []
    input_prompts = []
    for key, st in ut_states.items():
        input_tasks.append(key)
        input_prompts.append(st["input_prompt"])

    if input_prompts:
        print(f"    - UT refine: generating {len(input_prompts)} new inputs...", flush=True)
        input_outputs = runner.generate(input_prompts)
        record_direct_usage(
            args,
            [ut_states[key]["prob_idx"] for key in input_tasks],
            input_prompts,
            input_outputs,
            "self_play_ut_input_generation",
            round_key,
        )
        for key, input_raw in zip(input_tasks, input_outputs):
            st = ut_states[key]
            new_input = extract_ut_input(input_raw)
            if not new_input or "We can not extract" in new_input:
                st["input_output_raw"] = input_raw
                st["input"] = ""
                st["output_prompt"] = ""
                continue
            st["input_output_raw"] = input_raw
            st["input"] = new_input
            st["base_output_prompt"] = get_full_prompt(
                get_ut_output_refine_prompt(
                    st["problem"],
                    new_input,
                    st.get("previous_ut", ""),
                    st.get("previous_code", ""),
                )
            )
            st["input_ok"] = True

    for attempt in range(1, max_attempts + 1):
        pending_keys = [
            k for k, st in ut_states.items() if not st["success"] and st.get("base_output_prompt")
        ]
        if not pending_keys:
            break

        print(
            f"    - UT refine attempt {attempt}/{max_attempts}: pending {len(pending_keys)}",
            flush=True,
        )

        for key in pending_keys:
            ut_states[key]["attempts"] += 1

        attempt_prompts = []
        for key in pending_keys:
            st = ut_states[key]
            if attempt == 1:
                prompt = st["base_output_prompt"]
            else:
                prompt_list = get_ut_output_refine_prompt(
                    st["problem"],
                    st.get("input", ""),
                    st.get("previous_ut", ""),
                    st.get("previous_code", ""),
                )
                prompt = get_full_prompt(prompt_list)
            attempt_prompts.append(prompt)

        if sc_num > 1:
            expanded_prompts = []
            for prompt in attempt_prompts:
                expanded_prompts.extend([prompt] * sc_num)
            all_outputs = runner.generate(expanded_prompts)
            record_direct_usage(
                args,
                [ut_states[key]["prob_idx"] for key in pending_keys for _ in range(sc_num)],
                expanded_prompts,
                all_outputs,
                "self_play_ut_output_generation",
                round_key,
            )

            grouped_outputs = []
            cursor = 0
            for _ in pending_keys:
                grouped_outputs.append(all_outputs[cursor : cursor + sc_num])
                cursor += sc_num
        else:
            raw_outputs = runner.generate(attempt_prompts)
            record_direct_usage(
                args,
                [ut_states[key]["prob_idx"] for key in pending_keys],
                attempt_prompts,
                raw_outputs,
                "self_play_ut_output_generation",
                round_key,
            )
            grouped_outputs = [[o] for o in raw_outputs]

        for key, samples_raw in zip(pending_keys, grouped_outputs):
            if not samples_raw:
                continue

            samples_extracted = [extract_ut_output(o) for o in samples_raw]
            counts = Counter([s for s in samples_extracted if s.strip()])
            if not counts:
                st = ut_states[key]
                data[st["prob_idx"]].setdefault("ut_refine_round_trace", []).append(
                    {
                        "attempt": st["attempts"],
                        "prompt_type": "base" if attempt == 1 else "refine",
                        "samples_raw": samples_raw,
                        "samples_extracted": samples_extracted,
                        "winner_idx": None,
                        "winner_extracted": None,
                        "winner_raw": None,
                        "winner_freq": 0,
                        "min_consistency": min_consistency,
                        "carry_forward_idx": 0 if samples_raw else None,
                        "carry_forward_raw": samples_raw[0] if samples_raw else None,
                        "carry_forward_extracted": samples_extracted[0] if samples_extracted else None,
                        "accepted": False,
                    }
                )
                st["last_failed_output_raw"] = samples_raw[0] if samples_raw else ""
                continue
            winner, freq = counts.most_common(1)[0]
            winner_idx = samples_extracted.index(winner)
            winner_raw = samples_raw[winner_idx]
            if freq < min_consistency or "We can not extract" in winner:
                st = ut_states[key]
                data[st["prob_idx"]].setdefault("ut_refine_round_trace", []).append(
                    {
                        "attempt": st["attempts"],
                        "prompt_type": "base" if attempt == 1 else "refine",
                        "samples_raw": samples_raw,
                        "samples_extracted": samples_extracted,
                        "winner_idx": winner_idx,
                        "winner_extracted": winner,
                        "winner_raw": winner_raw,
                        "winner_freq": freq,
                        "min_consistency": min_consistency,
                        "carry_forward_idx": 0 if samples_raw else None,
                        "carry_forward_raw": samples_raw[0] if samples_raw else None,
                        "carry_forward_extracted": samples_extracted[0] if samples_extracted else None,
                        "accepted": False,
                    }
                )
                st["last_failed_output_raw"] = samples_raw[0] if samples_raw else ""
                continue

            ut_output_raw = winner_raw
            ut_output = winner

            st = ut_states[key]
            prob_idx = st["prob_idx"]
            ut_idx = st["ut_idx"]
            data[prob_idx].setdefault("ut_refine_round_trace", []).append(
                {
                    "attempt": st["attempts"],
                    "prompt_type": "base" if attempt == 1 else "refine",
                    "samples_raw": samples_raw,
                    "samples_extracted": samples_extracted,
                    "winner_idx": winner_idx,
                    "winner_extracted": winner,
                    "winner_raw": ut_output_raw,
                    "winner_freq": freq,
                    "min_consistency": min_consistency,
                    "carry_forward_idx": None,
                    "carry_forward_raw": None,
                    "carry_forward_extracted": None,
                    "accepted": True,
                }
            )
            data_i = data[prob_idx]
            case_outputs = data_i.get("case_output", [])
            case_output_original = data_i.get("case_output_original", [])
            case_is_valid = data_i.get("case_is_valid", [])
            case_inputs = data_i.get("case_input", [])
            case_input_original = data_i.get("case_input_original", [])
            case_sources = data_i.get("case_source", [])
            full_case_generation = data_i.get("full_case_generation", [])
            case_text = data_i.get("case_text", [])
            attack_ideas = data_i.get("attack_ideas", [])

            if ut_idx < len(case_inputs):
                case_inputs[ut_idx] = st.get("input", "")
            if ut_idx < len(case_input_original):
                case_input_original[ut_idx] = st.get("input_output_raw", "")
            if ut_idx < len(case_outputs):
                case_outputs[ut_idx] = ut_output
            if ut_idx < len(case_output_original):
                case_output_original[ut_idx] = ut_output_raw
            if ut_idx < len(case_is_valid):
                case_is_valid[ut_idx] = True
            if ut_idx < len(attack_ideas):
                attack_ideas[ut_idx] = st.get("idea", "Unknown")
            if ut_idx < len(case_sources):
                case_sources[ut_idx] = "idea"
            full_log = (
                f"Idea:\n{st.get('idea','')}\n\nInput:\n{st.get('input','')}\n\nOutput:\n{ut_output_raw}\n"
                f"\n[Refined with new attack idea in round {round_num}]"
            )
            if ut_idx < len(full_case_generation):
                full_case_generation[ut_idx] = full_log
            if ut_idx < len(case_text):
                case_text[ut_idx] = full_log

            if "case_output_samples" in data_i and ut_idx < len(data_i["case_output_samples"]):
                data_i["case_output_samples"][ut_idx] = samples_raw
            if "case_output_samples_extracted" in data_i and ut_idx < len(data_i["case_output_samples_extracted"]):
                data_i["case_output_samples_extracted"][ut_idx] = samples_extracted

            data_i["case_input"] = case_inputs
            data_i["case_output"] = case_outputs
            data_i["case_output_original"] = case_output_original
            data_i["case_input_original"] = case_input_original
            data_i["case_is_valid"] = case_is_valid
            data_i["attack_ideas"] = attack_ideas
            data_i["case_source"] = case_sources
            data_i["full_case_generation"] = full_case_generation
            data_i["case_text"] = case_text
            data_i.setdefault("ut_resample_stats", {"generator": 0, "self_play": 0})
            data_i["ut_resample_stats"]["self_play"] += st["attempts"]
            print(
                f"    - Problem {prob_idx}: Refined UT index {ut_idx} (pass rate {st['selected_rate']:.2f}) "
                f"success at attempt {st['attempts']}",
                flush=True,
            )
            st["success"] = True
            did_refine = True

    for st in ut_states.values():
        if st["success"]:
            continue
        prob_idx = st["prob_idx"]
        ut_idx = st["ut_idx"]
        data_i = data[prob_idx]
        case_is_valid = data_i.get("case_is_valid", [])
        if ut_idx < len(case_is_valid):
            case_is_valid[ut_idx] = False
        data_i["case_is_valid"] = case_is_valid
        if not st.get("input_ok"):
            print(
                f"    - Problem {prob_idx}: Refine UT index {ut_idx} failed to generate input, "
                "blocked for next round.",
                flush=True,
            )
        else:
            print(
                f"    - Problem {prob_idx}: Refine UT index {ut_idx} failed after {max_attempts} attempts, "
                "blocked for next round.",
                flush=True,
            )

    return data, did_refine

# Step 3: execute repaired code and replace bad UTs through self-consistency.
def fix_code_attack_ut(data, runner, tokenizer, args, round_num=None):
    print("\n=== Step 2: Execute Refined Codes + Regenerate Bad UTs ===", flush=True)
    round_key = self_play_round_key(round_num)


    temp_data_for_exec = []
    for data_i in data:
        temp_item = {
            "question": data_i.get("question", ""),
            "generated_code": data_i.get("refined_codes", data_i.get("generated_code", [])),
            "case_input": data_i.get("case_input", []),
            "case_output": data_i.get("case_output", []),
            "test_input": data_i.get("test_input", []),
            "test_output": data_i.get("test_output", []),
            "test_time_limit": data_i.get("test_time_limit", 1),
        }
        temp_data_for_exec.append(temp_item)

    run_all_executions_generate(temp_data_for_exec, args)


    for data_i, temp_item in zip(data, temp_data_for_exec):
        data_i["refined_bool_table"] = temp_item.get("case_bool_table", [])
        data_i["refined_exe_results"] = temp_item.get("case_exe_results", [])


    max_attempts = getattr(args, "ut_regen_max_attempts", 5)
    sc_num = getattr(args, "self_consistency_num", 4)
    

    for idx, data_i in enumerate(data):
        data_i.setdefault("ut_resample_stats", {"generator": 0, "self_play": 0})

    round_label = f"Round {round_num}" if round_num is not None else "Round ?"
    problem_tasks = []
    for idx, data_i in enumerate(data):

        if data_i.get("skip_self_play", False):
            continue
        
        refined_bool = data_i.get("refined_bool_table")
        if refined_bool is None or len(refined_bool) == 0:
            print(f"    - Problem {idx}: No refined bool table.", flush=True)
            continue 

        num_codes = len(refined_bool)
        num_cases = len(refined_bool[0]) if num_codes > 0 else 0
        if num_codes == 0 or num_cases == 0:
            print(f"    - Problem {idx}: Empty refined bool table.", flush=True)
            continue 


        prev_case_is_valid = data_i.get("case_is_valid", [])
        prev_invalid_indices = [
            j for j, valid in enumerate(prev_case_is_valid) if not valid and j < num_cases
        ]


        case_is_valid = [True] * num_cases
        data_i["case_is_valid"] = case_is_valid
        

        bad_ut_indices = []
        for j in range(num_cases):
            if not case_is_valid[j]:
                continue
            all_failed = True
            for c in range(num_codes):
                if j < len(refined_bool[c]) and refined_bool[c][j]:
                    all_failed = False
                    break
            if all_failed:
                bad_ut_indices.append(j)

        if prev_invalid_indices:

            bad_ut_indices = sorted(set(bad_ut_indices).union(prev_invalid_indices))

        if not bad_ut_indices:
            continue 


        problem = data_i.get("question", "")
        attack_ideas = data_i.get("attack_ideas", [])
        case_inputs = data_i.get("case_input", [])
        case_outputs = data_i.get("case_output", [])


        def _ensure_len(lst, length, fill_val):
            while len(lst) < length:
                lst.append(fill_val)

        _ensure_len(case_inputs, num_cases, "")
        _ensure_len(case_outputs, num_cases, "")
        _ensure_len(case_is_valid, num_cases, True)
        case_sources = data_i.get("case_source", [])
        _ensure_len(case_sources, num_cases, "unknown")
        data_i["case_source"] = case_sources

        if "case_input_original" not in data_i:
            data_i["case_input_original"] = []
        if "case_output_original" not in data_i:
            data_i["case_output_original"] = []
        if "full_case_generation" not in data_i:
            data_i["full_case_generation"] = []
        if "case_text" not in data_i:
            data_i["case_text"] = []

        _ensure_len(data_i["case_input_original"], num_cases, "")
        _ensure_len(data_i["case_output_original"], num_cases, "")
        _ensure_len(data_i["full_case_generation"], num_cases, "")
        _ensure_len(data_i["case_text"], num_cases, "")
        _ensure_len(attack_ideas, num_cases, "Unknown")

        round_prefix = f"Round {round_num}: " if round_num is not None else ""


        problem_tasks.append(
            {
                "idx": idx,
                "problem": problem,
                "attack_ideas": attack_ideas,
                "case_inputs": case_inputs,
                "case_outputs": case_outputs,
                "case_is_valid": case_is_valid,
                "case_source": case_sources,
                "bad_ut_indices": bad_ut_indices,
                "round_prefix": round_prefix,
                "round_label": round_label,
            }
        )


    if not problem_tasks:
        print("    - No bad UTs detected, skip regeneration.", flush=True)
        return data

    meta_by_prob = {t["idx"]: t for t in problem_tasks}
    

    input_prompts = []
    input_tasks = []
    for t in problem_tasks:
        prob_idx = t["idx"]
        problem = t["problem"]
        

        all_attack_ideas = data[prob_idx].get("attack_ideas_candidates", [])
        if not all_attack_ideas:

            all_attack_ideas = t["attack_ideas"]
        
        k_code = len(data[prob_idx].get("generated_code", []))
        
        for ut_idx in t["bad_ut_indices"]:

            new_idea = _pick_new_attack_idea(data[prob_idx], all_attack_ideas, ut_idx, fallback="Unknown")
            consume_item_usage(args, prob_idx, "attack_idea", [new_idea], round_key)
            

            input_prompt = get_full_prompt(
                get_ut_input_generation_prompt(problem, new_idea)
            )
            input_prompts.append(input_prompt)
            input_tasks.append((prob_idx, ut_idx, new_idea))
    

    if not input_prompts:
        print("    - No inputs to generate for bad UTs.", flush=True)
        return data
    
    print(f"    - Generating {len(input_prompts)} new inputs for bad UTs (new attack ideas)...", flush=True)
    input_outputs = runner.generate(input_prompts)
    record_direct_usage(
        args,
        [prob_idx for prob_idx, _, _ in input_tasks],
        input_prompts,
        input_outputs,
        "self_play_bad_ut_input_generation",
        round_key,
    )
    

    ut_states = {}
    did_regen = False
    for (prob_idx, ut_idx, new_idea), input_output_raw in zip(input_tasks, input_outputs):
        new_input = extract_ut_input(input_output_raw)
        if not new_input or "We can not extract" in new_input:

            meta = meta_by_prob[prob_idx]
            new_input = meta["case_inputs"][ut_idx] if ut_idx < len(meta["case_inputs"]) else ""
        

        problem = meta_by_prob[prob_idx]["problem"]
        output_prompt = get_full_prompt(
            get_ut_output_generation_prompt(problem, new_input)
        )
        
        ut_states[(prob_idx, ut_idx)] = {
            "prob_idx": prob_idx,
            "ut_idx": ut_idx,
            "idea": new_idea,
            "input": new_input,
            "input_output_raw": input_output_raw,
            "output_prompt": output_prompt,
            "attempts": 0,
            "success": False,
        }

    for attempt in range(max_attempts):

        problem_success_count = {}
        for key, st in ut_states.items():
            if st["success"]:
                prob_idx = st["prob_idx"]
                problem_success_count[prob_idx] = problem_success_count.get(prob_idx, 0) + 1
        

        pending_keys = []
        for k, st in ut_states.items():
            if st["success"]:
                continue
            prob_idx = st["prob_idx"]
            meta = meta_by_prob[prob_idx]
            total_bad_count = len(meta["bad_ut_indices"])
            

            if problem_success_count.get(prob_idx, 0) >= total_bad_count:
                continue
            pending_keys.append(k)
        
        if not pending_keys:
            break


        output_tasks = []
        for key in pending_keys:
            st = ut_states[key]
            st["attempts"] += 1
            output_tasks.append({
                "prob_idx": st["prob_idx"],
                "ut_idx": st["ut_idx"],
                "idea": st["idea"],
                "ut_input": st["input"],
                "input_output_raw": st.get("input_output_raw", ""),
                "output_prompt": st["output_prompt"],
            })

        if not output_tasks:
            continue


        if sc_num > 1:
            expanded_prompts = []
            for task in output_tasks:
                expanded_prompts.extend([task["output_prompt"]] * sc_num)
            all_outputs = runner.generate(expanded_prompts)
            record_direct_usage(
                args,
                [task["prob_idx"] for task in output_tasks for _ in range(sc_num)],
                expanded_prompts,
                all_outputs,
                "self_play_bad_ut_output_generation",
                round_key,
            )

            grouped_outputs = []
            cursor = 0
            for _ in output_tasks:
                grouped_outputs.append(all_outputs[cursor : cursor + sc_num])
                cursor += sc_num
        else:
            raw_outputs = runner.generate([t["output_prompt"] for t in output_tasks])
            record_direct_usage(
                args,
                [task["prob_idx"] for task in output_tasks],
                [t["output_prompt"] for t in output_tasks],
                raw_outputs,
                "self_play_bad_ut_output_generation",
                round_key,
            )
            grouped_outputs = [[o] for o in raw_outputs]


        for task, samples_raw in zip(output_tasks, grouped_outputs):
            if not samples_raw:
                continue

            samples_extracted = [extract_ut_output(o) for o in samples_raw]
            counts = Counter(samples_extracted)
            winner, freq = counts.most_common(1)[0]

            if sc_num > 1:
                min_consistency = max(1, math.ceil(sc_num * 0.75))
                if freq < min_consistency or not winner or "We can not extract" in winner:
                    continue
                winner_idx = samples_extracted.index(winner)
                ut_output_raw = samples_raw[winner_idx]
                ut_output = winner
            else:
                ut_output_raw = samples_raw[0]
                ut_output = winner
                if not ut_output or "We can not extract" in ut_output:
                    continue

            prob_idx = task["prob_idx"]
            ut_idx = task["ut_idx"]
            idea = task["idea"]
            ut_input = task["ut_input"]

            meta = meta_by_prob[prob_idx]
            case_inputs = meta["case_inputs"]
            case_outputs = meta["case_outputs"]
            case_is_valid = meta["case_is_valid"]
            attack_ideas = meta["attack_ideas"]
            case_source = meta["case_source"]


            case_inputs[ut_idx] = ut_input
            case_outputs[ut_idx] = ut_output
            case_is_valid[ut_idx] = True
            attack_ideas[ut_idx] = idea
            if ut_idx < len(case_source):
                case_source[ut_idx] = "idea"

            input_output_raw = task.get("input_output_raw", "")
            data[prob_idx]["case_input_original"][ut_idx] = input_output_raw
            data[prob_idx]["case_output_original"][ut_idx] = ut_output_raw
            full_log = (
                f"Idea:\n{idea}\n\nInput:\n{ut_input}\n\nOutput:\n{ut_output_raw}\n"
                f"\n[Regenerated with new attack idea in round {round_num}]"
            )
            data[prob_idx]["full_case_generation"][ut_idx] = full_log
            data[prob_idx]["case_text"][ut_idx] = full_log

            ut_states[(prob_idx, ut_idx)]["success"] = True

            data[prob_idx]["ut_resample_stats"]["self_play"] += ut_states[(prob_idx, ut_idx)]["attempts"]
            print(
                f"    - {meta['round_prefix']}Problem {prob_idx}: Regenerated UT output at index {ut_idx} (attempt {ut_states[(prob_idx, ut_idx)]['attempts']})",
                flush=True,
            )

            if prob_idx == 0:
                _log_problem_bool_matrix(
                    data, 0, f"{meta['round_label']} After regenerating UT index {ut_idx}"
                )


    for key, st in ut_states.items():
        prob_idx, ut_idx = st["prob_idx"], st["ut_idx"]
        meta = meta_by_prob[prob_idx]
        case_is_valid = meta["case_is_valid"]
        if st["success"]:
            continue
        case_is_valid[ut_idx] = False


    for meta in problem_tasks:
        idx = meta["idx"]
        data[idx]["attack_ideas"] = meta["attack_ideas"]
        data[idx]["case_input"] = meta["case_inputs"]
        data[idx]["case_output"] = meta["case_outputs"]
        data[idx]["case_is_valid"] = meta["case_is_valid"]
        data[idx]["case_source"] = meta["case_source"]

    print("    - Step 2 UT regeneration complete.", flush=True)
    return data


# Optional step: regenerate UTs passed by every code candidate.
def regenerate_all_pass_ut(data, runner, tokenizer, args, round_num=None):
    print("\n=== Step 3.5: Regenerate All-Pass UTs ===", flush=True)
    round_key = self_play_round_key(round_num)

    did_regen = False
    max_attempts = getattr(args, "ut_regen_max_attempts", 5)
    sc_num = getattr(args, "self_consistency_num", 4)


    for data_i in data:
        data_i.setdefault("ut_resample_stats", {"generator": 0, "self_play": 0})

    round_label = f"Round {round_num}" if round_num is not None else "Round ?"
    problem_tasks = []
    for idx, data_i in enumerate(data):
        if data_i.get("skip_self_play", False):
            continue

        bool_table = data_i.get("case_bool_table")
        if bool_table is None or len(bool_table) == 0:
            print(f"    - Problem {idx}: No bool table.", flush=True)
            continue

        num_codes = len(bool_table)
        num_cases = len(bool_table[0]) if num_codes > 0 else 0
        if num_codes == 0 or num_cases == 0:
            print(f"    - Problem {idx}: Empty bool table.", flush=True)
            continue

        case_is_valid = data_i.get("case_is_valid", [True] * num_cases)
        all_pass_indices = []
        valid_count = 0
        for j in range(num_cases):
            if j < len(case_is_valid) and not case_is_valid[j]:
                continue
            valid_count += 1
            all_pass = True
            for c in range(num_codes):
                if j >= len(bool_table[c]) or not bool_table[c][j]:
                    all_pass = False
                    break
            if all_pass:
                all_pass_indices.append(j)

        if not all_pass_indices:
            continue

        if valid_count > 0 and len(all_pass_indices) == valid_count:
            print(f"    - Problem {idx}: all UTs passed by all codes, skip all-pass regeneration.", flush=True)
            continue

        problem = data_i.get("question", "")
        attack_ideas = data_i.get("attack_ideas", [])
        case_inputs = data_i.get("case_input", [])
        case_outputs = data_i.get("case_output", [])

        def _ensure_len(lst, length, fill_val):
            while len(lst) < length:
                lst.append(fill_val)

        _ensure_len(case_inputs, num_cases, "")
        _ensure_len(case_outputs, num_cases, "")
        _ensure_len(case_is_valid, num_cases, True)
        case_sources = data_i.get("case_source", [])
        _ensure_len(case_sources, num_cases, "unknown")
        data_i["case_source"] = case_sources

        if "case_input_original" not in data_i:
            data_i["case_input_original"] = []
        if "case_output_original" not in data_i:
            data_i["case_output_original"] = []
        if "full_case_generation" not in data_i:
            data_i["full_case_generation"] = []
        if "case_text" not in data_i:
            data_i["case_text"] = []

        _ensure_len(data_i["case_input_original"], num_cases, "")
        _ensure_len(data_i["case_output_original"], num_cases, "")
        _ensure_len(data_i["full_case_generation"], num_cases, "")
        _ensure_len(data_i["case_text"], num_cases, "")
        _ensure_len(attack_ideas, num_cases, "Unknown")

        round_prefix = f"Round {round_num}: " if round_num is not None else ""
        problem_tasks.append(
            {
                "idx": idx,
                "problem": problem,
                "attack_ideas": attack_ideas,
                "case_inputs": case_inputs,
                "case_outputs": case_outputs,
                "case_is_valid": case_is_valid,
                "case_source": case_sources,
                "target_ut_indices": all_pass_indices,
                "round_prefix": round_prefix,
                "round_label": round_label,
            }
        )

    if not problem_tasks:
        print("    - No all-pass UTs detected, skip regeneration.", flush=True)
        return data, False

    meta_by_prob = {t["idx"]: t for t in problem_tasks}


    input_prompts = []
    input_tasks = []
    for t in problem_tasks:
        prob_idx = t["idx"]
        problem = t["problem"]
        all_attack_ideas = data[prob_idx].get("attack_ideas_candidates", [])
        if not all_attack_ideas:
            all_attack_ideas = t["attack_ideas"]

        k_code = len(data[prob_idx].get("generated_code", []))
        for ut_idx in t["target_ut_indices"]:
            new_idea = _pick_new_attack_idea(data[prob_idx], all_attack_ideas, ut_idx, fallback="Unknown")
            consume_item_usage(args, prob_idx, "attack_idea", [new_idea], round_key)

            input_prompt = get_full_prompt(get_ut_input_generation_prompt(problem, new_idea))
            input_prompts.append(input_prompt)
            input_tasks.append((prob_idx, ut_idx, new_idea))

    if not input_prompts:
        print("    - No inputs to generate for all-pass UTs.", flush=True)
        return data, False

    print(
        f"    - Generating {len(input_prompts)} new inputs for all-pass UTs (new attack ideas)...",
        flush=True,
    )
    input_outputs = runner.generate(input_prompts)
    record_direct_usage(
        args,
        [prob_idx for prob_idx, _, _ in input_tasks],
        input_prompts,
        input_outputs,
        "self_play_all_pass_ut_input_generation",
        round_key,
    )


    ut_states = {}
    for (prob_idx, ut_idx, new_idea), input_output_raw in zip(input_tasks, input_outputs):
        new_input = extract_ut_input(input_output_raw)
        if not new_input or "We can not extract" in new_input:
            meta = meta_by_prob[prob_idx]
            new_input = meta["case_inputs"][ut_idx] if ut_idx < len(meta["case_inputs"]) else ""

        problem = meta_by_prob[prob_idx]["problem"]
        output_prompt = get_full_prompt(get_ut_output_generation_prompt(problem, new_input))

        ut_states[(prob_idx, ut_idx)] = {
            "prob_idx": prob_idx,
            "ut_idx": ut_idx,
            "idea": new_idea,
            "input": new_input,
            "input_output_raw": input_output_raw,
            "output_prompt": output_prompt,
            "attempts": 0,
            "success": False,
        }

    for attempt in range(max_attempts):
        problem_success_count = {}
        for st in ut_states.values():
            if st["success"]:
                prob_idx = st["prob_idx"]
                problem_success_count[prob_idx] = problem_success_count.get(prob_idx, 0) + 1

        pending_keys = []
        for k, st in ut_states.items():
            if st["success"]:
                continue
            prob_idx = st["prob_idx"]
            meta = meta_by_prob[prob_idx]
            total_target_count = len(meta["target_ut_indices"])
            if problem_success_count.get(prob_idx, 0) >= total_target_count:
                continue
            pending_keys.append(k)

        if not pending_keys:
            break

        output_tasks = []
        for key in pending_keys:
            st = ut_states[key]
            st["attempts"] += 1
            output_tasks.append(
                {
                    "prob_idx": st["prob_idx"],
                    "ut_idx": st["ut_idx"],
                    "idea": st["idea"],
                    "ut_input": st["input"],
                    "input_output_raw": st.get("input_output_raw", ""),
                    "output_prompt": st["output_prompt"],
                }
            )

        if not output_tasks:
            continue

        if sc_num > 1:
            expanded_prompts = []
            for task in output_tasks:
                expanded_prompts.extend([task["output_prompt"]] * sc_num)
            all_outputs = runner.generate(expanded_prompts)
            record_direct_usage(
                args,
                [task["prob_idx"] for task in output_tasks for _ in range(sc_num)],
                expanded_prompts,
                all_outputs,
                "self_play_all_pass_ut_output_generation",
                round_key,
            )

            grouped_outputs = []
            cursor = 0
            for _ in output_tasks:
                grouped_outputs.append(all_outputs[cursor : cursor + sc_num])
                cursor += sc_num
        else:
            raw_outputs = runner.generate([t["output_prompt"] for t in output_tasks])
            record_direct_usage(
                args,
                [task["prob_idx"] for task in output_tasks],
                [t["output_prompt"] for t in output_tasks],
                raw_outputs,
                "self_play_all_pass_ut_output_generation",
                round_key,
            )
            grouped_outputs = [[o] for o in raw_outputs]

        for task, samples_raw in zip(output_tasks, grouped_outputs):
            if not samples_raw:
                continue

            samples_extracted = [extract_ut_output(o) for o in samples_raw]
            counts = Counter(samples_extracted)
            winner, freq = counts.most_common(1)[0]

            if sc_num > 1:
                min_consistency = max(1, math.ceil(sc_num * 0.75))
                if freq < min_consistency or not winner or "We can not extract" in winner:
                    continue
                winner_idx = samples_extracted.index(winner)
                ut_output_raw = samples_raw[winner_idx]
                ut_output = winner
            else:
                ut_output_raw = samples_raw[0]
                ut_output = winner
                if not ut_output or "We can not extract" in ut_output:
                    continue

            prob_idx = task["prob_idx"]
            ut_idx = task["ut_idx"]
            idea = task["idea"]
            ut_input = task["ut_input"]

            meta = meta_by_prob[prob_idx]
            case_inputs = meta["case_inputs"]
            case_outputs = meta["case_outputs"]
            case_is_valid = meta["case_is_valid"]
            attack_ideas = meta["attack_ideas"]
            case_source = meta["case_source"]

            case_inputs[ut_idx] = ut_input
            case_outputs[ut_idx] = ut_output
            case_is_valid[ut_idx] = True
            attack_ideas[ut_idx] = idea
            if ut_idx < len(case_source):
                case_source[ut_idx] = "idea"

            input_output_raw = task.get("input_output_raw", "")
            data[prob_idx]["case_input_original"][ut_idx] = input_output_raw
            data[prob_idx]["case_output_original"][ut_idx] = ut_output_raw
            full_log = (
                f"Idea:\n{idea}\n\nInput:\n{ut_input}\n\nOutput:\n{ut_output_raw}\n"
                f"\n[Regenerated all-pass UT in round {round_num}]"
            )
            data[prob_idx]["full_case_generation"][ut_idx] = full_log
            data[prob_idx]["case_text"][ut_idx] = full_log

            ut_states[(prob_idx, ut_idx)]["success"] = True
            did_regen = True
            data[prob_idx]["ut_resample_stats"]["self_play"] += ut_states[(prob_idx, ut_idx)][
                "attempts"
            ]
            print(
                f"    - {meta['round_prefix']}Problem {prob_idx}: Regenerated all-pass UT index {ut_idx} "
                f"(attempt {ut_states[(prob_idx, ut_idx)]['attempts']})",
                flush=True,
            )
            if prob_idx == 0:
                _log_problem_bool_matrix(
                    data, 0, f"{meta['round_label']} After regenerating all-pass UT index {ut_idx}"
                )

    for key, st in ut_states.items():
        if st["success"]:
            continue
        prob_idx, ut_idx = st["prob_idx"], st["ut_idx"]
        meta = meta_by_prob[prob_idx]
        print(
            f"    - {meta['round_prefix']}Problem {prob_idx}: Failed to regenerate all-pass UT at index {ut_idx}, keeping original.",
            flush=True,
        )

    for meta in problem_tasks:
        idx = meta["idx"]
        data[idx]["attack_ideas"] = meta["attack_ideas"]
        data[idx]["case_input"] = meta["case_inputs"]
        data[idx]["case_output"] = meta["case_outputs"]
        data[idx]["case_is_valid"] = meta["case_is_valid"]
        data[idx]["case_source"] = meta["case_source"]

    print("    - Step 3.5 all-pass UT regeneration complete.", flush=True)
    return data, did_regen


def _propagate_refined_results(data):
    for item in data:
        if "refined_codes" in item:
            item["generated_code"] = item["refined_codes"]
        if "refined_codes_full" in item:
            item["full_code_generation"] = item["refined_codes_full"]
        if "refined_bool_table" in item:
            item["case_bool_table"] = item["refined_bool_table"]
        if "refined_exe_results" in item:
            item["case_exe_results"] = item["refined_exe_results"]
    return data


def _dump_round_data(data, args, outputs_name, round_idx):
    def _format_bool_matrix(mat):
        if mat is None:
            return None
        rows = []
        for row in mat:
            try:
                rows.append("".join("1" if bool(x) else "0" for x in row))
            except Exception:
                rows.append(str(row))
        return rows

    def _add_readable_matrices(item):
        item = item.copy()
        if "case_bool_table" in item:
            item["case_bool_table_rows"] = _format_bool_matrix(item.get("case_bool_table"))
        if "test_bool_table" in item:
            item["test_bool_table_rows"] = _format_bool_matrix(item.get("test_bool_table"))
        if "refined_bool_table" in item:
            item["refined_bool_table_rows"] = _format_bool_matrix(item.get("refined_bool_table"))
        return item

    round_dir = os.path.join("temp_data", args.mode, "self_play_v2_rounds")
    os.makedirs(round_dir, exist_ok=True)
    filename = f"round_{round_idx:02d}_{outputs_name}.json"
    file_path = os.path.join(round_dir, filename)
    readable_data = [_add_readable_matrices(item) for item in data]
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(readable_data, f, indent=2, ensure_ascii=False, default=convert_ndarray)
    print(f"    - Saved round {round_idx} snapshot to {file_path}", flush=True)

def _append_to_results_txt(args, outputs_name: str, text: str):
    if not outputs_name:
        return
    if getattr(args, "is_final_eval", False):
        out_path = f"../CURE_results/results_final_eval/{args.mode}/{outputs_name}_final_eval.txt"
    else:
        out_path = f"../CURE_results/results_optimization_eval/{args.mode}/{outputs_name}.txt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(text.rstrip("\n") + "\n")


def _write_round_metrics(data, args, outputs_name, round_idx):
    if not outputs_name:
        return
    tmp_args = copy.copy(args)
    tmp_outputs = f"{outputs_name}_round{round_idx}"
    try:
        compute_and_log_metrics(data, tmp_outputs, mean_code=0, mean_case=0, args=tmp_args)
    except Exception as e:
        print(f"[WARN] write round {round_idx} metrics failed: {e}", flush=True)

def _format_resample_stats(data, round_label=""):
    lines = []
    lines.append("\n" + "=" * 80)
    if round_label:
        lines.append(f"=== Resample Statistics ({round_label}) ===")
    else:
        lines.append("=== Resample Statistics Summary ===")
    lines.append("=" * 80)
    
    total_ut_generator = 0
    total_ut_selfplay = 0
    total_code_selfplay = 0
    
    lines.append("\nPer-Problem Breakdown:")
    lines.append("-" * 80)
    lines.append(f"{'Problem':<10} {'UT (Gen)':<15} {'UT (Self-Play)':<20} {'Code (Self-Play)':<20}")
    lines.append("-" * 80)
    
    for idx, data_i in enumerate(data):
        ut_stats = data_i.get("ut_resample_stats", {"generator": 0, "self_play": 0})
        code_stats = data_i.get("code_resample_stats", {"self_play": 0})
        
        ut_gen = ut_stats.get("generator", 0)
        ut_sp = ut_stats.get("self_play", 0)
        code_sp = code_stats.get("self_play", 0)
        
        total_ut_generator += ut_gen
        total_ut_selfplay += ut_sp
        total_code_selfplay += code_sp
        
        lines.append(f"{idx:<10} {ut_gen:<15} {ut_sp:<20} {code_sp:<20}")
    
    lines.append("-" * 80)
    lines.append(f"{'Total':<10} {total_ut_generator:<15} {total_ut_selfplay:<20} {total_code_selfplay:<20}")
    lines.append("-" * 80)
    

    num_problems = len(data)
    avg_ut_gen = total_ut_generator / num_problems if num_problems > 0 else 0
    avg_ut_sp = total_ut_selfplay / num_problems if num_problems > 0 else 0
    avg_code_sp = total_code_selfplay / num_problems if num_problems > 0 else 0
    
    lines.append(f"\nAverage per problem:")
    lines.append(f"  - UT resample (Generator):   {avg_ut_gen:.2f}")
    lines.append(f"  - UT resample (Self-Play):   {avg_ut_sp:.2f}")
    lines.append(f"  - Code resample (Self-Play): {avg_code_sp:.2f}")
    lines.append(f"\nTotal resample operations:")
    lines.append(f"  - UT (Generator + Self-Play): {total_ut_generator + total_ut_selfplay}")
    lines.append(f"  - Code (Self-Play):           {total_code_selfplay}")
    lines.append(f"  - Grand Total:                {total_ut_generator + total_ut_selfplay + total_code_selfplay}")
    lines.append("=" * 80 + "\n")
    
    return "\n".join(lines)


def _count_random_uts(data):
    total = 0
    for item in data:
        sources = item.get("case_source", [])
        valids = item.get("case_is_valid", [])
        for i, src in enumerate(sources):
            if src != "random":
                continue
            if valids:
                if i < len(valids) and valids[i]:
                    total += 1
            else:
                total += 1
    return total


def _count_random_uts_per_problem(data):
    stats = []
    for idx, item in enumerate(data):
        cnt = 0
        sources = item.get("case_source", [])
        valids = item.get("case_is_valid", [])
        for i, src in enumerate(sources):
            if src != "random":
                continue
            if valids:
                if i < len(valids) and valids[i]:
                    cnt += 1
            else:
                cnt += 1
        stats.append((idx, cnt))
    return stats


def _regenerate_random_ut_inputs(data, runner, args, round_num):
    total_k_case = int(getattr(args, "k_case", 0))
    if total_k_case <= 0:
        return
    random_candidates_per_problem = total_k_case * 2
    num_problems = len(data)


    prompt_items = []
    for idx in range(num_problems):
        for _ in range(random_candidates_per_problem):
            prompt_list = get_ut_input_random_generation_prompt(
                problem=data[idx]["question"],
                num_cases=1,
            )
            prompt_items.append((get_full_prompt(prompt_list), idx))

    if not prompt_items:
        return

    print(f">>> Regenerating random UT inputs (round {round_num})", flush=True)

    random.shuffle(prompt_items)
    random_prompts = [p for p, _ in prompt_items]
    prompt_to_problem = [idx for _, idx in prompt_items]
    outputs = runner.generate(random_prompts)
    record_direct_usage(
        args,
        prompt_to_problem,
        random_prompts,
        outputs,
        "self_play_random_ut_input_generation",
        self_play_round_key(round_num),
    )


    random_inputs_by_problem = [[] for _ in range(num_problems)]


    for output, idx in zip(outputs, prompt_to_problem):
        candidates = parse_random_case_inputs(output)
        ut_input = candidates[0] if candidates else extract_ut_input(output)
        ut_input = _strip_case_prefix(ut_input)
        if not ut_input:
            continue
        random_inputs_by_problem[idx].append(ut_input)


    from metrics import _select_random_inputs
    placeholder = getattr(args, "random_ut_placeholder", "We can not extract the input in the output. ")
    target_count = int(getattr(args, "k_case", total_k_case))
    for idx in range(num_problems):

        seen = set()
        deduped = []
        for ut_input in random_inputs_by_problem[idx]:
            key = str(ut_input).strip()
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ut_input)
        selected = _select_random_inputs(deduped, target_count, placeholder)
        data[idx]["random_case_input"] = selected
        data[idx]["random_case_input_selected"] = []

        for attempt in range(1, 6):
            data[idx].pop(f"random_case_exe_results_{attempt}", None)
            data[idx].pop(f"random_case_bool_table_rows_{attempt}", None)


# Self-play driver.
def run_self_play_iterations(data, tokenizer, args, outputs_name, shared_runner=None):
    total_rounds = max(int(getattr(args, "self_play_round", 0)), 0)
    if total_rounds == 0:
        print("self_play_round=0; skipping self-play.", flush=True)
        return data

    runner = shared_runner
    own_runner = False
    if runner is None:
        print(">>> Initializing dedicated ModelRunner for self-play", flush=True)
        runner = ModelRunner(args)
        own_runner = True
    else:
        print(">>> Reusing existing ModelRunner for self-play", flush=True)


    remaining_random = _count_random_uts(data)
    per_prob = _count_random_uts_per_problem(data)
    print(f">>> Random UT remaining before self-play: {remaining_random}", flush=True)
    for pid, cnt in per_prob:
        print(f"    - Problem {pid}: random UT = {cnt}", flush=True)
    _write_round_metrics(data, args, outputs_name, round_idx=0)

    print(">>> Dumping round 0 snapshot before self-play", flush=True)
    sync_usage_to_data(args, data)
    _dump_round_data(data, args, outputs_name, round_idx=0)
    for round_idx in range(total_rounds):
        print(f"\n{'=' * 80}", flush=True)
        print(f"=== Self-Play V2 Round {round_idx + 1}/{total_rounds} ===", flush=True)
        print(f"{'=' * 80}\n", flush=True)

        remaining_random = _count_random_uts(data)
        per_prob = _count_random_uts_per_problem(data)
        print(f">>> Random UT remaining at round start: {remaining_random}", flush=True)

        _regenerate_random_ut_inputs(data, runner, args, round_idx + 1)
        for pid, cnt in per_prob:
            print(f"    - Problem {pid}: random UT = {cnt}", flush=True)

        print(">>> Step 0/7: Resampling 0-accuracy codes (before code fix)", flush=True)
        data, did_resample = resample_code(
            data, runner, tokenizer, args, round_num=round_idx + 1
        )


        if did_resample:
            print(">>> Refreshing case executions after resample (case only)", flush=True)
            run_all_executions_generate(data, args)

        print(">>> Step 1/7: Refining low-accuracy UT outputs", flush=True)
        data, did_refine = ut_refine(data, runner, tokenizer, args, round_num=round_idx + 1)
        if did_refine:
            print(">>> Refreshing case executions after UT refine (case only)", flush=True)
            run_all_executions_generate(data, args)

        print(">>> Step 2/7: Running UT attack and code refinement", flush=True)
        data = ut_attack_code(data, runner, tokenizer, args, round_num=round_idx + 1)
        print(">>> Step 3/7: Executing refined codes + UT regeneration", flush=True)
        data = fix_code_attack_ut(data, runner, tokenizer, args, round_num=round_idx + 1)

        print(">>> Step 4/7: Propagating refined results back to data", flush=True)
        data = _propagate_refined_results(data)

        print(">>> Step 5/7: Re-running executions to refresh bool table", flush=True)
        data = run_all_executions_generate(data, args)

        print(">>> Step 6/7: Regenerating all-pass UTs", flush=True)
        data, did_all_pass = regenerate_all_pass_ut(
            data, runner, tokenizer, args, round_num=round_idx + 1
        )
        print(">>> Step 7/7: Re-running executions after all-pass UT regen", flush=True)
        data = run_all_executions(data, args, skip_random_ut=True, compute_new_bon=False)
        

        for data_i in data:
            if data_i.get("skip_self_play", False):
                continue
            bool_table = data_i.get("case_bool_table")
            if bool_table is not None:

                if "bool_table_history" not in data_i:
                    data_i["bool_table_history"] = []

                data_i["bool_table_history"].append([row[:] for row in bool_table])

                if len(data_i["bool_table_history"]) > 3:
                    data_i["bool_table_history"] = data_i["bool_table_history"][-3:]
        


            


            


                    


        

        _compute_new_bon_with_history(data, args, print_results=True)
        _write_round_metrics(data, args, outputs_name, round_idx=round_idx + 1)
        usage_text = format_usage_round_summary(
            args,
            self_play_round_key(round_idx + 1),
            label=f"Self-Play Round {round_idx + 1}",
        )
        if usage_text:
            print(usage_text, flush=True)
            _append_to_results_txt(args, outputs_name, usage_text)
        _log_problem_bool_matrix(
            data,
            0,
            f"Round {round_idx + 1} final bool matrix for Problem 0",
        )
        _record_round_bool_history(data, round_idx + 1)

        print(">>> Dumping round snapshot", flush=True)
        sync_usage_to_data(args, data)
        _dump_round_data(data, args, outputs_name, round_idx + 1)
        _clear_round_transient_fields(data)
        

        resample_stats_text = _format_resample_stats(data, round_label=f"Round {round_idx + 1}")
        print(resample_stats_text, flush=True)
        _append_to_results_txt(args, outputs_name, resample_stats_text)

    if own_runner:
        runner.close()
        print(">>> Self-play ModelRunner closed", flush=True)

    remaining_random = _count_random_uts(data)
    per_prob = _count_random_uts_per_problem(data)
    print(f">>> Random UT remaining after self-play: {remaining_random}", flush=True)
    for pid, cnt in per_prob:
        print(f"    - Problem {pid}: random UT = {cnt}", flush=True)


    convergence_text = []
    convergence_text.append("\n" + "=" * 80)
    convergence_text.append("=== Convergence Summary ===")
    convergence_text.append("=" * 80)
    converged_problems = []
    for idx, data_i in enumerate(data):
        if data_i.get("skip_self_play", False):
            converged_round = data_i.get("converged_round", "Unknown")
            reason = data_i.get("skip_self_play_reason", "Unknown reason")
            converged_problems.append((idx, converged_round, reason))
    
    if converged_problems:
        convergence_text.append(f"Total converged problems: {len(converged_problems)}/{len(data)}")
        for idx, round_num, reason in converged_problems:
            convergence_text.append(f"  - Problem {idx}: Converged in round {round_num} ({reason})")
    else:
        convergence_text.append("No problems converged during self-play.")
    convergence_text.append("=" * 80 + "\n")
    
    convergence_output = "\n".join(convergence_text)
    print(convergence_output, flush=True)
    _append_to_results_txt(args, outputs_name, convergence_output)
    

    final_resample_stats = _format_resample_stats(data, round_label="Final Summary")
    print(final_resample_stats, flush=True)
    _append_to_results_txt(args, outputs_name, final_resample_stats)

    return data
