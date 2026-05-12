# Generation utilities for PlanSearch, unit-test generation, and output parsing.
import random
import math
import numpy as np
from collections import Counter, defaultdict
from termcolor import cprint
from prompts import get_scaling_prompt 
import re
from itertools import combinations
from prompts import (
    get_stage2_prompt,
)
from UT_config import (
    get_ut_idea_prompt,
    get_ut_input_generation_prompt,
    get_ut_input_random_generation_prompt,
    get_ut_output_generation_prompt,
    get_full_prompt
)
from execution import run_scripts_with_chunk
from usage_tracking import (
    consume_item_usage,
    initial_round_key,
    record_direct_usage,
    reserve_item_usage,
)


# Output parsers and cleanup helpers.
def extract_code(full_output: str) -> str:
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = matches[-1].strip()
    else:
        code_output = "We can not extract the code in the output. "
    return code_output


def vote_ut_output_by_codes(
    failing_indices,
    output_prompt_idx_to_case_idx,
    parsed_inputs,
    data,
    args
):

    if not failing_indices:
        return {}

    print(f"    - [Code-Voting] Found {len(failing_indices)} low-consistency UTs; using code voting...", flush=True)
    voting_codes = []
    voting_inputs = []
    voting_time_limits = []
    

    ut_task_ranges = []
    current_task_idx = 0
    

    for i in failing_indices:
        idx = output_prompt_idx_to_case_idx[i]
        ut_input = parsed_inputs[i]

        codes = [c for c in data[idx]["generated_code"] if "We can not extract" not in c]
        

        if not codes:
            ut_task_ranges.append((current_task_idx, current_task_idx))
            continue
            

        time_limit = data[idx].get("test_time_limit", 1)
        

        for code in codes:
            voting_codes.append(code)
            voting_inputs.append(ut_input)
            voting_time_limits.append(time_limit)
        

        ut_task_ranges.append((current_task_idx, current_task_idx + len(codes)))
        current_task_idx += len(codes)
    
    voted_results_map = {}
    raw_voting_results_map = {}


    if voting_codes:
        print(f"    - [Code-Voting] Running {len(voting_codes)} code execution jobs...", flush=True)


        voting_results = run_scripts_with_chunk(
            voting_codes,
            voting_inputs,
            voting_time_limits,
            args.num_chunks,
            args.exe_verbose
        )
        

        for idx_in_failing, (start, end) in enumerate(ut_task_ranges):

            if start == end:
                continue
                
            ut_idx = failing_indices[idx_in_failing]
            results = voting_results[start:end]
            

            raw_voting_results_map[ut_idx] = results
            

            valid_results = [r.strip() for r in results if r and "Error" not in r and "Timeout" not in r]
            

            if valid_results:

                counts = Counter(valid_results)

                majority_output = counts.most_common(1)[0][0]
                

                synthetic_output = f"**Test Output:**\n```\n{majority_output}\n```\n\nExplanation:\n[Derived from code-based voting majority]"
                voted_results_map[ut_idx] = synthetic_output
                print(f"      > Problem {output_prompt_idx_to_case_idx[ut_idx]}: code voting succeeded", flush=True)
            else:

                print(f"      > Problem {output_prompt_idx_to_case_idx[ut_idx]}: code voting failed (no valid execution output)", flush=True)
                
    return voted_results_map, raw_voting_results_map


def modify(c):
    c = c.replace("plaintext\n", "")
    c = c.replace("\\n", "\n")
    if not c.endswith("\n"):
        c += "\n"
    return c

def _cut_at_first_marker(content: str, markers: tuple[str, ...]) -> str:
    lower = content.lower()
    cut_positions = [lower.find(m) for m in markers if m in lower]
    return content[: min(cut_positions)] if cut_positions else content

def _clean_ut_output_text(content: str) -> str:
    if not content:
        return content

    lower = content.lower()
    marker_positions = [
        lower.find(m)
        for m in (
            "```",
            "**test output:**",
            "**test output**",
            "**explanation:**",
            "**test input:**",
            "let's think step by step",
        )
        if m in lower
    ]
    other_cut = min(marker_positions) if marker_positions else None
    blank_cut = content.rfind("\n\n")
    if blank_cut < 0:
        blank_cut = None
    if other_cut is None and blank_cut is None:
        cut_pos = None
    elif other_cut is None:
        cut_pos = blank_cut
    elif blank_cut is None:
        cut_pos = other_cut
    else:
        cut_pos = min(other_cut, blank_cut)
    if cut_pos is not None:
        content = content[:cut_pos]

    content = re.sub(
        r"\s*(\*\*?Let's think step by step\.?\*\*?)\s*$",
        "",
        content,
        flags=re.IGNORECASE,
    )
    return content.strip()

def _looks_like_prompt_artifact(content: str) -> bool:
    if not content:
        return False
    lower = content.lower()
    if "let's think step by step" in lower:
        return True
    if "**explanation" in lower or "explanation:" in lower:
        return True
    if "test input" in lower or "test output" in lower:
        return True
    return False

def _clean_ut_input_text(content: str) -> str:
    if not content:
        return content
    content = _cut_at_first_marker(
        content,
        (
            "\n\n",
            "```",
            "**test output:**",
            "**explanation:**",
            "explanation:",
            "in this test case",
            "in this input",
            "in this case",
            "let's think step by step",
        ),
    )
    content = re.sub(
        r"\s*(\*\*?Let's think step by step\.?\*\*?)\s*$",
        "",
        content,
        flags=re.IGNORECASE,
    )
    return content.strip()


def extract_test_cases(full_output):

    pattern_input_backticks = r"\*\*Test Input:\*\*\s*```(.*?)```"
    pattern_output_backticks = r"\*\*Test Output:\*\*\s*```(.*?)```"
    matches_input = re.findall(
        pattern_input_backticks, full_output, re.DOTALL | re.IGNORECASE
    )
    matches_output = re.findall(
        pattern_output_backticks, full_output, re.DOTALL | re.IGNORECASE
    )


    if matches_input:
        raw_input = _clean_ut_input_text(matches_input[-1].lstrip("\n"))
        test_input = [modify(raw_input)]
    else:

        pattern_input_plain = r"\*\*Test Input:\*\*\s*([\s\S]*?)(?=\*\*Test Output:\*\*)"
        matches_input_plain = re.findall(
            pattern_input_plain, full_output, re.DOTALL | re.IGNORECASE
        )
        if matches_input_plain:
            raw_input = _clean_ut_input_text(matches_input_plain[-1].strip())
            test_input = [modify(raw_input)]
        else:
            test_input = []

    if matches_output:
        raw_output = _clean_ut_output_text(matches_output[-1].lstrip("\n"))
        test_output = [modify(raw_output)]
    else:
        pattern_output_plain = r"\*\*Test Output:\*\*\s*([\s\S]*?)(?=\*\*Explanation:|\*\*Test Input:|$)"
        matches_output_plain = re.findall(
            pattern_output_plain, full_output, re.DOTALL | re.IGNORECASE
        )
        if matches_output_plain:
            raw_output = _clean_ut_output_text(matches_output_plain[-1].strip())
            test_output = [modify(raw_output)]
        else:
            test_output = []


    example_text = []
    input_matches = list(re.finditer(r"\*\*Test Input:\*\*", full_output, re.IGNORECASE))
    if input_matches:
        example_text = [full_output[input_matches[-1].start():]]

    if example_text == [] or test_input == [] or test_output == []:
        return [], [], []

    return test_input, test_output, example_text


def get_token_lengths(strings, tokenizer):

    valid_strings = [s for s in strings if s is not None]
    return [len(tokenizer.encode(s, add_special_tokens=False)) for s in valid_strings]


def extract_ut_idea(full_output: str) -> list[str]:
    lines = full_output.strip().split('\n')
    ideas = []
    for line in lines:

        if re.match(r'^\d+[\.\)]', line.strip()):
             ideas.append(line.strip())
    

    if not ideas:
         ideas = [full_output.strip()]
    return [modify(idea) for idea in ideas]

def extract_ut_input(full_output: str) -> str:


    pattern = r"\*\*Test [Ii]nput:\*\*\s*(.*?)(?:\s*\*\*|$)"
    matches = re.findall(pattern, full_output, re.DOTALL | re.IGNORECASE)
    if matches:
        content = matches[-1]

        content = re.sub(r"^```(?:[a-zA-Z]*\n)?", "", content)
        content = re.sub(r"```$", "", content).strip()
        content = _clean_ut_input_text(content)
        if _looks_like_prompt_artifact(content):
            return "We can not extract the input in the output. "
        if content:
            return modify(content)

        return "\n"
    

    matches = re.findall(r"```(?:[a-zA-Z]*\n)?(.*?)```", full_output, re.DOTALL)
    if matches:
        content = _clean_ut_input_text(matches[0].strip())
        if _looks_like_prompt_artifact(content):
            return "We can not extract the input in the output. "
        if content:
            return modify(content)
    return "We can not extract the input in the output. "

def extract_ut_output(full_output: str) -> str:


    pattern = r"\*\*Test [Oo]utput:\*\*\s*(.*?)(?:\s*\*\*Explanation:\*\*|$)"
    matches = re.findall(pattern, full_output, re.DOTALL | re.IGNORECASE)
    if matches:
        content = matches[-1]

        content = re.sub(r"^```(?:[a-zA-Z]*\n)?", "", content)
        content = re.sub(r"```$", "", content).strip()
        content = _clean_ut_output_text(content)
        if _looks_like_prompt_artifact(content):
            return "We can not extract the output in the output. "
        if content:
            return modify(content)

        return "\n"
    

    matches = re.findall(r"```(?:[a-zA-Z]*\n)?(.*?)```", full_output, re.DOTALL)
    if matches:
        content = _clean_ut_output_text(matches[-1].strip())
        if _looks_like_prompt_artifact(content):
            return "We can not extract the output in the output. "
        if content:
            return modify(content)
    
    return "We can not extract the output in the output. "


def _strip_case_prefix(text: str) -> str:
    if text is None:
        return ""
    raw = str(text)
    if raw.lstrip().lower().startswith("case|"):
        raw = re.sub(r"^\s*case\|\s*", "", raw, flags=re.IGNORECASE)
    return raw


def parse_random_case_inputs(full_output: str) -> list[str]:
    if not full_output:
        return []
    pattern = r"CASE\|\s*```(.*?)```"
    matches = re.findall(pattern, full_output, re.DOTALL | re.IGNORECASE)
    if matches:
        cleaned = []
        for m in matches:
            m = m.strip()
            if not m:
                continue
            cleaned.append(_strip_case_prefix(modify(m)))
        return cleaned


    raw = str(full_output)
    case_indices = [m.start() for m in re.finditer(r"CASE\|", raw, flags=re.IGNORECASE)]
    if case_indices:
        cleaned = []
        for i, start in enumerate(case_indices):
            seg_start = start + len("CASE|")
            seg_end = case_indices[i + 1] if i + 1 < len(case_indices) else len(raw)
            segment = raw[seg_start:seg_end].strip()
            if not segment:
                continue
            segment = segment.replace("`", "")
            segment = segment.strip()
            if not segment:
                continue
            cleaned.append(_strip_case_prefix(modify(segment)))
        if cleaned:
            return cleaned


    fallback_input = extract_ut_input(full_output)
    if not fallback_input:
        return []
    fallback_input = str(fallback_input).replace("`", "")
    return [_strip_case_prefix(fallback_input)]


# PlanSearch observation parsing and prompt-path expansion.
def parse_observations(text: str, max_obs: int | None = None):


    lines = [ln.strip() for ln in text.splitlines()]
    
    obs = []
    


    pattern_new = r'^\[\s*Observation\s+\d+\s*\]:\s*(.+)$'
    pattern_old = r'^\d+[\.\)]\s+(.+)$'

    for ln in lines:
        if not ln:
            continue

        content = None
        

        m_new = re.match(pattern_new, ln, re.IGNORECASE)
        if m_new:
            content = m_new.group(1).strip()
        else:

            m_old = re.match(pattern_old, ln)
            if m_old:
                content = m_old.group(1).strip()
        
        if content:
            obs.append(content)
            if max_obs is not None and len(obs) >= max_obs:
                break
                
    return obs

def parse_observations_stage2(text: str):


    lines = [ln.strip() for ln in text.splitlines()]
    
    obs = []
    


    pattern_new = r'^\[\s*Observation\s+\d+\s*\]:\s*(.+)$'
    pattern_old = r'^\d+[\.\)]\s+(.+)$'

    for ln in lines:
        if not ln:
            continue

        content = None
        

        m_new = re.match(pattern_new, ln, re.IGNORECASE)
        if m_new:
            content = m_new.group(1).strip()
        else:

            m_old = re.match(pattern_old, ln)
            if m_old:
                content = m_old.group(1).strip()
        
        if content:
            obs.append(content)
                
    return obs


def format_observations(obs_list):
    if not obs_list:
        return "(no explicit observations provided; reason from the problem directly.)"
    return "\n".join(f"- {o}" for o in obs_list)


def build_observation_subsets(obs_list, max_subset_size=2, include_empty=True):
    subsets = []
    if include_empty:
        subsets.append([])

    n = len(obs_list)
    for size in range(1, min(max_subset_size, n) + 1):
        for combo in combinations(obs_list, size):
            subsets.append(list(combo))
    return subsets


def log_llm_interaction(stage_name, prompts, outputs, args, meta_info_list=None,):

    if not args.verbose_logging:
        return


    sep_thick = "=" * 80
    sep_thin = "-" * 80
    
    print(f"\n{sep_thick}", flush=True)
    print(f" [LOGGER] {stage_name} INTERACTION START | Batch Size: {len(prompts)}", flush=True)
    print(f"{sep_thick}\n", flush=True)

    for i, (p, o) in enumerate(zip(prompts, outputs)):

        p_idx = "Unknown"
        if meta_info_list and i < len(meta_info_list):

            item = meta_info_list[i]
            if isinstance(item, dict) and 'idx' in item:
                p_idx = item['idx']
            elif isinstance(item, int):
                p_idx = item
            else:
                p_idx = str(item)

        print(f"  >>> Item {i+1}/{len(prompts)} | Problem ID: {p_idx}", flush=True)
        print(f"{sep_thin}", flush=True)
        print(f"【PROMPT】:\n{p.strip()}\n", flush=True)
        print(f"{sep_thin}", flush=True)
        print(f"【OUTPUT】:\n{o.strip()}\n", flush=True)
        print(f"{sep_thick}\n", flush=True)

    print(f" [LOGGER] {stage_name} INTERACTION END", flush=True)
    print(f"{sep_thick}\n", flush=True)

# PlanSearch code generation.
def run_generation_plansearch(
    data,
    case_generation_prompts,
    case_index,
    runner,
    tokenizer,
    args,
):
    
    num = len(data)
    usage_round_key = initial_round_key()

    print(
        "Running PlanSearch staged code generation: "
        "Stage1(first-order observations) -> Stage2(second-order observations) -> "
        "[Shuffle & Select] → Code",
        flush=True,
    )


    codes_per_problem = [[] for _ in range(num)]


    plansearch_records_per_problem = [[] for _ in range(num)]

    failed_stage1_rounds = [0] * num
    problem_halted = [False] * num 
    global_round = 0

    while True:

        if global_round >= args.max_global_rounds:
            print(
                f"\n[PlanSearch] Reached max_global_rounds={args.max_global_rounds}; stopping PlanSearch",
                flush=True,
            )
            break


        active_indices = [
            i
            for i in range(num)
            if (not problem_halted[i]) and (len(codes_per_problem[i]) < args.k_code)
        ]

        if not active_indices:
            print("\n[PlanSearch] All problems have met k_code or were stopped; finishing", flush=True)
            break

        global_round += 1
        print(
            f"\n[PlanSearch] Global Round {global_round}, active problems: {len(active_indices)}",
            flush=True,
        )


        stage1_prompts = [
            data[idx]["code_generation_prompts"]["stage1"] for idx in active_indices
        ]
        print(f"  [Stage1] Generating first-order observations for {len(stage1_prompts)} problems...", flush=True)
        
        stage1_outputs = runner.generate(stage1_prompts)
        

        log_llm_interaction(f"Stage 1 (Round {global_round})", stage1_prompts, stage1_outputs, args, active_indices)


        first_order_obs = {}
        stage1_items_by_call = []
        
        for idx, out in zip(active_indices, stage1_outputs):
            data[idx]["stage1_observations_raw"] = out
            obs_list = parse_observations(out, max_obs=args.max_obs)
            data[idx]["stage1_observations_list"] = obs_list
            stage1_items_by_call.append(obs_list)
            
            if not obs_list:
                failed_stage1_rounds[idx] += 1
                print(
                    f"    [WARN] Problem {idx}: no first-order observations parsed this round (fail count: {failed_stage1_rounds[idx]})",
                    flush=True,
                )
            else:
                first_order_obs[idx] = obs_list
        reserve_item_usage(
            args,
            active_indices,
            stage1_prompts,
            stage1_outputs,
            "stage1_observation",
            usage_round_key,
            stage1_items_by_call,
        )

        if not first_order_obs:
            print("  [Stage1] No valid observations this round; retrying in the next round", flush=True)
            continue
        
        if args.ablation not in {"only_stage1", "only_stage2"}:
            raise ValueError(f"Unsupported ablation: {args.ablation}. Use 'only_stage1' or 'only_stage2'.")

        if args.ablation == "only_stage1":
            print("  [Stage4] Preparing code generation from first-order observation subsets...", flush=True)
            
            all_stage4_tasks = []
            for idx, obs_list in first_order_obs.items():
                c1_subsets = build_observation_subsets(obs_list, max_subset_size=2)
                
                problem_template = data[idx]["code_generation_prompts"]["stage2_template"]
                
                for c1 in c1_subsets:
                    first_subset_text = format_observations(c1)
                    
                    record = {
                        "global_round": global_round,
                        "first_obs_for_branch": c1,
                        "second_obs_for_leaf": [],
                        "first_obs_for_leaf_str": first_subset_text,
                        "second_obs_for_leaf_str": "",
                        "plan_type": "only_stage1",
                        "plan_text": first_subset_text,
                        "codes": [],
                    }
                    
                    p = get_stage2_prompt(
                        problem_template,
                        first_subset_text,
                        args.system_prompts_stage4_ablation_only_stage,
                    )
                    all_stage4_tasks.append({
                        "idx": idx,
                        "prompt": p,
                        "record": record
                    })

            tasks_by_problem = {idx: [] for idx in active_indices}
            for task in all_stage4_tasks:
                if task["idx"] in tasks_by_problem:
                    tasks_by_problem[task["idx"]].append(task)
            
            final_stage4_prompts = []
            final_stage4_meta = []
            final_stage4_records = []
            
            for idx in active_indices:
                current_count = len(codes_per_problem[idx])
                needed = args.k_code - current_count
                
                if needed <= 0:
                    continue
                
                available_tasks = tasks_by_problem[idx]
                random.shuffle(available_tasks)
                selected_tasks = available_tasks[:needed]
                
                for t in selected_tasks:
                    final_stage4_prompts.append(t["prompt"])
                    final_stage4_meta.append(t["idx"])
                    final_stage4_records.append(t["record"])
                    

                    if t["record"] is not None:
                        plansearch_records_per_problem[t["idx"]].append(t["record"])
                
                print(f"    - Problem {idx}: using first-order subsets, {len(available_tasks)} ideas available, {len(selected_tasks)} selected for code generation", flush=True)
            
            if not final_stage4_prompts:
                print("  [Info] No Stage 4 tasks to run; skipping", flush=True)
                continue
        
        elif args.ablation == "only_stage2":
            print("  [Stage4] Preparing code generation from second-order observation subsets...", flush=True)
            
            stage2_tasks = []
            for idx, obs_list in first_order_obs.items():
                c1_subsets = build_observation_subsets(obs_list, max_subset_size=2)
                for c1 in c1_subsets:
                    stage2_tasks.append({
                        "idx": idx,
                        "first_obs_for_branch": c1,
                    })

            if not stage2_tasks:
                print("  [WARN] Could not build first-order subsets; skipping this round", flush=True)
                continue

            print(f"  [Stage2 temporary] Generating second-order observations for {len(stage2_tasks)} paths...", flush=True)

            stage2_prompts = []
            for task in stage2_tasks:
                idx = task["idx"]
                problem_template = data[idx]["code_generation_prompts"]["stage2_template"]
                first_subset_text = format_observations(task["first_obs_for_branch"])
                task["first_subset_text"] = first_subset_text
                
                p = get_stage2_prompt(
                    problem_template,
                    first_subset_text,
                    args.system_prompts_stage2,
                )
                stage2_prompts.append(p)

            stage2_outputs = runner.generate(stage2_prompts)
            stage2_items_by_call = []
            
            log_llm_interaction(f"Stage 2 temporary (Round {global_round})", stage2_prompts, stage2_outputs, args, stage2_tasks)

            all_stage4_tasks = []
            
            for task, out in zip(stage2_tasks, stage2_outputs):
                idx = task["idx"]
                second_order_obs_list = parse_observations_stage2(out)
                stage2_items_by_call.append(second_order_obs_list)


                if data[idx].get("stage2_observations") is None:
                    data[idx]["stage2_observations"] = ""
                if data[idx].get("stage2_observations_list") is None:
                    data[idx]["stage2_observations_list"] = []
                

                data[idx]["stage2_observations_list"].extend(second_order_obs_list)

                for obs in second_order_obs_list:
                    data[idx]["stage2_observations"] += f"- {obs}\n"
                
                if not second_order_obs_list:
                    continue
                

                
                problem_template = data[idx]["code_generation_prompts"]["stage2_template"]
                
                for second_order_obs in second_order_obs_list:
                    second_obs_text = second_order_obs
                    record = {
                        "global_round": global_round,
                        "first_obs_for_branch": task["first_obs_for_branch"],
                        "second_obs_for_leaf": second_order_obs,
                        "first_obs_for_leaf_str": "",
                        "second_obs_for_leaf_str": second_obs_text,
                        "plan_type": "only_stage2",
                        "plan_text": second_obs_text,
                        "codes": [],
                    }
                    
                    p = get_stage2_prompt(
                        problem_template,
                        second_obs_text,
                        args.system_prompts_stage4_ablation_only_stage,
                    )
                    all_stage4_tasks.append({
                        "idx": idx,
                        "prompt": p,
                        "record": record
                    })
            reserve_item_usage(
                args,
                [task["idx"] for task in stage2_tasks],
                stage2_prompts,
                stage2_outputs,
                "stage2_observation",
                usage_round_key,
                stage2_items_by_call,
            )

            tasks_by_problem = {idx: [] for idx in active_indices}
            for task in all_stage4_tasks:
                if task["idx"] in tasks_by_problem:
                    tasks_by_problem[task["idx"]].append(task)
            
            final_stage4_prompts = []
            final_stage4_meta = []
            final_stage4_records = []
            
            for idx in active_indices:
                current_count = len(codes_per_problem[idx])
                needed = args.k_code - current_count
                
                if needed <= 0:
                    continue
                
                available_tasks = tasks_by_problem[idx]
                random.shuffle(available_tasks)
                selected_tasks = available_tasks[:needed]
                
                for t in selected_tasks:
                    final_stage4_prompts.append(t["prompt"])
                    final_stage4_meta.append(t["idx"])
                    final_stage4_records.append(t["record"])
                    
                    if t["record"] is not None:
                        plansearch_records_per_problem[t["idx"]].append(t["record"])
                
                print(f"    - Problem {idx}: using second-order subsets, {len(available_tasks)} ideas available, {len(selected_tasks)} selected for code generation", flush=True)
            
            if not final_stage4_prompts:
                print("  [Info] No Stage 4 tasks to run; skipping", flush=True)
                continue
            

        for idx, plan_record in zip(final_stage4_meta, final_stage4_records):
            if not isinstance(plan_record, dict):
                continue
            consume_item_usage(
                args,
                idx,
                "stage1_observation",
                plan_record.get("first_obs_for_branch", []),
                usage_round_key,
            )
            consume_item_usage(
                args,
                idx,
                "stage2_observation",
                plan_record.get("second_obs_for_leaf", []),
                usage_round_key,
            )

        stage4_outputs = runner.generate(final_stage4_prompts)
        record_direct_usage(
            args,
            final_stage4_meta,
            final_stage4_prompts,
            stage4_outputs,
            "stage4_code_generation",
            usage_round_key,
        )

        log_llm_interaction(f"Stage 4 - Code Gen (Round {global_round})", final_stage4_prompts, stage4_outputs, args, final_stage4_meta)

        

        for full_output, idx, plan_record in zip(stage4_outputs, final_stage4_meta, final_stage4_records):
            code_text = extract_code(full_output)
            codes_per_problem[idx].append(code_text)
            if plan_record is not None:
                plan_record["codes"].append(code_text)
        
            


    all_code_full_outputs = []
    
    for idx in range(num):
        data[idx]["num_codes_generated"] = len(codes_per_problem[idx])
        data[idx]["plansearch_plans_and_observations"] = plansearch_records_per_problem[idx]
        
        data[idx]["experiment_config"] = {
            "max_obs": args.max_obs,
            "max_global_rounds": args.max_global_rounds,
            "prompt_role_mode": args.prompt_role_mode,
            "ablation": args.ablation,
        }

        status = "OK" if len(codes_per_problem[idx]) >= args.k_code else "WARN"
        print(f"  [{status}] Problem {idx} final code count: {len(codes_per_problem[idx])}", flush=True)

        for code_str in codes_per_problem[idx]:
            all_code_full_outputs.append(code_str) 
            data[idx]["full_code_generation"].append(code_str)
            data[idx]["generated_code"].append(code_str)

    code_generation_result = all_code_full_outputs

    print(f"\n✓ PlanSearch finished: generated {len(code_generation_result)} code candidates.", flush=True)

    code_response_length = get_token_lengths(code_generation_result, tokenizer)
    mean_code = sum(code_response_length) / len(code_response_length) if code_response_length else 0

    case_generation_result = []
    mean_case = 0

    if args.eval_pass_at_k_only:
        print("\n[PlanSearch] eval_pass_at_k_only=True; skipping unit-test generation", flush=True)
    elif args.eval_bon:
        print("\n[PlanSearch] Generating unit tests for all problems (controlled by k_case)", flush=True)
        data, case_generation_result, mean_case = generate_unit_tests_for_dataset(
            data,
            case_generation_prompts,
            case_index,
            runner,
            tokenizer,
            args,
        )

    return data, code_generation_result, case_generation_result, mean_code, mean_case


# Original one-shot generation path.
def run_generation_original(
    data,
    code_generation_prompts,
    code_index,
    case_generation_prompts,
    case_index,
    runner,
    tokenizer,
    args,
):
    print("Starting inference (original unified mode)...", flush=True)
    usage_round_key = initial_round_key()

    all_prompts = code_generation_prompts + case_generation_prompts
    all_problem_indices = code_index + case_index
    N = len(all_prompts)
    indices = list(range(N))
    shuffled_idx = indices[:]
    random.shuffle(shuffled_idx)
    shuffled_prompts = [all_prompts[i] for i in shuffled_idx]

    shuffled_outputs = runner.generate(shuffled_prompts)
    record_direct_usage(
        args,
        [all_problem_indices[i] for i in shuffled_idx],
        shuffled_prompts,
        shuffled_outputs,
        "original_code_and_ut",
        usage_round_key,
    )
    restored_outputs = [None] * N
    for out, idx2 in zip(shuffled_outputs, shuffled_idx):
        restored_outputs[idx2] = out

    code_generation_result = restored_outputs[: len(code_generation_prompts)]
    case_generation_result = restored_outputs[len(code_generation_prompts):]
    print(
        f"✓ Generated code candidates: {len(code_generation_result)}, generated tests: {len(case_generation_result)}",
        flush=True,
    )


    code_response_length = get_token_lengths(code_generation_result, tokenizer)
    case_response_length = (
        get_token_lengths(case_generation_result, tokenizer)
        if len(case_generation_result) > 0
        else []
    )
    mean_code = sum(code_response_length) / len(code_response_length)
    mean_case = sum(case_response_length) / len(case_response_length) if case_response_length else 0


    i = 0
    for full_output in code_generation_result:
        code_output = extract_code(full_output)
        index_i = code_index[i]
        data[index_i]["full_code_generation"].append(full_output)
        data[index_i]["generated_code"].append(code_output)
        i += 1


    i = 0
    for full_output in case_generation_result:
        test_input, test_output, example_text = extract_test_cases(full_output)
        index_i = case_index[i]
        data[index_i]["full_case_generation"].append(full_output)
        data[index_i]["case_input"] += test_input
        data[index_i]["case_output"] += test_output
        data[index_i]["case_text"] += example_text
        i += 1

    return data, code_generation_result, case_generation_result, mean_code, mean_case

# Unit-test generation for BoN and idea-attack modes.
def generate_unit_tests_for_dataset(
    data,
    case_generation_prompts,
    case_index,
    runner,
    tokenizer,
    args,
):
    usage_round_key = initial_round_key()


    if args.use_idea_attack_ut == False:
        if args.eval_pass_at_k_only:
            case_generation_result = []
            print("\n=== pass@k-only mode: skipping unit-test generation ===", flush=True)
            mean_case = 0
            return data, case_generation_result, mean_case
        print("\n=== Generating unit tests ===", flush=True)
        all_case_prompts = case_generation_prompts
        N_case = len(all_case_prompts)
        

        indices_case = list(range(N_case))
        shuffled_idx_case = indices_case[:]
        random.shuffle(shuffled_idx_case)
        shuffled_case_prompts = [all_case_prompts[i] for i in shuffled_idx_case]


        shuffled_case_outputs = runner.generate(shuffled_case_prompts)
        record_direct_usage(
            args,
            [case_index[i] for i in shuffled_idx_case],
            shuffled_case_prompts,
            shuffled_case_outputs,
            "ut_generation_one_shot",
            usage_round_key,
        )
        print("First prompt: ", shuffled_case_prompts[0])
        print("First output: ", shuffled_case_outputs[0])


        restored_case_outputs = [None] * N_case
        for out, idx2 in zip(shuffled_case_outputs, shuffled_idx_case):
            restored_case_outputs[idx2] = out
        case_generation_result = restored_case_outputs
        print("✓ Test generation complete", flush=True)


        i = 0
        for full_output in case_generation_result:

            test_input, test_output, example_text = extract_test_cases(full_output)
            index_i = case_index[i]
            

            data[index_i]["full_case_generation"].append(full_output)

            data[index_i]["case_input"] += test_input
            data[index_i]["case_output"] += test_output
            data[index_i]["case_text"] += example_text
            

            data[index_i]["case_is_valid"] += [True] * len(test_input)
            
            i += 1


        case_response_length = get_token_lengths(case_generation_result, tokenizer)
        mean_case = (
            sum(case_response_length) / len(case_response_length)
            if case_response_length
            else 0
        )
    elif args.use_idea_attack_ut:
        print("\n=== Generating unit tests (Idea Attack Mode) ===", flush=True)
        total_k_case = getattr(args, "k_case", 0)
        if total_k_case <= 0:
            print("  [WARN] k_case=0; no UT generation needed in Idea Attack mode", flush=True)
            case_generation_result = []
            mean_case = 0
            return data, case_generation_result, mean_case

        idea_target = total_k_case // 2
        random_target = total_k_case - idea_target
        idea_candidates_per_problem = idea_target * 2 if idea_target > 0 else 0
        random_candidates_per_problem = random_target * 2 if random_target > 0 else 0
        num_problems = len(data)

        print(
            f"    - [Idea Attack] Total UTs: {total_k_case}; idea-based={idea_target}, random={random_target}",
            flush=True,
        )
        print(
            f"    - [Candidates] Idea candidates={idea_candidates_per_problem}/problem, random candidates={random_candidates_per_problem}/problem",
            flush=True,
        )

        def build_case_index(per_problem_count: int) -> list[int]:
            if per_problem_count <= 0:
                return []
            indices = []
            for idx in range(num_problems):
                indices.extend([idx] * per_problem_count)
            return indices

        def commit_records(records: list[dict]) -> list[str]:
            outputs = []
            for rec in records:
                idx = rec["idx"]
                outputs.append(rec["raw_output"])
                data[idx]["full_case_generation"].append(rec["full_log"])
                data[idx]["case_output_original"].append(rec["raw_output"])
                data[idx]["case_input"].append(rec["input"])
                data[idx]["case_output"].append(rec["output_text"])
                data[idx]["case_text"].append(rec["full_log"])
                data[idx]["case_is_valid"].append(rec["is_valid"])
                data[idx]["case_input_original"].append(rec["input_log"])

                src = rec.get("source_label", "unknown").lower()
                tag = "random" if "random" in src else "idea"
                data[idx].setdefault("case_source", [])
                data[idx]["case_source"].append(tag)
                if "case_output_samples" not in data[idx]:
                    data[idx]["case_output_samples"] = []
                if "case_output_samples_extracted" not in data[idx]:
                    data[idx]["case_output_samples_extracted"] = []
                data[idx]["case_output_samples"].append(rec["samples_raw"])
                data[idx]["case_output_samples_extracted"].append(rec["samples_extracted"])
            return outputs

        def process_ut_records(
            parsed_ideas,
            parsed_inputs,
            input_prompt_idx_to_case_idx,
            input_prompt_raw_outputs,
            source_label: str,
            target_per_problem: int,
        ) -> list[str]:
            if not parsed_inputs or target_per_problem <= 0:
                return []

            print(f"--- Step 3: Generating UT Outputs ({source_label}) ---", flush=True)
            output_prompts = []
            output_prompt_idx_to_case_idx = []
            for ut_input, idx in zip(parsed_inputs, input_prompt_idx_to_case_idx):
                data_i = data[idx].copy()
                prompt_list = get_ut_output_generation_prompt(
                    problem=data_i["question"],
                    ut_input=ut_input,
                )
                prompt_str = get_full_prompt(prompt_list)
                output_prompts.append(prompt_str)
                output_prompt_idx_to_case_idx.append(idx)

            count_same = [0] * args.self_consistency_num
            consistency_scores = []
            fallback_outputs = []
            max_resample_attempts = 0
            if args.self_consistency_num > 1:
                print(
                    f"    - [Self-Consistency] Samples: {args.self_consistency_num}; generating and voting...",
                    flush=True,
                )
                

                ut_states = {}
                for i in range(len(output_prompts)):
                    ut_states[i] = {
                        "prompt": output_prompts[i],
                        "prob_idx": output_prompt_idx_to_case_idx[i],
                        "idea": parsed_ideas[i],
                        "input": parsed_inputs[i],
                        "input_raw": input_prompt_raw_outputs[i],
                        "attempts": 0,
                        "success": False,
                        "final_output": None,
                        "samples_raw": [],
                        "samples_extracted": [],
                        "consistency": 0,
                        "fallback": None,
                    }
                
                max_resample_attempts = 5
                min_consistency_threshold = max(1, math.ceil(args.self_consistency_num * 0.75))
                

                for idx in set(input_prompt_idx_to_case_idx):
                    data[idx].setdefault("ut_resample_stats", {"generator": 0, "self_play": 0})
                
                for attempt in range(max_resample_attempts):

                    problem_success_count = {}
                    for i, st in ut_states.items():
                        if st["success"]:
                            prob_idx = st["prob_idx"]
                            problem_success_count[prob_idx] = problem_success_count.get(prob_idx, 0) + 1
                    

                    pending_indices = []
                    for i, st in ut_states.items():
                        if st["success"]:
                            continue
                        prob_idx = st["prob_idx"]
                        if problem_success_count.get(prob_idx, 0) >= target_per_problem:

                            continue
                        pending_indices.append(i)
                    
                    if not pending_indices:
                        print(f"    - [Resample] Round {attempt + 1}: all problems have enough target UTs", flush=True)
                        break
                    
                    print(f"    - [Resample] Round {attempt + 1}/{max_resample_attempts}: trying {len(pending_indices)} low-consistency UTs...", flush=True)
                    

                    expanded_prompts = []
                    for i in pending_indices:
                        expanded_prompts.extend([ut_states[i]["prompt"]] * args.self_consistency_num)
                        ut_states[i]["attempts"] += 1
                    

                    batch_outputs = runner.generate(expanded_prompts)
                    record_direct_usage(
                        args,
                        [ut_states[i]["prob_idx"] for i in pending_indices for _ in range(args.self_consistency_num)],
                        expanded_prompts,
                        batch_outputs,
                        "ut_output_generation",
                        usage_round_key,
                    )
                    

                    for idx_in_batch, i in enumerate(pending_indices):
                        start_idx = idx_in_batch * args.self_consistency_num
                        end_idx = start_idx + args.self_consistency_num
                        samples_raw = batch_outputs[start_idx:end_idx]
                        samples_extracted = [extract_ut_output(s) for s in samples_raw]
                        

                        ut_states[i]["samples_raw"] = samples_raw
                        ut_states[i]["samples_extracted"] = samples_extracted
                        

                        valid_samples = [
                            s for s in samples_extracted if s.strip() and "We can not extract" not in s
                        ]
                        if valid_samples:
                            counts = Counter(valid_samples)
                            winner_extracted, unique_count = counts.most_common(1)[0]
                        else:
                            counts = Counter(samples_extracted)
                            winner_extracted, unique_count = counts.most_common(1)[0]
                        

                        try:
                            winner_idx = samples_extracted.index(winner_extracted)
                            winner_raw = samples_raw[winner_idx]
                        except ValueError:
                            winner_raw = samples_raw[0]
                        
                        ut_states[i]["consistency"] = unique_count
                        ut_states[i]["fallback"] = winner_raw
                        

                        if (
                            unique_count >= min_consistency_threshold
                            and winner_extracted
                            and "We can not extract" not in winner_extracted
                        ):
                            ut_states[i]["success"] = True
                            ut_states[i]["final_output"] = winner_raw
                            count_same[unique_count - 1] += 1

                            prob_idx = ut_states[i]["prob_idx"]
                            data[prob_idx]["ut_resample_stats"]["generator"] += ut_states[i]["attempts"]
                        else:

                            count_same[unique_count - 1] += 1
                    

                    newly_succeeded = sum(1 for i in pending_indices if ut_states[i]["success"])
                    print(f"      > Round {attempt + 1}: {newly_succeeded}/{len(pending_indices)} UTs passed validation", flush=True)
                

                final_outputs = []
                all_samples_raw = []
                all_samples_extracted = []
                
                for i in range(len(output_prompts)):
                    st = ut_states[i]
                    all_samples_raw.append(st["samples_raw"])
                    all_samples_extracted.append(st["samples_extracted"])
                    consistency_scores.append(st["consistency"])
                    fallback_outputs.append(st["fallback"])
                    
                    if st["success"]:
                        final_outputs.append(st["final_output"])
                    else:

                        final_outputs.append(None)
                        prob_idx = st["prob_idx"]


                print("count_same=", count_same)
                print("Fully consistent count:", count_same[-1])
                print("Fully consistent ratio:", count_same[-1] / len(output_prompts))
                print("Fully inconsistent count:", count_same[0])
                print("Fully inconsistent ratio:", count_same[0] / len(output_prompts))
            else:
                final_outputs = runner.generate(output_prompts)
                record_direct_usage(
                    args,
                    output_prompt_idx_to_case_idx,
                    output_prompts,
                    final_outputs,
                    "ut_output_generation",
                    usage_round_key,
                )
                all_samples_raw = [[o] for o in final_outputs]
                all_samples_extracted = [[extract_ut_output(o)] for o in final_outputs]
                consistency_scores = [1] * len(final_outputs)
                fallback_outputs = list(final_outputs)

            records = []
            records_by_problem = defaultdict(list)
            for i, (ut_input, ut_output_raw) in enumerate(zip(parsed_inputs, final_outputs)):
                idx = output_prompt_idx_to_case_idx[i]
                idea_text = parsed_ideas[i]
                input_log = input_prompt_raw_outputs[i]
                fallback_raw = fallback_outputs[i] if i < len(fallback_outputs) else None
                samples_raw = all_samples_raw[i]
                samples_extracted = all_samples_extracted[i]

                invalid_input = (not ut_input) or ("We can not extract" in str(ut_input))
                if invalid_input:
                    ut_output = None
                    ut_output_raw = None
                    full_log = (
                        f"Idea:\n{idea_text}\n\nInput:\n{ut_input}\n\n"
                        f"Output: [Blocked due to invalid input parsing]"
                    )
                    is_valid = False
                elif ut_output_raw is not None:
                    ut_output = extract_ut_output(ut_output_raw)
                    full_log = (
                        f"Idea:\n{idea_text}\n\nInput:\n{ut_input}\n\nOutput:\n{ut_output_raw}"
                    )
                    is_valid = True
                else:
                    ut_output = None
                    if max_resample_attempts > 0:
                        blocked_reason = (
                            f"Blocked due to low consistency after {max_resample_attempts} attempts"
                        )
                    else:
                        blocked_reason = "Blocked because no valid output was produced in single-pass generation"
                    full_log = (
                        f"Idea:\n{idea_text}\n\nInput:\n{ut_input}\n\n"
                        f"Output: [{blocked_reason}]"
                    )
                    is_valid = False
                record = {
                    "idx": idx,
                    "idea": idea_text,
                    "input": ut_input,
                    "input_log": input_log,
                    "raw_output": ut_output_raw,
                    "output_text": ut_output,
                    "full_log": full_log,
                    "is_valid": is_valid,
                    "fallback_raw": fallback_raw,
                    "consistency": consistency_scores[i] if i < len(consistency_scores) else 0,
                    "samples_raw": samples_raw,
                    "samples_extracted": samples_extracted,
                    "source_label": source_label,
                }
                records.append(record)
                records_by_problem[idx].append(record)


            for idx, record_list in records_by_problem.items():
                valid_count = sum(1 for rec in record_list if rec["is_valid"])
                if valid_count < target_per_problem:
                    if max_resample_attempts > 0:
                        retry_note = f"after {max_resample_attempts} resample rounds"
                    else:
                        retry_note = "self-consistency resample disabled"
                    print(
                        f"    - [WARN] Problem {idx}: valid UTs only {valid_count}/{target_per_problem} ({retry_note})",
                        flush=True,
                    )

            selected_records = []
            per_problem_selected = defaultdict(int)
            selected_ids = set()
            for rec in records:
                idx = rec["idx"]
                if per_problem_selected[idx] >= target_per_problem:
                    continue
                if rec["is_valid"]:
                    selected_records.append(rec)
                    per_problem_selected[idx] += 1
                    selected_ids.add(id(rec))

            for rec in records:
                idx = rec["idx"]
                if per_problem_selected[idx] >= target_per_problem:
                    continue
                if id(rec) in selected_ids:
                    continue
                selected_records.append(rec)
                per_problem_selected[idx] += 1
                selected_ids.add(id(rec))

            return commit_records(selected_records)

        def generate_idea_cases() -> list[str]:
            if idea_candidates_per_problem <= 0:
                return []
            case_index_for_generation = build_case_index(idea_candidates_per_problem)
            if not case_index_for_generation:
                return []

            print("--- Step 1: Generating Attack Ideas ---", flush=True)
            idea_prompts = []
            prompt_idx_to_case_idx = []
            ideas_count_per_prompt = []
            problem_usage_count = {}

            for idx in case_index_for_generation:
                data_i = data[idx].copy()
                mode = args.ablation

                if mode == "only_stage2" and data_i.get("stage2_observations_list"):
                    obs_list = data_i["stage2_observations_list"]
                    total_ideas = args.num_ideas
                    num_obs = len(obs_list)
                    if num_obs > 0:
                        if total_ideas < num_obs:
                            current_usage = problem_usage_count.get(idx, 0)
                            problem_usage_count[idx] = current_usage + 1
                            ideas_per_obs_map = {}
                            for k in range(total_ideas):
                                obs_idx = (current_usage * total_ideas + k) % num_obs
                                ideas_per_obs_map[obs_idx] = ideas_per_obs_map.get(obs_idx, 0) + 1
                        else:
                            base_ideas = total_ideas // num_obs
                            remainder = total_ideas % num_obs
                            ideas_per_obs_map = {
                                obs_idx: base_ideas + (1 if obs_idx < remainder else 0)
                                for obs_idx in range(num_obs)
                            }


                        for j, obs in enumerate(obs_list):
                            ideas_for_this_obs = ideas_per_obs_map.get(j, 0)
                            if ideas_for_this_obs <= 0:
                                continue
                            prompt_list = get_ut_idea_prompt(
                                mode=mode,
                                num_ideas=ideas_for_this_obs,
                                problem=data_i["question"],
                                input_mode=obs,
                            )
                            prompt_str = get_full_prompt(prompt_list)
                            idea_prompts.append(prompt_str)
                            prompt_idx_to_case_idx.append(idx)
                            ideas_count_per_prompt.append(ideas_for_this_obs)
                    else:
                        input_mode = data_i.get("stage2_observations", "")
                        prompt_list = get_ut_idea_prompt(
                            mode=mode,
                            num_ideas=args.num_ideas,
                            problem=data_i["question"],
                            input_mode=input_mode,
                        )
                        prompt_str = get_full_prompt(prompt_list)
                        idea_prompts.append(prompt_str)
                        prompt_idx_to_case_idx.append(idx)
                        ideas_count_per_prompt.append(args.num_ideas)
                else:
                    if mode == "only_stage2":
                        input_mode = data_i.get("stage2_observations", "")
                    elif mode == "only_stage1":
                        input_mode = data_i.get("stage1_observations", "")
                    else:
                        input_mode = ""

                    prompt_list = get_ut_idea_prompt(
                        mode=mode,
                        num_ideas=args.num_ideas,
                        problem=data_i["question"],
                        input_mode=input_mode,
                    )
                    prompt_str = get_full_prompt(prompt_list)
                    idea_prompts.append(prompt_str)
                    prompt_idx_to_case_idx.append(idx)
                    ideas_count_per_prompt.append(args.num_ideas)

            for prompt_str, idx in zip(idea_prompts, prompt_idx_to_case_idx):
                data[idx].setdefault("idea_generation_prompts", []).append(prompt_str)

            idea_outputs = runner.generate(idea_prompts)
            parsed_idea_candidates = []
            used_ideas_by_problem = defaultdict(list)
            input_prompts = []
            parsed_ideas = []
            input_prompt_idx_to_case_idx = []
            input_prompt_raw_outputs = []

            for i, output in enumerate(idea_outputs):
                idx = prompt_idx_to_case_idx[i]
                expected_num = ideas_count_per_prompt[i]
                all_ideas = extract_ut_idea(output)
                current_ideas = all_ideas[:expected_num]
                parsed_idea_candidates.append(all_ideas)
                data[idx].setdefault("idea_generation_outputs", []).append(output)
                data[idx].setdefault("attack_ideas", []).extend(current_ideas)
                used_ideas_by_problem[idx].extend(current_ideas)


                data_i = data[idx].copy()

                for idea in current_ideas:
                    parsed_ideas.append(idea)
                    prompt_list = get_ut_input_generation_prompt(
                        problem=data_i["question"],
                        attact_idea=idea,
                    )
                    prompt_str = get_full_prompt(prompt_list)
                    input_prompts.append(prompt_str)
                    input_prompt_idx_to_case_idx.append(idx)
            reserve_item_usage(
                args,
                prompt_idx_to_case_idx,
                idea_prompts,
                idea_outputs,
                "attack_idea",
                usage_round_key,
                parsed_idea_candidates,
            )
            for idx, used_ideas in used_ideas_by_problem.items():
                consume_item_usage(
                    args,
                    idx,
                    "attack_idea",
                    used_ideas,
                    usage_round_key,
                )

            if not input_prompts:
                return []

            print("--- Step 2: Generating UT Inputs ---", flush=True)
            input_outputs = runner.generate(input_prompts)
            record_direct_usage(
                args,
                input_prompt_idx_to_case_idx,
                input_prompts,
                input_outputs,
                "ut_input_generation",
                usage_round_key,
            )
            parsed_inputs = []

            for output, idx in zip(input_outputs, input_prompt_idx_to_case_idx):
                ut_input = extract_ut_input(output)
                parsed_inputs.append(ut_input)
                input_prompt_raw_outputs.append(output)


            return process_ut_records(
                parsed_ideas,
                parsed_inputs,
                input_prompt_idx_to_case_idx,
                input_prompt_raw_outputs,
                "Idea Attack",
                idea_target,
            )

        def generate_random_cases() -> list[str]:
            if random_target <= 0:
                return []

            case_index_for_generation = build_case_index(random_candidates_per_problem)
            if not case_index_for_generation:
                return []

            print("--- Random UT: Generating Inputs ---", flush=True)
            placeholder = getattr(
                args, "random_ut_placeholder", "We can not extract the input in the output. "
            )
            random_prompts = []
            random_prompt_idx_to_case_idx = []
            for idx in case_index_for_generation:
                prompt_list = get_ut_input_random_generation_prompt(
                    problem=data[idx]["question"],
                    num_cases=1,
                )
                prompt_str = get_full_prompt(prompt_list)
                random_prompts.append(prompt_str)
                random_prompt_idx_to_case_idx.append(idx)

            random_outputs = runner.generate(random_prompts)
            record_direct_usage(
                args,
                random_prompt_idx_to_case_idx,
                random_prompts,
                random_outputs,
                "random_ut_input_generation",
                usage_round_key,
            )
            parsed_inputs = []
            parsed_ideas = []
            input_prompt_raw_outputs = []
            input_prompt_idx_to_case_idx = []

            for output, idx in zip(random_outputs, random_prompt_idx_to_case_idx):
                candidates = parse_random_case_inputs(output)
                ut_input = candidates[0] if candidates else extract_ut_input(output)
                ut_input = _strip_case_prefix(ut_input)
                parsed_inputs.append(ut_input)
                parsed_ideas.append("[Random Range Sampling]")
                input_prompt_raw_outputs.append(output)
                input_prompt_idx_to_case_idx.append(idx)


            for idx in range(num_problems):
                data[idx]["random_case_input"] = []
            total_k_case = int(getattr(args, "k_case", 0))
            target_count = total_k_case if total_k_case > 0 else int(getattr(args, "random_ut_batch", 16))
            bon_candidates = (total_k_case * 2) if total_k_case > 0 else (target_count * 2)
            if bon_candidates > 0:
                bon_prompts = []
                bon_prompt_idx_to_case_idx = []
                for idx in range(num_problems):
                    for _ in range(bon_candidates):
                        prompt_list = get_ut_input_random_generation_prompt(
                            problem=data[idx]["question"],
                            num_cases=1,
                        )
                        prompt_str = get_full_prompt(prompt_list)
                        bon_prompts.append(prompt_str)
                        bon_prompt_idx_to_case_idx.append(idx)
                bon_outputs = runner.generate(bon_prompts)
                record_direct_usage(
                    args,
                    bon_prompt_idx_to_case_idx,
                    bon_prompts,
                    bon_outputs,
                    "random_bon_input_generation",
                    usage_round_key,
                )
                random_inputs_by_problem = [[] for _ in range(num_problems)]
                for output, idx in zip(bon_outputs, bon_prompt_idx_to_case_idx):
                    candidates = parse_random_case_inputs(output)
                    ut_input = candidates[0] if candidates else extract_ut_input(output)
                    ut_input = _strip_case_prefix(ut_input)
                    if not ut_input:
                        continue
                    key = str(ut_input).strip()
                    if not key or key == placeholder:
                        continue
                    random_inputs_by_problem[idx].append(ut_input)

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
                    selected = deduped[:target_count]
                    while len(selected) < target_count:
                        selected.append(placeholder)
                    data[idx]["random_case_input"] = selected
            return process_ut_records(
                parsed_ideas,
                parsed_inputs,
                input_prompt_idx_to_case_idx,
                input_prompt_raw_outputs,
                "Random Range",
                random_target,
            )

        case_generation_result = []
        if idea_target > 0 and idea_candidates_per_problem > 0:
            case_generation_result.extend(generate_idea_cases())
        else:
            print("    - [Idea Attack] Idea-based UT count is 0; skipping", flush=True)

        if random_target > 0:
            case_generation_result.extend(generate_random_cases())
        else:
            print("    - [Idea Attack] Random UT count is 0; skipping", flush=True)

        print("✓ Test generation complete (Idea Attack Mode)", flush=True)
        case_response_length = get_token_lengths(case_generation_result, tokenizer)
        mean_case = (
            sum(case_response_length) / len(case_response_length)
            if case_response_length
            else 0
        )

    return data, case_generation_result, mean_case


# Public generation entrypoint used by main_self_play_v3.py.
def run_generation_pipeline(
    data,
    code_generation_prompts,
    code_index,
    case_generation_prompts,
    case_index,
    runner,
    tokenizer,
    args,
):
    if args.use_multi_stage_generation:

        return run_generation_plansearch(
            data,
            case_generation_prompts,
            case_index,
            runner,
            tokenizer,
            args,
        )
    else:
        return run_generation_original(
            data,
            code_generation_prompts,
            code_index,
            case_generation_prompts,
            case_index,
            runner,
            tokenizer,
            args,
        )
