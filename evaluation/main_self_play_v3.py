def convert_ndarray(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)

import os
import ast
import json
import argparse

import numpy as np
from transformers import AutoTokenizer
from termcolor import cprint

from prompts import build_prompts_for_dataset
from inference import ModelRunner
from generator_v3 import run_generation_pipeline
from execution import run_all_executions
from metrics import compute_and_log_metrics
from self_play_v3 import run_self_play_iterations
from usage_tracking import (
    format_usage_final_summary,
    format_usage_round_summary,
    init_usage_tracker,
    initial_round_key,
    sync_usage_to_data,
)

import evaluation_config


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def str2bool(x):
    return x.lower() in ("1", "true", "yes")
# Main evaluation entrypoint: load data, generate candidates, execute, and score.
import argparse
import ast


# Runtime configuration.
def parse_args():
    parser = argparse.ArgumentParser()
    ec = evaluation_config


    parser.add_argument(
        "--generation_mode",
        type=str,
        default=getattr(ec, "generation_mode", "original"),
        help='code generation mode: "original", "plansearch", "original_resample"',
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default=getattr(ec, "eval_mode", "oneshot"),
        help='evaluation mode: "oneshot", "passatk", "bon"',
    )


    parser.add_argument("--pretrained_model", type=str, default=ec.pretrained_model)
    parser.add_argument("--dataset", type=str, default=ec.dataset)
    parser.add_argument("--use_api", type=str2bool, default=ec.use_api)
    parser.add_argument("--k_code", type=int, default=ec.k_code)
    parser.add_argument("--k_case", type=int, default=ec.k_case)
    parser.add_argument("--max_model_len", type=int, default=ec.max_model_len)
    parser.add_argument("--max_generation_token", type=int, default=ec.max_generation_token)
    parser.add_argument("--temp", type=float, default=ec.temp)
    parser.add_argument("--num_chunks", type=int, default=ec.num_chunks)
    parser.add_argument("--no_example", type=str2bool, default=ec.no_example)
    parser.add_argument("--max_test", type=int, default=ec.max_test)
    parser.add_argument("--trust_remote_code", type=str2bool, default=ec.trust_remote_code)
    parser.add_argument("--exe_verbose", type=str2bool, default=ec.exe_verbose)
    parser.add_argument("--is_final_eval", type=str2bool, default=ec.is_final_eval)
    parser.add_argument("--verbose_logging", type=str2bool, default=ec.verbose_logging)


    parser.add_argument(
        "--pass_at_k_list",
        type=ast.literal_eval,
        default=getattr(ec, "pass_at_k_list", []),
    )
    parser.add_argument(
        "--scale_tuple_list",
        type=ast.literal_eval,
        default=getattr(ec, "scale_tuple_list", []),
    )
    parser.add_argument(
        "--use_random_ut_cluster",
        type=str2bool,
        default=getattr(ec, "use_random_ut_cluster", False),
    )
    parser.add_argument(
        "--random_ut_total",
        type=int,
        default=getattr(ec, "random_ut_total", 0),
    )
    parser.add_argument(
        "--random_ut_batch",
        type=int,
        default=getattr(ec, "random_ut_batch", 16),
    )
    parser.add_argument(
        "--random_ut_max_attempts",
        type=int,
        default=getattr(ec, "random_ut_max_attempts", 5),
    )
    parser.add_argument(
        "--random_ut_min_top_count",
        type=int,
        default=getattr(ec, "random_ut_min_top_count", 2),
    )
    parser.add_argument(
        "--random_ut_cluster_max_diff",
        type=int,
        default=getattr(ec, "random_ut_cluster_max_diff", 0),
    )
    parser.add_argument(
        "--random_ut_placeholder",
        type=str,
        default=getattr(ec, "random_ut_placeholder", "We can not extract the input in the output. "),
    )

    parser.add_argument("--single_eval", type=str2bool, default=getattr(ec, "single_eval", False))
    parser.add_argument("--eval_pass_at_k_only", type=str2bool, default=getattr(ec, "eval_pass_at_k_only", False))
    parser.add_argument("--eval_bon", type=str2bool, default=getattr(ec, "eval_bon", False))


    parser.add_argument("--api_model_name", type=str, default=ec.api_model_name)
    parser.add_argument("--api_key", type=str, default=ec.api_key)
    parser.add_argument("--base_url", type=str, default=ec.base_url)
    parser.add_argument("--api_temperature", type=float, default=ec.api_temperature)
    parser.add_argument("--max_workers", type=int, default=ec.max_workers)
    parser.add_argument("--use_openai_batch_api", type=str2bool, default=ec.use_openai_batch_api)
    parser.add_argument("--max_tokens", type=int, default=ec.max_tokens)
    parser.add_argument("--rpm_limit", type=int, default=ec.rpm_limit)


    parser.add_argument("--gpu_groups", type=ast.literal_eval, default=ec.gpu_groups)
    parser.add_argument("--mode", type=str, default=ec.mode)

    parser.add_argument("--prompt_role_mode", type=int, default=getattr(ec, "prompt_role_mode", 0))
    parser.add_argument("--use_multi_stage_generation", type=str2bool, default=getattr(ec, "use_multi_stage_generation", False))
    parser.add_argument("--use_original_resample", type=str2bool, default=getattr(ec, "use_original_resample", False))
    parser.add_argument("--max_obs", type=int, default=ec.max_obs)
    parser.add_argument("--max_global_rounds", type=int, default=getattr(ec, "max_global_rounds", 50))
    parser.add_argument("--ablation", type=str, default=getattr(ec, "ablation", "only_stage1"))
    parser.add_argument("--use_all_second_order_obs", type=str2bool, default=getattr(ec, "use_all_second_order_obs", False))
    parser.add_argument("--use_idea_attack_ut", type=str2bool, default=getattr(ec, "use_idea_attack_ut", False))

    parser.add_argument("--num_ideas", type=int, default=getattr(ec, "num_ideas", 1))
    parser.add_argument("--self_consistency_num", type=int, default=getattr(ec, "self_consistency_num", 1))
    parser.add_argument("--self_play_round", type=int, default=getattr(ec, "self_play_round", 1))
    parser.add_argument("--is_empty", type=str2bool, default=getattr(ec, "is_empty", True))
    parser.add_argument("--ut_vote_by_code", type=str2bool, default=getattr(ec, "ut_vote_by_code", False))
    parser.add_argument("--use_self_play", type=str2bool, default=getattr(ec, "use_self_play", True))

    parser.add_argument("--ut_accuracy_target", type=float, default=getattr(ec, "ut_accuracy_target", 0.5))
    parser.add_argument("--ut_regen_max_attempts", type=int, default=getattr(ec, "ut_regen_max_attempts", 3))
    parser.add_argument("--skip_attack_when_all_pass", type=str2bool, default=getattr(ec, "skip_attack_when_all_pass", True))

    parser.add_argument(
        "--resume_round00",
        type=str,
        default=None,
        help="Path to round_00_results_*.json to skip generation and resume self-play.",
    )
    parser.add_argument(
        "--resume_round",
        type=str,
        default=None,
        help="Path to round_XX_results_*.json to resume from specific round (e.g., round_02).",
    )
    parser.add_argument(
        "--start_round",
        type=int,
        default=None,
        help="Round number to start from when using --resume_round (auto-detected if not specified).",
    )

    args = parser.parse_args()


    if args.generation_mode == "plansearch":
        args.use_multi_stage_generation = True
        args.use_original_resample = False
    elif args.generation_mode == "original_resample":
        args.use_multi_stage_generation = False
        args.use_original_resample = True
    elif args.generation_mode == "original":
        args.use_multi_stage_generation = False
        args.use_original_resample = False
    else:
        raise ValueError(f"Unknown generation_mode: {args.generation_mode}")


    if args.eval_mode == "oneshot":
        args.single_eval = True
        args.eval_pass_at_k_only = False
        args.eval_bon = False
        args.k_case = 0
        args.scale_tuple_list = []
        args.pass_at_k_list = []

    elif args.eval_mode == "passatk":
        args.single_eval = False
        args.eval_pass_at_k_only = True
        args.eval_bon = False
        args.k_case = 0
        args.scale_tuple_list = []

    elif args.eval_mode == "bon":
        args.single_eval = False
        args.eval_pass_at_k_only = False
        args.eval_bon = True
        if args.k_case <= 0:
            cprint("[WARNING] eval_mode='bon' but k_case <= 0, so no unit tests will be generated.", "red")

    else:
        raise ValueError(f"Unknown eval_mode: {args.eval_mode}")
    
    try:

        prompt_data = ec.PROMPT_REGISTRY[args.prompt_role_mode]
        

        args.system_prompts_stage1 = prompt_data["stage1"]
        args.system_prompts_stage2 = prompt_data["stage2"]
        args.system_prompts = prompt_data["original"]
        args.system_case_prompts = prompt_data["case"]
        args.special_requirements = prompt_data["special_requirements"]
        args.system_prompts_stage4_ablation_only_stage = prompt_data["stage4"][args.ablation]
        
        cprint(f"[INFO] Loaded Prompts for Mode {args.prompt_role_mode}", "green")
        
    except KeyError as e:
        raise ValueError(f"Unsupported prompt_role_mode: {args.prompt_role_mode}. Check PROMPT_REGISTRY.") from e
    


    cprint(f"\n[CONFIG SUMMARY] (All Arguments)", "yellow")
    

    args_dict = vars(args)
    

    print(f"{'ARGUMENT':<35} : {'VALUE'}")
    print("-" * 80)
    for key, value in sorted(args_dict.items()):


        print(f"{key:<35} : {value}")
    print("-" * 80 + "\n", flush=True)

    return args


def _append_to_results_txt(args, outputs_name: str, text: str):
    if not outputs_name or not text:
        return
    if getattr(args, "is_final_eval", False):
        out_path = f"../CURE_results/results_final_eval/{args.mode}/{outputs_name}_final_eval.txt"
    else:
        out_path = f"../CURE_results/results_optimization_eval/{args.mode}/{outputs_name}.txt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(text.rstrip("\n") + "\n")

# End-to-end evaluation flow.
def main():
    args = parse_args()


    if args.resume_round:
        round_path = os.path.abspath(args.resume_round)
        print(f"[RESUME] Loading round data: {round_path}", flush=True)
        with open(round_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        data = loaded["data"] if isinstance(loaded, dict) and "data" in loaded else loaded
        if not isinstance(data, list):
            raise ValueError("resume_round JSON must be a list or dict with key 'data'.")
        

        if args.start_round is None:
            import re
            match = re.search(r'round_(\d+)_', os.path.basename(round_path))
            if match:
                args.start_round = int(match.group(1))
                print(f"✓ Auto-detected start round: {args.start_round}", flush=True)
            else:
                raise ValueError("Could not auto-detect round number. Please specify --start_round.")
        
        print(f"✓ Loaded round_{args.start_round:02d} data: {len(data)} tasks", flush=True)
        print(f"✓ Will resume from round {args.start_round + 1}", flush=True)


        model_basename = os.path.basename(args.pretrained_model.rstrip("/"))
        model_basename_api = os.path.basename(args.api_model_name.rstrip("/"))
        if not args.use_api:
            outputs_name = "results_eval_" + model_basename + "_" + args.dataset
        else:
            outputs_name = "results_eval_" + model_basename_api + "_" + args.dataset
        print(f"[RESUME] outputs_name = {outputs_name}", flush=True)


        if not args.use_api:
            tokenizer = AutoTokenizer.from_pretrained(
                args.pretrained_model,
                trust_remote_code=args.trust_remote_code,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                "/data/chenhui/models/Qwen_Qwen2.5-7B-Instruct",
                trust_remote_code=True,
            )
        init_usage_tracker(args, tokenizer, data)


        os.makedirs(
            os.path.dirname(f"./temp_data/{args.mode}/outputs_" + outputs_name + ".json"),
            exist_ok=True,
        )


        runner = None
        if getattr(args, "use_self_play", True) and args.self_play_round > args.start_round:
            print(f"[RESUME] Entering Self_Play from round {args.start_round + 1}...", flush=True)
            runner = ModelRunner(args)

            remaining_rounds = args.self_play_round - args.start_round
            print(f"[RESUME] Remaining rounds to execute: {remaining_rounds}", flush=True)
            original_rounds = args.self_play_round
            args.self_play_round = remaining_rounds
            data = run_self_play_iterations(data, tokenizer, args, outputs_name, shared_runner=runner)
            args.self_play_round = original_rounds
        else:
            print(f"[RESUME] No remaining self-play rounds (start={args.start_round}, total={args.self_play_round})", flush=True)


        data = compute_and_log_metrics(data, outputs_name, 0, 0, args)


        sync_usage_to_data(args, data)
        with open(
            f"./temp_data/{args.mode}/outputs_" + outputs_name + ".json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=convert_ndarray)
        usage_final_text = format_usage_final_summary(args, label="Final")
        if usage_final_text:
            print(usage_final_text, flush=True)
            _append_to_results_txt(args, outputs_name, usage_final_text)

        if runner is not None:
            runner.close()
        cprint(f"[RESUME] ALL DONE - Resumed from round {args.start_round}", "green")
        return


    elif args.resume_round00:
        round_path = os.path.abspath(args.resume_round00)
        print(f"[RESUME] Loading round_00 data: {round_path}", flush=True)
        with open(round_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        data = loaded["data"] if isinstance(loaded, dict) and "data" in loaded else loaded
        if not isinstance(data, list):
            raise ValueError("resume_round00 JSON must be a list or dict with key 'data'.")
        print(f"✓ Loaded round_00 data: {len(data)} tasks", flush=True)


        model_basename = os.path.basename(args.pretrained_model.rstrip("/"))
        model_basename_api = os.path.basename(args.api_model_name.rstrip("/"))
        if not args.use_api:
            outputs_name = "results_eval_" + model_basename + "_" + args.dataset
        else:
            outputs_name = "results_eval_" + model_basename_api + "_" + args.dataset
        print(f"[RESUME] outputs_name = {outputs_name}", flush=True)


        if not args.use_api:
            tokenizer = AutoTokenizer.from_pretrained(
                args.pretrained_model,
                trust_remote_code=args.trust_remote_code,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                "/data/chenhui/models/Qwen_Qwen2.5-7B-Instruct",
                trust_remote_code=True,
            )
        init_usage_tracker(args, tokenizer, data)


        os.makedirs(
            os.path.dirname(f"./temp_data/{args.mode}/outputs_" + outputs_name + ".json"),
            exist_ok=True,
        )
        sync_usage_to_data(args, data)
        with open(
            f"./temp_data/{args.mode}/outputs_" + outputs_name + ".json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=convert_ndarray)


        data = run_all_executions(data, args)


        runner = None
        if getattr(args, "use_self_play", True) and args.self_play_round > 0:
            print("[RESUME] Entering Self_Play_V2 stage...", flush=True)
            runner = ModelRunner(args)
            data = run_self_play_iterations(data, tokenizer, args, outputs_name, shared_runner=runner)
        else:
            print("[RESUME] self_play_round <= 0 or use_self_play=False; skipping self-play.", flush=True)


        data = compute_and_log_metrics(data, outputs_name, 0, 0, args)


        sync_usage_to_data(args, data)
        with open(
            f"./temp_data/{args.mode}/outputs_" + outputs_name + ".json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=convert_ndarray)
        usage_final_text = format_usage_final_summary(args, label="Final")
        if usage_final_text:
            print(usage_final_text, flush=True)
            _append_to_results_txt(args, outputs_name, usage_final_text)

        if runner is not None:
            runner.close()
        cprint("[RESUME] ALL DONE - Self-Play V2", "green")
        return


    print(f"Current working directory: {os.getcwd()}")
    dataset_path = os.path.abspath("../CURE_data/" + args.dataset + ".json")
    print(f"Looking for dataset file: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    num = len(data)
    print(f"✓ Loaded dataset {args.dataset}: {num} problems", flush=True)


    model_basename = os.path.basename(args.pretrained_model.rstrip("/"))
    model_basename_api = os.path.basename(args.api_model_name.rstrip("/"))
    if not args.use_api:
        outputs_name = "results_eval_" + model_basename + "_" + args.dataset
    else:
        outputs_name = "results_eval_" + model_basename_api + "_" + args.dataset


    if not args.use_api:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model,
            trust_remote_code=args.trust_remote_code,
        )
    else:

        tokenizer = AutoTokenizer.from_pretrained(
            "/data/chenhui/models/Qwen_Qwen2.5-7B-Instruct",
            trust_remote_code=True,
        )
    init_usage_tracker(args, tokenizer, data)


    (
        code_generation_prompts,
        code_index,
        case_generation_prompts,
        case_index,
    ) = build_prompts_for_dataset(data, args)


    runner = ModelRunner(args)
    (
        data,
        code_generation_result,
        case_generation_result,
        mean_code,
        mean_case,
    ) = run_generation_pipeline(
        data,
        code_generation_prompts,
        code_index,
        case_generation_prompts,
        case_index,
        runner,
        tokenizer,
        args,
    )

    cprint("generation job done!", "green")
    print(
        f"✓ Generated code candidates: {len(code_generation_result)}, generated tests: {len(case_generation_result)}",
        flush=True,
    )
    usage_initial_text = format_usage_round_summary(
        args,
        initial_round_key(),
        label="Initial Generation",
    )
    if usage_initial_text:
        print(usage_initial_text, flush=True)
        _append_to_results_txt(args, outputs_name, usage_initial_text)


    sync_usage_to_data(args, data)
    os.makedirs(
        os.path.dirname(f"./temp_data/{args.mode}/outputs_" + outputs_name + ".json"),
        exist_ok=True,
    )
    with open(
        f"./temp_data/{args.mode}/outputs_" + outputs_name + ".json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=convert_ndarray)


    data = run_all_executions(data, args)


    if getattr(args, "use_self_play", True) and args.self_play_round > 0:
        print("Entering self-play stage...", flush=True)
        data = run_self_play_iterations(data, tokenizer, args, outputs_name, shared_runner=runner)

    else:
        print("self_play_round <= 0 or use_self_play=False; skipping self-play.", flush=True)


    data = compute_and_log_metrics(data, outputs_name, mean_code, mean_case, args)


    sync_usage_to_data(args, data)
    with open(
        f"./temp_data/{args.mode}/outputs_" + outputs_name + ".json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=convert_ndarray)

    usage_final_text = format_usage_final_summary(args, label="Final")
    if usage_final_text:
        print(usage_final_text, flush=True)
        _append_to_results_txt(args, outputs_name, usage_final_text)
    cprint("ALL DONE - Self-Play V2", "green")
    runner.close()


if __name__ == "__main__":
    main()
