def convert_ndarray(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)
# main.py
import os
import ast
import json
import argparse

import numpy as np
from transformers import AutoTokenizer
from termcolor import cprint

from prompts import build_prompts_for_dataset
from inference import ModelRunner
from generator_v3 import run_generation_pipeline  # 运行生成主流程
from execution import run_all_executions
from metrics import compute_and_log_metrics
from self_play_v3 import run_self_play_iterations  # 导入 self-play 函数
from usage_tracking import (
    format_usage_final_summary,
    format_usage_round_summary,
    init_usage_tracker,
    initial_round_key,
    sync_usage_to_data,
)

import evaluation_config



# 可选：关掉 tokenizer 的并行 warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def str2bool(x):
    return x.lower() in ("1", "true", "yes")


import argparse
import ast
# 假设 evaluation_config, str2bool, cprint 已经在外部定义或导入
# from utils import str2bool, cprint (根据你的实际项目结构)
# import evaluation_config as ec

def parse_args():
    """
    从 evaluation_config 读取默认配置，并统一整理两类模式，
    并在最后打印所有参数供检查。
    """
    parser = argparse.ArgumentParser()
    ec = evaluation_config

    # ================== 核心模式选择 ==================
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

    # ================== 基础参数 ==================
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

    # ================== 评估配置 (BoN / Pass@k) ==================
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
    parser.add_argument(  # random UT 聚类开关
        "--use_random_ut_cluster",  # 开关参数名
        type=str2bool,  # 布尔解析
        default=getattr(ec, "use_random_ut_cluster", False),  # 默认值
    )  # random UT 开关参数结束
    parser.add_argument(  # random UT 总数
        "--random_ut_total",  # 总数参数名
        type=int,  # 整数解析
        default=getattr(ec, "random_ut_total", 0),  # 默认值
    )  # random UT 总数参数结束
    parser.add_argument(  # random UT 分批大小
        "--random_ut_batch",  # 分批参数名
        type=int,  # 整数解析
        default=getattr(ec, "random_ut_batch", 16),  # 默认值
    )  # random UT 分批参数结束
    parser.add_argument(  # random UT 重试次数
        "--random_ut_max_attempts",  # 重试参数名
        type=int,  # 整数解析
        default=getattr(ec, "random_ut_max_attempts", 5),  # 默认值
    )  # random UT 重试参数结束
    parser.add_argument(  # random UT 聚类触发阈值
        "--random_ut_min_top_count",  # top-pass 阈值参数名
        type=int,  # 整数解析
        default=getattr(ec, "random_ut_min_top_count", 2),  # 默认值
    )  # random UT 阈值参数结束
    parser.add_argument(  # random UT 聚类距离
        "--random_ut_cluster_max_diff",  # 聚类距离参数名
        type=int,  # 整数解析
        default=getattr(ec, "random_ut_cluster_max_diff", 0),  # 默认值
    )  # random UT 距离参数结束
    parser.add_argument(  # random UT 占位符
        "--random_ut_placeholder",  # 占位符参数名
        type=str,  # 字符串解析
        default=getattr(ec, "random_ut_placeholder", "We can not extract the input in the output. "),  # 默认值
    )  # random UT 占位符参数结束
    # ================== 内部控制开关 ==================
    parser.add_argument("--single_eval", type=str2bool, default=getattr(ec, "single_eval", False))
    parser.add_argument("--eval_pass_at_k_only", type=str2bool, default=getattr(ec, "eval_pass_at_k_only", False))
    parser.add_argument("--eval_bon", type=str2bool, default=getattr(ec, "eval_bon", False))

    # ================== API 相关 ==================
    parser.add_argument("--api_model_name", type=str, default=ec.api_model_name)
    parser.add_argument("--api_key", type=str, default=ec.api_key)
    parser.add_argument("--base_url", type=str, default=ec.base_url)
    parser.add_argument("--api_temperature", type=float, default=ec.api_temperature)
    parser.add_argument("--max_workers", type=int, default=ec.max_workers)
    parser.add_argument("--use_openai_batch_api", type=str2bool, default=ec.use_openai_batch_api)
    parser.add_argument("--max_tokens", type=int, default=ec.max_tokens)
    parser.add_argument("--rpm_limit", type=int, default=ec.rpm_limit)

    # ================== Prompt & PlanSearch 相关 ==================
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
    # parser.add_argument("--use_idea_attack_ut_step", type=str, default=getattr(ec, "use_idea_attack_ut_step", "default"))
    parser.add_argument("--num_ideas", type=int, default=getattr(ec, "num_ideas", 1))
    parser.add_argument("--self_consistency_num", type=int, default=getattr(ec, "self_consistency_num", 1))
    parser.add_argument("--self_play_round", type=int, default=getattr(ec, "self_play_round", 1))
    parser.add_argument("--is_empty", type=str2bool, default=getattr(ec, "is_empty", True))
    parser.add_argument("--ut_vote_by_code", type=str2bool, default=getattr(ec, "ut_vote_by_code", False))
    parser.add_argument("--use_self_play", type=str2bool, default=getattr(ec, "use_self_play", True))
    # ================== 自博弈相关参数 ==================
    parser.add_argument("--ut_accuracy_target", type=float, default=getattr(ec, "ut_accuracy_target", 0.5))
    parser.add_argument("--ut_regen_max_attempts", type=int, default=getattr(ec, "ut_regen_max_attempts", 3))
    parser.add_argument("--skip_attack_when_all_pass", type=str2bool, default=getattr(ec, "skip_attack_when_all_pass", True))
    # ================== Resume from any round ==================
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
    # ================== 解析参数 ==================
    args = parser.parse_args()

    # ===============================================================
    # 1. 统一整理「生成模式」 (Generation Logic)
    # ===============================================================
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

    # ===============================================================
    # 2. 统一整理「评测模式」 (Evaluation Logic)
    # ===============================================================
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
            cprint("[WARNING] eval_mode='bon' 但 k_case <= 0，将无法生成测试用例！", "red")

    else:
        raise ValueError(f"Unknown eval_mode: {args.eval_mode}")
    
    try:
        # 获取对应 Mode 的所有 Prompts 数据
        prompt_data = ec.PROMPT_REGISTRY[args.prompt_role_mode]
        
        # 1. 加载固定 Prompts
        args.system_prompts_stage1 = prompt_data["stage1"]
        args.system_prompts_stage2 = prompt_data["stage2"]
        args.system_prompts = prompt_data["original"]
        args.system_case_prompts = prompt_data["case"]
        args.special_requirements = prompt_data["special_requirements"]
        args.system_prompts_stage4_ablation_only_stage = prompt_data["stage4"][args.ablation]
        
        cprint(f"[INFO] Loaded Prompts for Mode {args.prompt_role_mode}", "green")
        
    except KeyError as e:
        raise ValueError(f"Unsupported prompt_role_mode: {args.prompt_role_mode}. Check PROMPT_REGISTRY.") from e
    

    # ================== 打印所有调试信息 (FULL DEBUG) ==================
    cprint(f"\n[CONFIG SUMMARY] (All Arguments)", "yellow")
    
    # 获取参数字典
    args_dict = vars(args)
    
    # 按照 Key 排序打印，方便查找
    print(f"{'ARGUMENT':<35} : {'VALUE'}")
    print("-" * 80)
    for key, value in sorted(args_dict.items()):
        # 对部分长文本（如 Prompt）可以做截断显示，或者直接全部打印
        # 这里选择全部打印，保持原始值
        print(f"{key:<35} : {value}")
    print("-" * 80 + "\n", flush=True)

    return args


def _append_to_results_txt(args, outputs_name: str, text: str):
    """
    将 usage summary 追加写入与 metrics.py 相同目录结构下的 .txt 结果文件。
    """
    if not outputs_name or not text:
        return
    if getattr(args, "is_final_eval", False):
        out_path = f"../CURE_results/results_final_eval/{args.mode}/{outputs_name}_final_eval.txt"
    else:
        out_path = f"../CURE_results/results_optimization_eval/{args.mode}/{outputs_name}.txt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(text.rstrip("\n") + "\n")

def main():
    """
    主函数流程：
    1. 解析参数 (parse_args)
    2. 加载数据集
    3. 初始化 Tokenizer (用于计算 token 数)
    4. 构建 Prompts (build_prompts_for_dataset)
    5. 初始化 ModelRunner 并执行推理 (run_generation_pipeline)
    6. 保存生成结果
    7. 执行代码和测试用例 (run_all_executions)
    8. 触发 self_play_v2 的多轮迭代
    9. 再次执行并评测
    """
    args = parse_args()

    # ===== Resume from any round: prioritize --resume_round over --resume_round00 =====
    if args.resume_round:
        round_path = os.path.abspath(args.resume_round)
        print(f"[RESUME] Loading round data: {round_path}", flush=True)
        with open(round_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        data = loaded["data"] if isinstance(loaded, dict) and "data" in loaded else loaded
        if not isinstance(data, list):
            raise ValueError("resume_round JSON must be a list or dict with key 'data'.")
        
        # Auto-detect round number from filename if not specified
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

        # outputs_name: keep the same logic as original main
        model_basename = os.path.basename(args.pretrained_model.rstrip("/"))
        model_basename_api = os.path.basename(args.api_model_name.rstrip("/"))
        if not args.use_api:
            outputs_name = "results_eval_" + model_basename + "_" + args.dataset
        else:
            outputs_name = "results_eval_" + model_basename_api + "_" + args.dataset
        print(f"[RESUME] outputs_name = {outputs_name}", flush=True)

        # tokenizer
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

        # 创建输出目录
        os.makedirs(
            os.path.dirname(f"./temp_data/{args.mode}/outputs_" + outputs_name + ".json"),
            exist_ok=True,
        )

        # 自博弈阶段 - 从指定轮次开始
        runner = None
        if getattr(args, "use_self_play", True) and args.self_play_round > args.start_round:
            print(f"[RESUME] Entering Self_Play from round {args.start_round + 1}...", flush=True)
            runner = ModelRunner(args)
            # 调整 self_play_round 为剩余轮数
            remaining_rounds = args.self_play_round - args.start_round
            print(f"[RESUME] Remaining rounds to execute: {remaining_rounds}", flush=True)
            original_rounds = args.self_play_round
            args.self_play_round = remaining_rounds
            data = run_self_play_iterations(data, tokenizer, args, outputs_name, shared_runner=runner)
            args.self_play_round = original_rounds  # 恢复原值
        else:
            print(f"[RESUME] No remaining self-play rounds (start={args.start_round}, total={args.self_play_round})", flush=True)

        # 计算指标
        data = compute_and_log_metrics(data, outputs_name, 0, 0, args)

        # 写最终结果
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

    # ===== Resume: load round_00 data and skip generation =====
    elif args.resume_round00:
        round_path = os.path.abspath(args.resume_round00)
        print(f"[RESUME] Loading round_00 data: {round_path}", flush=True)
        with open(round_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        data = loaded["data"] if isinstance(loaded, dict) and "data" in loaded else loaded
        if not isinstance(data, list):
            raise ValueError("resume_round00 JSON must be a list or dict with key 'data'.")
        print(f"✓ Loaded round_00 data: {len(data)} tasks", flush=True)

        # outputs_name: keep the same logic as original main
        model_basename = os.path.basename(args.pretrained_model.rstrip("/"))
        model_basename_api = os.path.basename(args.api_model_name.rstrip("/"))
        if not args.use_api:
            outputs_name = "results_eval_" + model_basename + "_" + args.dataset
        else:
            outputs_name = "results_eval_" + model_basename_api + "_" + args.dataset
        print(f"[RESUME] outputs_name = {outputs_name}", flush=True)

        # tokenizer：用于统计 response 长度（和模型本身无关，只是计长度）
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

        # 先写一份中间结果，方便出错时排查（保持原逻辑顺序）
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

        # 3) 执行代码，跑 UT（保持与原逻辑一致）
        data = run_all_executions(data, args)

        # 4) 自博弈阶段
        runner = None
        if getattr(args, "use_self_play", True) and args.self_play_round > 0:
            print("[RESUME] Entering Self_Play_V2 stage...", flush=True)
            runner = ModelRunner(args)
            data = run_self_play_iterations(data, tokenizer, args, outputs_name, shared_runner=runner)
        else:
            print("[RESUME] self_play_round <= 0 或 use_self_play=False，跳过 self-play。", flush=True)

        # 5) 计算指标（保持原逻辑）
        data = compute_and_log_metrics(data, outputs_name, 0, 0, args)

        # 写最终结果
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

    # 读数据集
    print(f"当前工作目录: {os.getcwd()}")
    dataset_path = os.path.abspath("../CURE_data/" + args.dataset + ".json")
    print(f"寻找文件: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    num = len(data)
    print(f"✓ 载入数据集 {args.dataset}, 共 {num} 道题", flush=True)

    # 输出文件名前缀
    model_basename = os.path.basename(args.pretrained_model.rstrip("/"))
    model_basename_api = os.path.basename(args.api_model_name.rstrip("/"))
    if not args.use_api:
        outputs_name = "results_eval_" + model_basename + "_" + args.dataset
    else:
        outputs_name = "results_eval_" + model_basename_api + "_" + args.dataset

    # tokenizer：用于统计 response 长度（和模型本身无关，只是计长度）
    if not args.use_api:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        # API 场景下随便选一个兼容 tokenizer 用来算 token 数即可
        tokenizer = AutoTokenizer.from_pretrained(
            "/data/chenhui/models/Qwen_Qwen2.5-7B-Instruct",
            trust_remote_code=True,
        )
    init_usage_tracker(args, tokenizer, data)

    # 1) 生成 prompts（内部根据 use_multi_stage_generation / generation_mode 决定是否走 PlanSearch）
    (
        code_generation_prompts,
        code_index,
        case_generation_prompts,
        case_index,
    ) = build_prompts_for_dataset(data, args)

    # 2) 用 LLM 生成输出（代码 + 可选 UT）
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
    # runner.close()
    cprint("generation job done!", "green")
    print(
        f"✓ 生成代码: {len(code_generation_result)} 条, 生成测试: {len(case_generation_result)} 条",
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

    # 先写一份中间结果，方便出错时排查
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

    # 3) 执行代码，跑 UT（内部会根据 single_eval / 是否有 case_input 等决定跑什么）
    data = run_all_executions(data, args)

    # 4) 自博弈阶段 (Self-Play) 交由 self_play_v2 内部循环
    if getattr(args, "use_self_play", True) and args.self_play_round > 0:
        print("进入 Self_Play_V2 阶段...", flush=True)
        data = run_self_play_iterations(data, tokenizer, args, outputs_name, shared_runner=runner)
        # 注意：self_play_v3 最后一轮已经执行过 run_all_executions，这里不需要重复执行
    else:
        print("self_play_round <= 0 或 use_self_play=False，跳过 self-play。", flush=True)

    # 5) 计算指标（oneshot / pass@k-only / BoN 都在 metrics 里统一处理）
    data = compute_and_log_metrics(data, outputs_name, mean_code, mean_case, args)

    # 写最终结果（带执行结果 & 布尔表）
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
