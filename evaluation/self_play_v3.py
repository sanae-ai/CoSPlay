import json  # 写入每轮 self-play 快照
import os  # 处理快照存储路径
import copy  # 生成临时 args

from execution import run_all_executions_generate, run_all_executions, _compute_new_bon_with_history  # 导入代码执行引擎
from inference import ModelRunner  # 自博弈阶段也需要推理
from UT_config import (
    get_ut_input_generation_prompt,
    get_ut_input_random_generation_prompt,
    get_ut_output_generation_prompt,
    get_ut_output_refine_prompt,
    get_full_prompt,
    get_fix_code_prompt_with_ut,
)  # 导入修复和扰动 Prompt 构造函数
from generator_v3 import extract_code, extract_ut_input, extract_ut_output  # 导入代码与 UT 解析函数
from generator_v3 import parse_random_case_inputs, _strip_case_prefix  # random UT 输入解析工具
from collections import Counter  # 用于自一致性投票
import math  # 计算自一致结果的阈值
import random  # 用于随机选择攻击想法
import hashlib  # 导入哈希库，用于去重
from prompts import get_stage2_prompt  # 用二阶观察生成代码
from metrics import compute_and_log_metrics  # 复用正式评测逻辑写入独立文件
from usage_tracking import (
    consume_item_usage,
    format_usage_round_summary,
    record_direct_usage,
    self_play_round_key,
    sync_usage_to_data,
)


def convert_ndarray(obj):
    import numpy as np  # 延迟导入以避免主模块加载开销

    if isinstance(obj, np.ndarray):  # ndarray 转普通 list
        return obj.tolist()
    if isinstance(obj, (np.integer,)):  # numpy 整数转 Python int
        return int(obj)
    if isinstance(obj, (np.floating,)):  # numpy 浮点转 Python float
        return float(obj)
    return str(obj)  # 其他类型统一转字符串，确保可序列化

"""
自博弈模块

逻辑：
0. 从二阶观察生成16段代码和16个通过了自一致性验证的UT(多取几个二阶观察去进行自一致验证生成, 比如25, 保证凑够16个验证过的UT)
0.5. 对所有正确率为0的code重新采样一次, 使用新的二阶观察去生成代码, 不需要自一致性验证，直接替换
1. UT 攻击代码, 使用接近50%正确的 UT, 用 Idea 和 UT 本身让模型 fix 代码
2. 用 fix 后的代码执行原来的 UT, 对于让所有代码全部false的UT, 认为该UT本身有问题, 删除这个UT, 并重新取一个攻击想法去生成一个通过自一致验证的UT
3. 停止条件：循环重复上述过程直到最大轮数
4. BoN

数据结构（关键字段）：
- data: list[dict]，每个 dict 是一道题的数据
  - question: str，题目文本
  - generated_code: list[str]，原始代码列表
  - full_code_generation: list[str]，原始代码完整输出
  - case_input: list[str]，UT 输入
  - case_output: list[str]，UT 输出
  - case_is_valid: list[bool]，UT 是否有效的掩码
  - case_bool_table: list[list[bool]]，执行结果矩阵，形状 [num_codes][num_uts]
  - attack_ideas: list[str]，UT 对应的攻击想法
  - refined_codes: list[str]，修复后代码（由本 Step 写回）
  - refined_codes_full: list[str]，修复后完整输出（由本 Step 写回）
  - attack_ut_indices: list[int]，被选中的“接近 50%”的 UT 索引
  - attack_ut_pass_rates: dict[int,float]，UT 索引到通过率的映射
"""

# bool矩阵的行是代码，列是UT

def _format_bool_matrix(matrix, max_rows=8, max_cols=16):
    """
    将布尔矩阵裁剪并格式化为可打印字符串。
    """
    if matrix is None:  # 如果矩阵为空直接返回占位说明
        return "[empty bool matrix]"
    try:
        import numpy as np  # 延迟导入 numpy 以避免主路径性能损耗

        if isinstance(matrix, np.ndarray):  # 如果是 numpy 数组则转换为列表
            matrix = matrix.tolist()  # 转成原生列表方便后续操作
    except Exception:
        pass  # 转换失败时忽略，沿用原值
    if not matrix:  # 再次检查为空列表的情况
        return "[empty bool matrix]"
    first_row = matrix[0]  # 抓取第一行检测列结构
    if isinstance(first_row, (list, tuple)):  # 如果第一行本身是序列直接取长度
        cols = len(first_row)  # 记录列数量
    else:
        cols = len(matrix)  # 若第一行是标量，说明矩阵是一维，需要把长度视作列数
        matrix = [matrix]  # 将一维转换为单行二维列表
    preview = []  # 保存裁剪后的行文本
    row_count = min(len(matrix), max_rows)  # 控制输出的最大行数
    col_count = min(cols, max_cols)  # 控制输出的最大列数
    for r in range(row_count):
        row = matrix[r][:col_count]
        preview.append(" ".join("1" if cell else "0" for cell in row))
    suffix = ""
    if len(matrix) > max_rows or len(matrix[0]) > max_cols:
        suffix = f" ... (showing {row_count}x{col_count} of {len(matrix)}x{len(matrix[0])})"
    return "\n        " + "\n        ".join(preview) + suffix


def _log_problem_bool_matrix(data, problem_idx, label):
    """
    打印指定题目的布尔矩阵（如果存在）。
    """
    if not data or problem_idx >= len(data):  # 检查题目索引是否有效
        print(f"    - {label}: problem data unavailable.", flush=True)
        return
    table = data[problem_idx].get("case_bool_table")  # 优先使用最新的 case_bool_table
    if table is None:  # 如果没有则尝试使用 refined 结果
        table = data[problem_idx].get("refined_bool_table")
    if table is None:  # 两者都不存在时给出提示
        print(f"    - {label}: bool matrix missing.", flush=True)
        return
    print(f"    - {label}: bool matrix preview:\n{_format_bool_matrix(table)}", flush=True)  # 打印格式化后的矩阵


def _record_round_bool_history(data, round_idx):
    """
    将当前轮的布尔矩阵快照附加到每道题的数据中，便于最终 JSON 排查。
    """
    for prob_idx, item in enumerate(data):  # 遍历每道题
        bool_table = item.get("case_bool_table")  # 获取最新的布尔矩阵
        if bool_table is None:  # 没有矩阵时跳过
            continue
        history = item.setdefault("round_bool_history", [])  # 确保历史列表存在
        history.append(  # 追加当前轮的快照
            {
                "round": round_idx,  # 记录轮数
                "problem_index": prob_idx,  # 记录题目索引
                "bool_matrix": bool_table,  # 保存矩阵本体
            }
        )


def _clear_round_transient_fields(data):
    """
    清理只需要出现在单轮 snapshot 中的临时字段，避免串到下一轮。
    """
    transient_fields = [
        "ut_refine_round_trace",
    ]
    for item in data:
        for field in transient_fields:
            item.pop(field, None)


def _pick_new_attack_idea(data_i, idea_pool, ut_idx, fallback="Unknown"):
    """
    从 idea_pool 里选择未使用过的 idea；若池子用尽则重置。
    """
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


####################################
##             主逻辑             ##
####################################

def resample_code(data, runner, tokenizer, args, round_num=None):
    """
    Step 0.5: 对所有正确率为0的code重新采样一次, 使用新的二阶观察去生成代码, 不需要自一致性验证，直接替换
    
    功能：
    1. 识别正确率为0的代码
    2. 使用新的二阶观察重新生成代码
    3. 替换原有代码
    """
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

    resample_tasks = []  # (prob_idx, code_idx, obs)
    resample_prompts = []
    
    # 初始化 code resample 统计（如果不存在）
    for idx, data_i in enumerate(data):
        data_i.setdefault("code_resample_stats", {"self_play": 0})

    for idx, data_i in enumerate(data):  # 遍历所有题目
        # 跳过已完成的题目（有代码全通过）
        if data_i.get("skip_self_play", False):
            continue
        
        bool_table = data_i.get("case_bool_table")  # 获取执行结果矩阵（代码 × UT）
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

        # 记录已用于重采样的二阶观察（全局维度：同一题目内所有 code 共用）
        used_map = data_i.setdefault("resample_used_stage2_obs", {})
        used_global = set(used_map.get("_global", []))
        start_idx = int(getattr(args, "k_code", 16))  # 例如初始 16 个 code，则从第 17 个观察开始补
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

            # 从前往后选择没用过的二阶观察：优先从 cursor(>=k_code) 往后取，
            # 用尽后再从 start_idx 往后扫，仍用尽才允许从头扫。
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
        # 记录该 code 在 self_play 阶段的 resample 次数
        data[prob_idx]["code_resample_stats"]["self_play"] += 1

    return data, True

def ut_attack_code(data, runner, tokenizer, args, round_num=None):
    """
    Step 1: UT 攻击代码

    功能：
    1. UT 攻击代码
    2. 筛选使得代码接近 50% 正确的 UT
    3. 用 Idea 和 UT 本身让模型 fix 代码
    """
    print("\n=== Step 1: UT Attack + 50% Filter + Code Fix ===", flush=True)
    round_key = self_play_round_key(round_num)

    # target = getattr(args, "ut_accuracy_target", 0.5)  # 目标通过率（默认 0.5，可配置）
    target = 1.0
    skip_all_pass = getattr(args, "skip_attack_when_all_pass", True)
    refine_prompts = []  # 待生成修复代码的 prompt 列表
    refine_info = []  # 记录每个 prompt 对应的 (题目 idx, 代码 idx)

    for idx, data_i in enumerate(data):  # 遍历所有题目
        # 跳过已完成的题目（有代码全通过）
        if data_i.get("skip_self_play", False):
            continue
        
        data_i["attack_ut_indices"] = []  # 初始化被选中的 UT 索引列表
        data_i["attack_ut_pass_rates"] = {}  # 初始化 UT 通过率映射

        ## ----------- 结果矩阵处理 ----------- ##
        bool_table = data_i.get("case_bool_table")  # 获取执行结果矩阵（代码 × UT）
        if bool_table is None or len(bool_table) == 0:  # 如果没有执行结果，跳过
            print(f"Error: - Problem {idx}: No execution results.", flush=True)  
            continue  
        
        ## ---------- 获取代码 ----------- ##
        num_codes = len(bool_table)  # 获取代码总数
        num_cases = len(bool_table[0]) if num_codes > 0 else 0  # 获取 UT 总数
        if num_codes == 0 or num_cases == 0:  # 检查空矩阵
            print(f"Error: - Problem {idx}: Empty bool table.", flush=True)  
            continue  
        
        ## --------- 获取可以使用的 UT ----------- ##
        case_is_valid = data_i.get("case_is_valid", [True] * num_cases)  # 获取 UT 有效性掩码
        valid_indices = [j for j, valid in enumerate(case_is_valid) if valid]  # 筛选出有效 UT 的索引
        if not valid_indices:  # 如果没有有效 UT
            print(f"Error: - Problem {idx}: No valid UTs.", flush=True)  
            continue  

        # 计算每个 UT 的通过率（纵轴）
        ut_stats = []  # 存储 (ut_idx, pass_rate)
        for j in valid_indices:  # 遍历有效 UT
            passed = 0  # 统计该 UT 被多少代码通过
            for c in range(num_codes):  # 遍历所有代码
                if j < len(bool_table[c]) and bool_table[c][j]:  # 该代码通过此 UT
                    passed += 1  # 通过数 +1
            rate = passed / num_codes if num_codes > 0 else 0.0  # 计算通过率
            ut_stats.append((j, rate))  # 记录该 UT 的通过率

        if not ut_stats:
            print(f"Error: - Problem {idx}: No UT stats to select.", flush=True)
            continue
        # 挑选“未被所有代码通过”的 UT 中通过率最高的；若全通过且配置跳过，则不进行攻击
        failing_ut = [(j, r) for (j, r) in ut_stats if r < 1.0]
        if failing_ut:
            selected, selected_rate = max(failing_ut, key=lambda x: x[1])
        else:
            if skip_all_pass:
                print(f"    - Problem {idx}: all UTs passed by all codes, skip attack.", flush=True)
                continue
            selected, selected_rate = max(ut_stats, key=lambda x: x[1])

        if selected_rate <= 0.0:  # 防止 0 通过率 UT 带来噪声
            print(f"    - Problem {idx}: selected UT pass rate is 0, skip attack.", flush=True)  # 记录跳过
            continue  # 跳过该题

        data_i["attack_ut_indices"].append(selected)  # 保存每一轮被选中的 UT 索引
        data_i["attack_ut_pass_rates"] = {j: rate for j, rate in ut_stats}  # 保存 UT 通过率映射

        selected_input = ""
        selected_output = ""
        if selected < len(data_i.get("case_input", [])):
            selected_input = data_i["case_input"][selected]
        if selected < len(data_i.get("case_output", [])):
            selected_output = data_i["case_output"][selected]
        # 打印本轮ut的index，通过率和内容
        round_prefix = f"Round {round_num}: " if round_num is not None else ""
        print(
            f"{round_prefix}Problem {idx}: Selected UT index: {selected}, Pass Rate: {selected_rate:.2f}, "
            f"Content: Input: {selected_input}, Output: {selected_output}",
            flush=True,
        )

        if selected is None:  # 若仍为空则无法修复
            continue

        problem = data_i.get("question", "")  # 题目文本
        attack_ideas = data_i.get("attack_ideas", ["Unknown"] * num_cases)  # UT 攻击想法
        case_inputs = data_i.get("case_input", [])  # UT 输入
        case_outputs = data_i.get("case_output", [])  # UT 输出
        generated_codes = data_i.get("generated_code", [])  # 原始代码列表

        for c_idx in range(num_codes):
            if c_idx >= len(generated_codes):  # 防御性检查
                continue  

            # 查看代码是否在这个ut上出错，如果出错说明要fix
            if selected < len(bool_table[c_idx]) and not bool_table[c_idx][selected]:  # 该代码未通过选中的 UT
                failed_code = generated_codes[c_idx]  # 获取失败的代码
                attack_idea = attack_ideas[selected] if selected < len(attack_ideas) else "Unknown"  # 获取对应的攻击想法
                attack_ut_input = case_inputs[selected] if selected < len(case_inputs) else ""
                attack_ut_output = case_outputs[selected] if selected < len(case_outputs) else ""
                exe_output = ""
                case_exe = data_i.get("case_exe_results")
                if case_exe and c_idx < len(case_exe) and selected < len(case_exe[c_idx]):
                    exe_output = case_exe[c_idx][selected]

                # 构造修复 prompt：按 UT_config 的列表签名传入单条 UT
                prompt_list = get_fix_code_prompt_with_ut(
                    problem=problem,
                    failed_code=failed_code,
                    attack_ut_input=[attack_ut_input],
                    attack_ut_output=[attack_ut_output],
                    exe_output=[exe_output],
                    num_to_include=1,
                )
                prompt = get_full_prompt(prompt_list)
                refine_prompts.append(prompt)  # 添加到待生成列表
                refine_info.append((idx, c_idx))  # 记录对应的题目和代码索引

    # 无需修复时，保持原代码列表
    if not refine_prompts:
        for data_i in data:
            data_i["refined_codes"] = data_i.get("generated_code", []).copy()  # 直接复用原代码
            data_i["refined_codes_full"] = data_i.get("full_code_generation", []).copy()  # 复用完整输出
        return data  # 提前返回

    print(f"    - Generating {len(refine_prompts)} refined codes...", flush=True)  # 打印修复数量
    refined_outputs = runner.generate(refine_prompts)  # 批量生成修复代码
    record_direct_usage(
        args,
        [idx for idx, _ in refine_info],
        refine_prompts,
        refined_outputs,
        "self_play_code_fix",
        round_key,
    )

    # 初始化 refined 列表，保持代码数量不变
    for data_i in data:
        data_i["refined_codes"] = data_i.get("generated_code", []).copy()  # 先复制原代码
        data_i["refined_codes_full"] = data_i.get("full_code_generation", []).copy()  # 复制原输出

    round_label = f"Round {round_num}" if round_num is not None else "Round ?"  # 构造轮次标签用于日志
    for (idx, c_idx), output in zip(refine_info, refined_outputs):  # 逐条回填修复结果
        new_code = extract_code(output)  # 解析代码块
        data[idx]["refined_codes"][c_idx] = new_code  # 替换对应代码
        data[idx]["refined_codes_full"][c_idx] = output  # 保存完整输出
        # if idx == 0:  # 仅打印第一题的布尔矩阵
            # _log_problem_bool_matrix(  # 打印当前布尔矩阵状态，方便观察变化
            #     data,
            #     0,
            #     f"{round_label} After refining code index {c_idx}",
            # )

    print("    - Step 1 refine complete.", flush=True)  # 打印完成信息
    return data  # 返回更新后的数据

def ut_refine(data, runner, tokenizer, args, round_num=None):  # UT 细化入口
    # Step 0.5: 选择通过率最低(非0非1)的 UT 输出进行修复，降低噪声。  # 步骤说明
    # - 每轮为该 UT 更换 attack idea，并重新生成 input -> output。  # 范围说明
    # - 每轮最多 ut_regen_max_attempts 次，且每次都做自一致性。  # 次数说明
    # - 失败则标记无效，下一轮继续重试。  # 重试说明
    print("\n=== Step 0.5: UT Refine (lowest non-0/1 pass-rate) ===", flush=True)  # 打印开始
    round_key = self_play_round_key(round_num)
    max_attempts = getattr(args, "ut_regen_max_attempts", 5)  # 每个 UT 最大尝试次数
    sc_num = getattr(args, "self_consistency_num", 4)  # 自一致性采样次数
    min_consistency = max(1, math.ceil(sc_num * 0.75)) if sc_num > 1 else 1  # 一致性阈值
    did_refine = False  # 是否发生过细化
    ut_states = {}
    for data_i in data:
        data_i.pop("ut_refine_round_trace", None)
    for idx, data_i in enumerate(data):  # 遍历题目
        if data_i.get("skip_self_play", False):  # 跳过已完成题目
            continue  # 继续下一题
        bool_table = data_i.get("case_bool_table")  # 读取布尔矩阵
        if bool_table is None or len(bool_table) == 0:  # 结果为空
            print(f"    - Problem {idx}: No execution results.", flush=True)  # 记录跳过
            continue  # 跳过该题
        num_codes = len(bool_table)  # 代码数量
        num_cases = len(bool_table[0]) if num_codes > 0 else 0  # UT 数量
        if num_codes == 0 or num_cases == 0:  # 空矩阵保护
            print(f"    - Problem {idx}: Empty bool table.", flush=True)  # 记录跳过
            continue  # 跳过该题
        case_is_valid = data_i.get("case_is_valid", [True] * num_cases)  # UT 有效性掩码
        valid_indices = [j for j, valid in enumerate(case_is_valid) if valid and j < num_cases]  # 有效 UT
        if not valid_indices:  # 无有效 UT
            print(f"    - Problem {idx}: No valid UTs.", flush=True)  # 记录跳过
            continue  # 跳过该题
        ut_stats = []  # 记录 (ut_idx, pass_rate)
        for j in valid_indices:  # 计算每个 UT 通过率
            passed = 0  # 通过计数
            for c in range(num_codes):  # 遍历代码
                if j < len(bool_table[c]) and bool_table[c][j]:  # 统计通过
                    passed += 1  # 累计通过数
            rate = passed / num_codes if num_codes > 0 else 0.0  # 通过率
            ut_stats.append((j, rate))  # 保存通过率
        candidates = [(j, r) for (j, r) in ut_stats if 0.0 < r < 1.0]  # 过滤非 0/1 UT
        if not candidates:  # 无候选
            continue  # 跳过该题
        selected, selected_rate = min(candidates, key=lambda x: x[1])  # 选择最低通过率
        problem = data_i.get("question", "")  # 题目文本
        case_inputs = data_i.get("case_input", [])  # UT 输入
        case_outputs = data_i.get("case_output", [])  # UT 输出
        case_output_original = data_i.get("case_output_original", [])  # 原始输出
        case_input_original = data_i.get("case_input_original", [])  # 原始输入
        case_sources = data_i.get("case_source", [])  # UT 来源
        full_case_generation = data_i.get("full_case_generation", [])  # 完整日志
        case_text = data_i.get("case_text", [])  # 文本日志
        attack_ideas = data_i.get("attack_ideas", [])  # 攻击想法
        attack_ideas_candidates = data_i.get("attack_ideas_candidates", [])  # 候选攻击想法
        k_code = len(data_i.get("generated_code", []))
        def _ensure_len(lst, length, fill_val):  # 补齐列表
            while len(lst) < length:  # 直到足够长度
                lst.append(fill_val)  # 追加默认值
        _ensure_len(case_inputs, num_cases, "")  # 补齐输入
        _ensure_len(case_outputs, num_cases, "")  # 补齐输出
        _ensure_len(case_output_original, num_cases, "")  # 补齐原始输出
        _ensure_len(case_input_original, num_cases, "")  # 补齐原始输入
        _ensure_len(case_sources, num_cases, "unknown")  # 补齐来源
        _ensure_len(full_case_generation, num_cases, "")  # 补齐完整日志
        _ensure_len(case_text, num_cases, "")  # 补齐文本日志
        _ensure_len(attack_ideas, num_cases, "Unknown")  # 补齐攻击想法
        _ensure_len(case_is_valid, num_cases, True)  # 补齐有效性
        data_i["case_input"] = case_inputs  # 写回输入
        data_i["case_output"] = case_outputs  # 写回输出
        data_i["case_output_original"] = case_output_original  # 写回原始输出
        data_i["case_input_original"] = case_input_original  # 写回原始输入
        data_i["case_source"] = case_sources  # 写回来源
        data_i["full_case_generation"] = full_case_generation  # 写回完整日志
        data_i["case_text"] = case_text  # 写回文本日志
        data_i["attack_ideas"] = attack_ideas  # 写回攻击想法
        data_i["case_is_valid"] = case_is_valid  # 写回有效性

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
        
        input_prompt = get_full_prompt(  # 先生成新的 input
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

    return data, did_refine  # 返回结果

def fix_code_attack_ut(data, runner, tokenizer, args, round_num=None):
    """
    Step 2: 执行修复后代码并清洗 UT

    功能：
    - 用 refined_codes 执行原有 UT
    - 找到“所有代码都失败”的 UT，判定为坏 UT
    - 重新取攻击想法生成新 UT（含自一致性验证），替换坏 UT

    约束/策略：
    - 自一致性要求“全票一致”才接受（freq == self_consistency_num）
    - 每个坏 UT 最多尝试 ut_regen_max_attempts 次
    - 优先使用“未尝试过的 idea”，用尽后才允许复用
    - 超过最大次数仍失败，则把该 UT 标为无效
    """
    print("\n=== Step 2: Execute Refined Codes + Regenerate Bad UTs ===", flush=True)
    round_key = self_play_round_key(round_num)

    # 1) 执行 refined_codes，得到新的布尔矩阵
    temp_data_for_exec = []  # 临时执行数据列表
    for data_i in data:  # 遍历每个题目
        temp_item = {  # 构造执行所需字段
            "question": data_i.get("question", ""),  # 题目文本
            "generated_code": data_i.get("refined_codes", data_i.get("generated_code", [])),  # 修复后代码
            "case_input": data_i.get("case_input", []),  # UT 输入
            "case_output": data_i.get("case_output", []),  # UT 输出
            "test_input": data_i.get("test_input", []),  # 真实测试输入（可为空）
            "test_output": data_i.get("test_output", []),  # 真实测试输出（可为空）
            "test_time_limit": data_i.get("test_time_limit", 1),  # 运行时限
        }
        temp_data_for_exec.append(temp_item)  # 加入临时执行列表

    run_all_executions_generate(temp_data_for_exec, args)  # 批量执行修复后代码

    # 写回 refined 结果矩阵（后续用来判断哪些 UT 全部失败）
    for data_i, temp_item in zip(data, temp_data_for_exec):  # 对齐原数据与执行结果
        data_i["refined_bool_table"] = temp_item.get("case_bool_table", [])  # 记录布尔矩阵
        data_i["refined_exe_results"] = temp_item.get("case_exe_results", [])  # 记录执行日志

    # 2) 检测坏 UT 并重生成
    max_attempts = getattr(args, "ut_regen_max_attempts", 5)  # 每个坏 UT 的最大重试次数
    sc_num = getattr(args, "self_consistency_num", 4)  # 自一致性采样次数
    
    # 初始化 UT resample 统计（如果不存在）
    for idx, data_i in enumerate(data):
        data_i.setdefault("ut_resample_stats", {"generator": 0, "self_play": 0})

    round_label = f"Round {round_num}" if round_num is not None else "Round ?"  # 保存轮次标签用于日志
    problem_tasks = []  # 收集所有题目的坏 UT，统一批量处理
    for idx, data_i in enumerate(data):  # 遍历每个题目
        # 跳过已完成的题目（有代码全通过）
        if data_i.get("skip_self_play", False):
            continue
        
        refined_bool = data_i.get("refined_bool_table")  # 取出修复后布尔矩阵
        if refined_bool is None or len(refined_bool) == 0:  # 若没有执行结果
            print(f"    - Problem {idx}: No refined bool table.", flush=True)
            continue 

        num_codes = len(refined_bool)  # 代码数量
        num_cases = len(refined_bool[0]) if num_codes > 0 else 0  # UT 数量
        if num_codes == 0 or num_cases == 0:  # 空矩阵保护
            print(f"    - Problem {idx}: Empty refined bool table.", flush=True)
            continue 

        # 记录上轮被屏蔽的 UT 索引（本轮强制重置）
        prev_case_is_valid = data_i.get("case_is_valid", [])
        prev_invalid_indices = [
            j for j, valid in enumerate(prev_case_is_valid) if not valid and j < num_cases
        ]

        # 每轮重置所有 UT 为有效状态（给每个 UT 新的 resample 机会）
        case_is_valid = [True] * num_cases
        data_i["case_is_valid"] = case_is_valid
        
        # 找出“所有代码都失败”的 UT 索引
        bad_ut_indices = []  # 记录“所有代码都失败”的 UT 索引
        for j in range(num_cases):  # 遍历每个 UT
            if not case_is_valid[j]:  # 若该 UT 已无效
                continue  # 跳过
            all_failed = True  # 默认认为所有代码都失败
            for c in range(num_codes):  # 遍历每个代码
                if j < len(refined_bool[c]) and refined_bool[c][j]:  # 只要有一个通过
                    all_failed = False  # 标记为非全失败
                    break  # 提前退出
            if all_failed:  # 若确实全失败
                bad_ut_indices.append(j)  # 记录该 UT 索引

        if prev_invalid_indices:
            # 上轮被屏蔽的 UT 本轮也强制重置
            bad_ut_indices = sorted(set(bad_ut_indices).union(prev_invalid_indices))

        if not bad_ut_indices:  # 若没有坏 UT
            continue 

        # 读取该题相关字段，准备替换 UT
        problem = data_i.get("question", "")  # 题目文本
        attack_ideas = data_i.get("attack_ideas", [])  # 攻击想法列表
        case_inputs = data_i.get("case_input", [])  # UT 输入列表
        case_outputs = data_i.get("case_output", [])  # UT 输出列表

        # 保证相关字段长度足够（避免越界写）
        def _ensure_len(lst, length, fill_val):  # 内部函数：补齐列表长度
            while len(lst) < length:  # 不足则补
                lst.append(fill_val)  # 追加默认值

        _ensure_len(case_inputs, num_cases, "")  # 补齐 UT 输入
        _ensure_len(case_outputs, num_cases, "")  # 补齐 UT 输出
        _ensure_len(case_is_valid, num_cases, True)  # 补齐有效掩码
        case_sources = data_i.get("case_source", [])
        _ensure_len(case_sources, num_cases, "unknown")
        data_i["case_source"] = case_sources

        if "case_input_original" not in data_i:  # 缺少原始输入字段
            data_i["case_input_original"] = []  # 初始化
        if "case_output_original" not in data_i:  # 缺少原始输出字段
            data_i["case_output_original"] = []  # 初始化
        if "full_case_generation" not in data_i:  # 缺少完整日志字段
            data_i["full_case_generation"] = []  # 初始化
        if "case_text" not in data_i:  # 缺少文本字段
            data_i["case_text"] = []  # 初始化

        _ensure_len(data_i["case_input_original"], num_cases, "")  # 补齐原始输入
        _ensure_len(data_i["case_output_original"], num_cases, "")  # 补齐原始输出
        _ensure_len(data_i["full_case_generation"], num_cases, "")  # 补齐完整日志
        _ensure_len(data_i["case_text"], num_cases, "")  # 补齐文本字段
        _ensure_len(attack_ideas, num_cases, "Unknown")  # 补齐攻击想法

        round_prefix = f"Round {round_num}: " if round_num is not None else ""  # 可选轮次前缀

        # 累积到全局任务，稍后跨题目一次性批量生成
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

    # ---------- 全局批量重生成坏 UT（跨题目合并，再批次生成，最多 max_attempts 轮） ----------
    if not problem_tasks:
        print("    - No bad UTs detected, skip regeneration.", flush=True)
        return data

    meta_by_prob = {t["idx"]: t for t in problem_tasks}
    
    # Step A: 为每个坏 UT 选择新的 attack idea（每轮 round 都换新的）
    input_prompts = []
    input_tasks = []
    for t in problem_tasks:
        prob_idx = t["idx"]
        problem = t["problem"]
        
        # 获取候选 attack ideas（从 data 中读取完整候选池）
        all_attack_ideas = data[prob_idx].get("attack_ideas_candidates", [])
        if not all_attack_ideas:
            # 如果没有候选池，使用当前的 attack_ideas
            all_attack_ideas = t["attack_ideas"]
        
        k_code = len(data[prob_idx].get("generated_code", []))
        
        for ut_idx in t["bad_ut_indices"]:
            # 为这个坏 UT 选择一个新的 attack idea（全局去重，池子用尽后重置）
            new_idea = _pick_new_attack_idea(data[prob_idx], all_attack_ideas, ut_idx, fallback="Unknown")
            consume_item_usage(args, prob_idx, "attack_idea", [new_idea], round_key)
            
            # 使用新 idea 生成新的 input
            input_prompt = get_full_prompt(
                get_ut_input_generation_prompt(problem, new_idea)
            )
            input_prompts.append(input_prompt)
            input_tasks.append((prob_idx, ut_idx, new_idea))
    
    # 批量生成所有 inputs
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
    
    # Step B: 构造 ut_states，保存每个 UT 的固定 idea、input 和 output_prompt
    ut_states = {}
    did_regen = False
    for (prob_idx, ut_idx, new_idea), input_output_raw in zip(input_tasks, input_outputs):
        new_input = extract_ut_input(input_output_raw)
        if not new_input or "We can not extract" in new_input:
            # 如果提取失败，使用原有的 input
            meta = meta_by_prob[prob_idx]
            new_input = meta["case_inputs"][ut_idx] if ut_idx < len(meta["case_inputs"]) else ""
        
        # 构造 output 生成的 prompt（基于新生成的 input）
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
        # 统计每个题目当前成功重生成的 UT 数量
        problem_success_count = {}
        for key, st in ut_states.items():
            if st["success"]:
                prob_idx = st["prob_idx"]
                problem_success_count[prob_idx] = problem_success_count.get(prob_idx, 0) + 1
        
        # 筛选待处理的 UT：排除已成功的，以及所属题目坏 UT 已全部修复的
        pending_keys = []
        for k, st in ut_states.items():
            if st["success"]:
                continue  # 已经成功的跳过
            prob_idx = st["prob_idx"]
            meta = meta_by_prob[prob_idx]
            total_bad_count = len(meta["bad_ut_indices"])
            
            # 如果该题目的坏 UT 已经全部修复成功，不再尝试其他坏 UT
            if problem_success_count.get(prob_idx, 0) >= total_bad_count:
                continue
            pending_keys.append(k)
        
        if not pending_keys:
            break

        # 构造扩展的 prompts（每个 UT 生成 self_consistency_num 次，保持相同 idea 和 input）
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

        # 批量生成 outputs（每个 UT 采样 self_consistency_num 次）
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

        # 自一致性筛选并写回成功的 UT
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

            # 更新 idea、input 和 output（每轮 round 都换新的）
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
            # 记录该 UT 在 self_play 阶段的重生成次数（尝试次数）
            data[prob_idx]["ut_resample_stats"]["self_play"] += ut_states[(prob_idx, ut_idx)]["attempts"]
            print(
                f"    - {meta['round_prefix']}Problem {prob_idx}: Regenerated UT output at index {ut_idx} (attempt {ut_states[(prob_idx, ut_idx)]['attempts']})",
                flush=True,
            )
            # 仅打印题目 0 的矩阵，方便调试
            if prob_idx == 0:
                _log_problem_bool_matrix(
                    data, 0, f"{meta['round_label']} After regenerating UT index {ut_idx}"
                )

    # Step D: 处理仍失败的 UT，并写回最新列表
    for key, st in ut_states.items():
        prob_idx, ut_idx = st["prob_idx"], st["ut_idx"]
        meta = meta_by_prob[prob_idx]
        case_is_valid = meta["case_is_valid"]
        if st["success"]:
            continue
        case_is_valid[ut_idx] = False
        # print(
        #     f"    - {meta['round_prefix']}Problem {prob_idx}: Failed to regenerate UT at index {ut_idx}",
        #     flush=True,
        # )

    for meta in problem_tasks:
        idx = meta["idx"]
        data[idx]["attack_ideas"] = meta["attack_ideas"]
        data[idx]["case_input"] = meta["case_inputs"]
        data[idx]["case_output"] = meta["case_outputs"]
        data[idx]["case_is_valid"] = meta["case_is_valid"]
        data[idx]["case_source"] = meta["case_source"]

    print("    - Step 2 UT regeneration complete.", flush=True)
    return data


def regenerate_all_pass_ut(data, runner, tokenizer, args, round_num=None):
    """
    Step 3.5: 重生成“全通过”的 UT（所有代码均通过的列）

    功能：
    - 找出当前布尔矩阵中“所有代码都通过”的 UT
    - 对这些 UT 重新生成 input/output（带自一致性筛选）
    - 失败则标记为无效
    """
    print("\n=== Step 3.5: Regenerate All-Pass UTs ===", flush=True)
    round_key = self_play_round_key(round_num)

    did_regen = False  # 是否发生过有效重生成
    max_attempts = getattr(args, "ut_regen_max_attempts", 5)  # 每个 UT 最大尝试次数
    sc_num = getattr(args, "self_consistency_num", 4)  # 自一致性采样次数

    # 初始化 UT resample 统计（如果不存在）
    for data_i in data:
        data_i.setdefault("ut_resample_stats", {"generator": 0, "self_play": 0})

    round_label = f"Round {round_num}" if round_num is not None else "Round ?"  # 保存轮次标签用于日志
    problem_tasks = []  # 收集所有题目的“全通过”UT，统一批量处理
    for idx, data_i in enumerate(data):  # 遍历每个题目
        if data_i.get("skip_self_play", False):
            continue

        bool_table = data_i.get("case_bool_table")  # 取出当前布尔矩阵
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
        valid_count = 0  # 统计有效 UT 数量（仅用于判断全列全通过）
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
        # 若该题所有有效 UT 都被所有 code 通过，则不进行 all-pass 重生成
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

    # Step A: 为每个全通过 UT 选择新的 attack idea（每轮 round 都换新的）
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

    # Step B: 构造 ut_states，保存每个 UT 的固定 idea、input 和 output_prompt
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
    """
    将 refined_* 字段写回主字段，方便下一轮使用以及最终评测。
    """
    for item in data:  # 遍历每道题的数据项
        if "refined_codes" in item:  # 若存在修复后的代码
            item["generated_code"] = item["refined_codes"]  # 用修复后的代码覆盖原始代码
        if "refined_codes_full" in item:  # 若存在修复后的完整输出
            item["full_code_generation"] = item["refined_codes_full"]  # 将完整输出同步回主字段
        if "refined_bool_table" in item:  # 若存在修复后的布尔矩阵
            item["case_bool_table"] = item["refined_bool_table"]  # 更新主布尔矩阵
        if "refined_exe_results" in item:  # 若存在修复后的执行日志
            item["case_exe_results"] = item["refined_exe_results"]  # 用修复日志替换原日志
    return data  # 返回写回后的数据列表


def _dump_round_data(data, args, outputs_name, round_idx):
    """
    将每一轮 self-play 的 data 写入单独 JSON，便于排查。
    """
    def _format_bool_matrix(mat):
        """将二维 bool/0/1 矩阵转成易读的字符串行列表。"""
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

    round_dir = os.path.join("temp_data", args.mode, "self_play_v2_rounds")  # 轮次快照保存目录
    os.makedirs(round_dir, exist_ok=True)  # 确保目录存在
    filename = f"round_{round_idx:02d}_{outputs_name}.json"  # 轮次文件名，带两位序号
    file_path = os.path.join(round_dir, filename)  # 文件完整路径
    readable_data = [_add_readable_matrices(item) for item in data]
    with open(file_path, "w", encoding="utf-8") as f:  # 以 UTF-8 写入 JSON
        json.dump(readable_data, f, indent=2, ensure_ascii=False, default=convert_ndarray)  # 序列化 data
    print(f"    - Saved round {round_idx} snapshot to {file_path}", flush=True)  # 打印保存路径

def _append_to_results_txt(args, outputs_name: str, text: str):
    """
    将 self-play 中间日志追加写入与 metrics.py 相同目录结构下的 .txt 结果文件。
    """
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
    """
    复用正式 metrics 逻辑，每轮写入独立评测文件（追加模式），避免污染主结果。
    """
    if not outputs_name:
        return
    tmp_args = copy.copy(args)
    tmp_outputs = f"{outputs_name}_round{round_idx}"
    try:
        compute_and_log_metrics(data, tmp_outputs, mean_code=0, mean_case=0, args=tmp_args)
    except Exception as e:
        print(f"[WARN] write round {round_idx} metrics failed: {e}", flush=True)

def _format_resample_stats(data, round_label=""):
    """
    格式化 resample 统计信息，返回格式化后的字符串。
    """
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
    
    # 计算平均值
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
    """
    统计当前仍标记为 random 的 UT 数量（仅计入有效 UT）。
    """
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
    """
    按题目统计 random UT 数量（仅计入有效 UT）。
    返回 list[(problem_idx, count)]。
    """
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
    """
    每轮 self-play 重新生成 random UT inputs（32 选 16 用于 new_bon 聚类）。
    - 仅更新 data[i]["random_case_input"]
    - 清空 random_case_exe_results / random_case_bool_table_rows，强制重新执行
    """
    total_k_case = int(getattr(args, "k_case", 0))  # 读取每题 UT 数
    if total_k_case <= 0:
        return
    random_candidates_per_problem = total_k_case * 2  # 32 候选（用于后续选 16）
    num_problems = len(data)

    # 构造随机输入生成 prompt（每题 random_candidates_per_problem 条）
    # 这里先收集为 (prompt, problem_idx) 的列表，后续整体打乱顺序，避免同质化输出
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
    # 打乱 prompt 顺序后生成，再按映射回填
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

    # 初始化容器
    random_inputs_by_problem = [[] for _ in range(num_problems)]

    # 解析输入并写回
    for output, idx in zip(outputs, prompt_to_problem):
        candidates = parse_random_case_inputs(output)
        ut_input = candidates[0] if candidates else extract_ut_input(output)
        ut_input = _strip_case_prefix(ut_input)
        if not ut_input:
            continue
        random_inputs_by_problem[idx].append(ut_input)

    # 写回 data，并清空旧执行缓存
    from metrics import _select_random_inputs  # 复用同一套补齐逻辑
    placeholder = getattr(args, "random_ut_placeholder", "We can not extract the input in the output. ")
    target_count = int(getattr(args, "k_case", total_k_case))
    for idx in range(num_problems):
        # 先对 32 个候选去重（保序），再选 16 个，不足用占位符补齐
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
        # 清空历史 random 执行缓存，确保本轮重新执行
        for attempt in range(1, 6):
            data[idx].pop(f"random_case_exe_results_{attempt}", None)
            data[idx].pop(f"random_case_bool_table_rows_{attempt}", None)


def run_self_play_iterations(data, tokenizer, args, outputs_name, shared_runner=None):
    """
    在 self_play_v2 中执行多轮迭代。

    main_self_play_v2 不再负责多轮循环，统一由该函数管理：
    1. 读取 data（需包含 case_bool_table）
    2. 依次执行 ut_attack_code / fix_code_attack_ut
    3. 将 refined_* 结果写回主字段，便于下一轮
    4. 再次运行 run_all_executions，刷新 case_bool_table + test_bool_table
    5. 每轮结束后，将 data 写入 JSON 便于排查
    """
    total_rounds = max(int(getattr(args, "self_play_round", 0)), 0)  # 读取迭代轮次，确保非负
    if total_rounds == 0:  # 若无需迭代
        print("self_play_round=0, 跳过 self-play 流程", flush=True)  # 提示跳过 self-play
        return data  # 原样返回

    runner = shared_runner  # 默认复用外部传入的 runner
    own_runner = False  # 标记是否需要在本函数内关闭 runner
    if runner is None:  # 如果外部没有提供 runner
        print(">>> Initializing dedicated ModelRunner for self-play", flush=True)
        runner = ModelRunner(args)  # 自己实例化一个
        own_runner = True  # 记下需要在结束时关闭
    else:
        print(">>> Reusing existing ModelRunner for self-play", flush=True)

    # 进入 self-play 前：打印一次正式评测（独立文件）
    remaining_random = _count_random_uts(data)
    per_prob = _count_random_uts_per_problem(data)
    print(f">>> Random UT remaining before self-play: {remaining_random}", flush=True)
    for pid, cnt in per_prob:
        print(f"    - Problem {pid}: random UT = {cnt}", flush=True)
    _write_round_metrics(data, args, outputs_name, round_idx=0)
    # 在self-play开始前，保存初始快照
    print(">>> Dumping round 0 snapshot before self-play", flush=True)
    sync_usage_to_data(args, data)
    _dump_round_data(data, args, outputs_name, round_idx=0)
    for round_idx in range(total_rounds):  # 循环执行多轮自博弈
        print(f"\n{'=' * 80}", flush=True)  # 轮次分隔线
        print(f"=== Self-Play V2 Round {round_idx + 1}/{total_rounds} ===", flush=True)  # 打印轮次信息
        print(f"{'=' * 80}\n", flush=True)  # 尾部分隔线

        remaining_random = _count_random_uts(data)
        per_prob = _count_random_uts_per_problem(data)
        print(f">>> Random UT remaining at round start: {remaining_random}", flush=True)
        # 每轮开始先重生成 random UT 输入（用于 new_bon 聚类）
        _regenerate_random_ut_inputs(data, runner, args, round_idx + 1)
        for pid, cnt in per_prob:
            print(f"    - Problem {pid}: random UT = {cnt}", flush=True)

        print(">>> Step 0/7: Resampling 0-accuracy codes (before code fix)", flush=True)  # 步骤日志
        data, did_resample = resample_code(  # 执行代码重采样
            data, runner, tokenizer, args, round_num=round_idx + 1  # 传入轮次信息
        )  # 重采样结束
        # 注意：resample 后需要执行一次代码（只执行 case，不执行 test），更新 case_bool_table
        # 这样 ut_attack_code 才能基于最新的代码执行结果选择 UT
        if did_resample:  # 仅在发生重采样时刷新
            print(">>> Refreshing case executions after resample (case only)", flush=True)  # 刷新日志
            run_all_executions_generate(data, args)  # 只执行 case，不执行 test

        print(">>> Step 1/7: Refining low-accuracy UT outputs", flush=True)  # 步骤日志
        data, did_refine = ut_refine(data, runner, tokenizer, args, round_num=round_idx + 1)  # 执行 UT 细化
        if did_refine:  # 仅在发生细化时刷新
            print(">>> Refreshing case executions after UT refine (case only)", flush=True)  # 刷新日志
            run_all_executions_generate(data, args)  # 只执行 case

        print(">>> Step 2/7: Running UT attack and code refinement", flush=True)  # 步骤日志
        data = ut_attack_code(data, runner, tokenizer, args, round_num=round_idx + 1)  # Step2: UT 攻击与修复
        print(">>> Step 3/7: Executing refined codes + UT regeneration", flush=True)  # 步骤日志
        data = fix_code_attack_ut(data, runner, tokenizer, args, round_num=round_idx + 1)  # Step2: 重新生成坏 UT

        print(">>> Step 4/7: Propagating refined results back to data", flush=True)  # 步骤日志
        data = _propagate_refined_results(data)  # Step3: 将修复结果写回主字段

        print(">>> Step 5/7: Re-running executions to refresh bool table", flush=True)  # 步骤日志
        data = run_all_executions_generate(data, args)  # 刷新布尔矩阵（不跑 random UT）

        print(">>> Step 6/7: Regenerating all-pass UTs", flush=True)
        data, did_all_pass = regenerate_all_pass_ut(
            data, runner, tokenizer, args, round_num=round_idx + 1
        )
        print(">>> Step 7/7: Re-running executions after all-pass UT regen", flush=True)
        data = run_all_executions(data, args, skip_random_ut=True, compute_new_bon=False)
        
        # 在检查收敛前，保存本轮执行结束后的矩阵到历史
        for data_i in data:
            if data_i.get("skip_self_play", False):  # 已跳过的不需要保存
                continue
            bool_table = data_i.get("case_bool_table")
            if bool_table is not None:
                # 初始化历史列表（如果不存在）
                if "bool_table_history" not in data_i:
                    data_i["bool_table_history"] = []
                # 深拷贝矩阵并添加到历史列表
                data_i["bool_table_history"].append([row[:] for row in bool_table])
                # 只保留最近3轮
                if len(data_i["bool_table_history"]) > 3:
                    data_i["bool_table_history"] = data_i["bool_table_history"][-3:]
        
        # 检测收敛条件：1) 有代码全通过 2) 连续3轮矩阵完全相同
        # for idx, data_i in enumerate(data):
        #     if data_i.get("skip_self_play", False):  # 已经标记过的跳过
        #         continue
        #     bool_table = data_i.get("case_bool_table")
        #     if bool_table is None or len(bool_table) == 0:
        #         continue
            
            # 条件1: 检查是否有代码行全部为True（全通过所有UT）
            # for c_idx, row in enumerate(bool_table):
            #     if all(row):  # 该代码通过了所有UT
            #         data_i["skip_self_play"] = True
            #         data_i["skip_self_play_reason"] = f"Code {c_idx} passed all UTs in round {round_idx + 1}"
            #         data_i["converged_round"] = round_idx + 1  # 记录收敛轮数
            #         print(
            #             f"    >>> Problem {idx}: Code {c_idx} passed all {len(row)} UTs. "
            #             f"Skipping this problem in future iterations.",
            #             flush=True,
            #         )
            #         break  # 找到一个全通过的代码就足够了
            
            # 条件2: 检查矩阵是否连续三轮完全相同（收敛）
            # if not data_i.get("skip_self_play", False):  # 如果还没被标记跳过
            #     history = data_i.get("bool_table_history", [])
            #     # 需要至少有3轮历史才能判断连续3轮相同
            #     if len(history) >= 3:
            #         # 检查最后3轮（history[-3], history[-2], history[-1]）是否完全相同
            #         def matrices_equal(mat1, mat2):
            #             """检查两个矩阵是否完全相同"""
            #             if len(mat1) != len(mat2):
            #                 return False
            #             for row1, row2 in zip(mat1, mat2):
            #                 if len(row1) != len(row2):
            #                     return False
            #                 for val1, val2 in zip(row1, row2):
            #                     if bool(val1) != bool(val2):
            #                         return False
            #             return True
                    
            #         # 检查最后3轮是否完全相同
            #         if matrices_equal(history[-3], history[-2]) and matrices_equal(history[-2], history[-1]):
            #             data_i["skip_self_play"] = True
            #             data_i["skip_self_play_reason"] = f"Matrix converged (unchanged for 3 consecutive rounds) in round {round_idx + 1}"
            #             data_i["converged_round"] = round_idx + 1  # 记录收敛轮数
            #             print(
            #                 f"    >>> Problem {idx}: Bool matrix unchanged for 3 consecutive rounds. "
            #                 f"No progress detected, skipping future iterations.",
            #                 flush=True,
            #             )
        
        # 本轮仅执行一次 random UT 聚类（new_bon），避免重复 512
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
        _log_problem_bool_matrix(  # 打印该轮完成后的布尔矩阵
            data,
            0,
            f"Round {round_idx + 1} final bool matrix for Problem 0",
        )
        _record_round_bool_history(data, round_idx + 1)  # 记录轮次快照

        print(">>> Dumping round snapshot", flush=True)
        sync_usage_to_data(args, data)
        _dump_round_data(data, args, outputs_name, round_idx + 1)  # Step5: 保存本轮的完整快照
        _clear_round_transient_fields(data)
        
        # 打印和写入每轮的 resample 统计
        resample_stats_text = _format_resample_stats(data, round_label=f"Round {round_idx + 1}")
        print(resample_stats_text, flush=True)
        _append_to_results_txt(args, outputs_name, resample_stats_text)

    if own_runner:  # 若在本函数中创建了 runner，则负责关闭
        runner.close()
        print(">>> Self-play ModelRunner closed", flush=True)

    remaining_random = _count_random_uts(data)
    per_prob = _count_random_uts_per_problem(data)
    print(f">>> Random UT remaining after self-play: {remaining_random}", flush=True)
    for pid, cnt in per_prob:
        print(f"    - Problem {pid}: random UT = {cnt}", flush=True)

    # 打印收敛统计信息
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
    
    # 打印最终 Resample 统计信息
    final_resample_stats = _format_resample_stats(data, round_label="Final Summary")
    print(final_resample_stats, flush=True)
    _append_to_results_txt(args, outputs_name, final_resample_stats)

    return data
