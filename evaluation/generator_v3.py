# generator.py
# 利用 prompt 调模型 + 整理生成结果
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
    get_stage3_prompt,
    get_stage3_critique_prompt,
    get_stage3_75_prompt,
    get_stage4_prompt,
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

# 从llm输出里解析出代码部分
def extract_code(full_output: str) -> str:
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = matches[-1].strip()
    else:
        code_output = "We can not extract the code in the output. "
    return code_output


def vote_ut_output_by_codes(
    failing_indices,                 # 自一致性校验失败的 UT 索引列表
    output_prompt_idx_to_case_idx,   # UT 索引到题目 ID 的映射
    parsed_inputs,                   # 已经解析好的 UT 输入字符串列表
    data,                            # 包含所有题目数据的列表
    args                             # 全局参数配置
):
    """
    对于 LLM 自一致性较低的 UT，通过运行该题目下的所有代码并取众数来确定输出。
    """
    # 如果没有需要投票的 UT，直接返回空映射
    if not failing_indices:
        return {}

    print(f"    - [Code-Voting] 发现 {len(failing_indices)} 个低一致性 UT，准备通过代码投票...", flush=True)
    voting_codes = []        # 待执行的代码列表
    voting_inputs = []       # 对应的输入列表
    voting_time_limits = []  # 对应的超时限制列表
    
    # 记录每个 failing UT 对应的任务在 voting_codes 列表中的起始和结束位置
    ut_task_ranges = []
    current_task_idx = 0
    
    # 遍历所有需要投票的 UT
    for i in failing_indices:
        idx = output_prompt_idx_to_case_idx[i] # 获取该 UT 对应的题目 ID
        ut_input = parsed_inputs[i]            # 获取该 UT 的输入内容
        # 获取该题目下所有成功解析出的代码（过滤掉解析失败的提示）
        codes = [c for c in data[idx]["generated_code"] if "We can not extract" not in c]
        
        # 如果该题目没有生成出任何有效代码，则无法投票，跳过
        if not codes:
            ut_task_ranges.append((current_task_idx, current_task_idx))
            continue
            
        # 获取该题目的运行时间限制（默认为 5 秒）
        time_limit = data[idx].get("test_time_limit", 1)
        
        # 将该 UT 输入与所有代码配对，加入待执行队列
        for code in codes:
            voting_codes.append(code)
            voting_inputs.append(ut_input)
            voting_time_limits.append(time_limit)
        
        # 记录该 UT 对应的代码执行任务范围
        ut_task_ranges.append((current_task_idx, current_task_idx + len(codes)))
        current_task_idx += len(codes)
    
    voted_results_map = {} # 存储投票结果的映射: ut_idx -> synthetic_output_str
    raw_voting_results_map = {} # 存储原始执行结果的映射: ut_idx -> list of execution_results

    # 如果有待执行的任务，则开始批量执行
    if voting_codes:
        print(f"    - [Code-Voting] 正在执行 {len(voting_codes)} 个代码任务...", flush=True)
        # 【关键步骤】调用 execution.py 中的函数，在沙盒中批量运行代码
        # 这里不是用 LLM runner，而是真实的 Python 解释器执行
        voting_results = run_scripts_with_chunk(
            voting_codes,
            voting_inputs,
            voting_time_limits,
            args.num_chunks,
            args.exe_verbose
        )
        
        # 遍历每个 UT 的执行结果，进行众数投票
        for idx_in_failing, (start, end) in enumerate(ut_task_ranges):
            # 如果该 UT 没有对应的执行任务，跳过
            if start == end:
                continue
                
            ut_idx = failing_indices[idx_in_failing] # 获取 UT 在原始列表中的索引
            results = voting_results[start:end]      # 获取该 UT 对应的所有代码运行结果
            
            # 记录原始执行结果
            raw_voting_results_map[ut_idx] = results
            
            # 过滤掉运行报错（Error）或超时（Timeout）的结果，只保留正常的标准输出
            valid_results = [r.strip() for r in results if r and "Error" not in r and "Timeout" not in r]
            
            # 如果有至少一个代码成功运行并产生了输出
            if valid_results:
                # 使用 Counter 统计所有输出出现的频率
                counts = Counter(valid_results)
                # 取出现次数最多的输出作为“众数”结果
                majority_output = counts.most_common(1)[0][0]
                
                # 构造一个符合 LLM 输出格式的字符串，方便后续的 extract_ut_output 函数统一解析
                synthetic_output = f"**Test Output:**\n```\n{majority_output}\n```\n\nExplanation:\n[Derived from code-based voting majority]"
                voted_results_map[ut_idx] = synthetic_output
                print(f"      > 题目 {output_prompt_idx_to_case_idx[ut_idx]}: 代码投票成功", flush=True)
            else:
                # 如果所有代码都运行失败，则该 UT 投票失败
                print(f"      > 题目 {output_prompt_idx_to_case_idx[ut_idx]}: 代码投票失败 (无有效执行结果)", flush=True)
                
    return voted_results_map, raw_voting_results_map

# 对解析出来的代码 / 测试用例做一些后处理
def modify(c):
    c = c.replace("plaintext\n", "")
    c = c.replace("\\n", "\n")
    if not c.endswith("\n"):
        c += "\n"
    return c

def _cut_at_first_marker(content: str, markers: tuple[str, ...]) -> str:
    """
    Cut content at the earliest occurrence of any marker (case-insensitive).
    """
    lower = content.lower()
    cut_positions = [lower.find(m) for m in markers if m in lower]
    return content[: min(cut_positions)] if cut_positions else content

def _clean_ut_output_text(content: str) -> str:
    """
    Normalize UT output text by removing common trailing prompt artifacts.
    """
    if not content:
        return content
    # For output, cut at the last blank-line separator, but still respect earlier markers.
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
    # Strip trailing "Let's think step by step." artifacts.
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
    """
    Normalize UT input text by removing trailing prompt artifacts.
    """
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

# 从llm输出里解析出测试用例部分
def extract_test_cases(full_output):
    # First, try extracting with the updated triple-backtick pattern
    pattern_input_backticks = r"\*\*Test Input:\*\*\s*```(.*?)```"
    pattern_output_backticks = r"\*\*Test Output:\*\*\s*```(.*?)```"
    matches_input = re.findall(
        pattern_input_backticks, full_output, re.DOTALL | re.IGNORECASE
    )
    matches_output = re.findall(
        pattern_output_backticks, full_output, re.DOTALL | re.IGNORECASE
    )

    # For Test Input: either use the updated triple-backtick version or fallback to plain text
    if matches_input:
        raw_input = _clean_ut_input_text(matches_input[-1].lstrip("\n"))
        test_input = [modify(raw_input)]
    else:
        # Fallback pattern without backticks: capture until **Test Output:**
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

    # Also extract from the last occurrence of **Test Input:** to the end
    example_text = []
    input_matches = list(re.finditer(r"\*\*Test Input:\*\*", full_output, re.IGNORECASE))
    if input_matches:
        example_text = [full_output[input_matches[-1].start():]]

    if example_text == [] or test_input == [] or test_output == []:
        return [], [], []

    return test_input, test_output, example_text


def get_token_lengths(strings, tokenizer):
    # 过滤掉 None 值，防止 tokenizer.encode 报错
    valid_strings = [s for s in strings if s is not None]
    return [len(tokenizer.encode(s, add_special_tokens=False)) for s in valid_strings]


def extract_ut_idea(full_output: str) -> list[str]:
    """
    Parses the numbered list of attack ideas from the model output.
    Returns a list of idea strings.
    """
    lines = full_output.strip().split('\n')
    ideas = []
    for line in lines:
        # Simple heuristic to find numbered list items
        if re.match(r'^\d+[\.\)]', line.strip()):
             ideas.append(line.strip())
    
    # If regex fails, maybe the whole output is one idea or different format
    if not ideas:
         ideas = [full_output.strip()]
    return [modify(idea) for idea in ideas]

def extract_ut_input(full_output: str) -> str:
    # 尝试匹配带标记的情况 (注意：这里必须是 Input)
    # 匹配 **Test Input:** 后面跟着的内容，直到 ** 或 字符串结尾
    pattern = r"\*\*Test [Ii]nput:\*\*\s*(.*?)(?:\s*\*\*|$)"
    matches = re.findall(pattern, full_output, re.DOTALL | re.IGNORECASE)
    if matches:
        content = matches[-1]
        # 去掉可能存在的代码块标记 ```
        content = re.sub(r"^```(?:[a-zA-Z]*\n)?", "", content)
        content = re.sub(r"```$", "", content).strip()
        content = _clean_ut_input_text(content)
        if _looks_like_prompt_artifact(content):
            return "We can not extract the input in the output. "
        if content:
            return modify(content)
        # 允许空输入（例如题面允许空字符串）
        return "\n"
    
    # 备选方案：如果没有标记，但有代码块，尝试取第一个代码块
    matches = re.findall(r"```(?:[a-zA-Z]*\n)?(.*?)```", full_output, re.DOTALL)
    if matches:
        content = _clean_ut_input_text(matches[0].strip())
        if _looks_like_prompt_artifact(content):
            return "We can not extract the input in the output. "
        if content:
            return modify(content)
    return "We can not extract the input in the output. "

def extract_ut_output(full_output: str) -> str:
    # 尝试匹配带标记的情况
    # 匹配 **Test Output:** 后面跟着的内容，直到 **Explanation:** 或 字符串结尾
    pattern = r"\*\*Test [Oo]utput:\*\*\s*(.*?)(?:\s*\*\*Explanation:\*\*|$)"
    matches = re.findall(pattern, full_output, re.DOTALL | re.IGNORECASE)
    if matches:
        content = matches[-1]
        # 去掉可能存在的代码块标记 ```
        content = re.sub(r"^```(?:[a-zA-Z]*\n)?", "", content)
        content = re.sub(r"```$", "", content).strip()
        content = _clean_ut_output_text(content)
        if _looks_like_prompt_artifact(content):
            return "We can not extract the output in the output. "
        if content:
            return modify(content)
        # 允许空输出（题面可能期望空行）
        return "\n"
    
    # 备选方案：如果没有标记，但有代码块，尝试取最后一个代码块
    matches = re.findall(r"```(?:[a-zA-Z]*\n)?(.*?)```", full_output, re.DOTALL)
    if matches:
        content = _clean_ut_output_text(matches[-1].strip())
        if _looks_like_prompt_artifact(content):
            return "We can not extract the output in the output. "
        if content:
            return modify(content)
    
    return "We can not extract the output in the output. "


def _strip_case_prefix(text: str) -> str:  # 去掉输入里的 CASE| 前缀
    if text is None:  # 空值直接返回空串
        return ""  # 返回空串
    raw = str(text)  # 统一转字符串
    if raw.lstrip().lower().startswith("case|"):  # 检测 CASE| 前缀
        raw = re.sub(r"^\s*case\|\s*", "", raw, flags=re.IGNORECASE)  # 移除前缀
    return raw  # 返回清理后的文本


def parse_random_case_inputs(full_output: str) -> list[str]:  # 解析 random UT input 列表
    """  # 说明起始
    Parse randomly generated UT inputs from the custom CASE| format.  # 功能描述
    优先匹配 CASE|```...```；若失败，匹配 CASE| 后的所有内容并去除反引号。  # 规则说明
    """  # 说明结束
    if not full_output:  # 空输出直接返回
        return []  # 返回空列表
    pattern = r"CASE\|\s*```(.*?)```"  # 匹配 CASE|```...``` 的模式
    matches = re.findall(pattern, full_output, re.DOTALL | re.IGNORECASE)  # 抽取所有匹配
    if matches:  # 如果匹配到内容
        cleaned = []  # 保存清洗后的输入
        for m in matches:  # 遍历匹配项
            m = m.strip()  # 去首尾空白
            if not m:  # 空内容跳过
                continue  # 跳过空项
            cleaned.append(_strip_case_prefix(modify(m)))  # 规范化 + 去 CASE|
        return cleaned  # 返回清洗结果

    # Fallback: 直接匹配 CASE| 后的全部内容（允许多段 CASE|）
    raw = str(full_output)  # 统一转字符串
    case_indices = [m.start() for m in re.finditer(r"CASE\|", raw, flags=re.IGNORECASE)]  # 找 CASE| 位置
    if case_indices:  # 有 CASE| 才处理
        cleaned = []  # 保存清洗后的输入
        for i, start in enumerate(case_indices):  # 遍历 CASE| 位置
            seg_start = start + len("CASE|")  # 跳过 CASE|
            seg_end = case_indices[i + 1] if i + 1 < len(case_indices) else len(raw)  # 到下一个 CASE| 或结尾
            segment = raw[seg_start:seg_end].strip()  # 取片段并去空白
            if not segment:  # 空段跳过
                continue  # 跳过空项
            segment = segment.replace("`", "")  # 去掉反引号
            segment = segment.strip()  # 再次去空白
            if not segment:  # 清理后为空则跳过
                continue  # 跳过空项
            cleaned.append(_strip_case_prefix(modify(segment)))  # 规范化 + 去 CASE|
        if cleaned:  # 有效结果返回
            return cleaned  # 返回结果

    # 最后兜底：通用解析 + 去反引号
    fallback_input = extract_ut_input(full_output)  # 尝试通用解析
    if not fallback_input:  # 兜底失败
        return []  # 返回空列表
    fallback_input = str(fallback_input).replace("`", "")  # 去反引号
    return [_strip_case_prefix(fallback_input)]  # 去 CASE| 后返回

# ======================= PlanSearch 专用工具 =======================

# 从llm输出里解析出观察并格式化为列表
# def parse_observations(text: str, max_obs: int | None = None):
#     """
#     解析形如：
#       1. xxx
#       2. yyy
#       3) zzz
#     的编号行，只保留内容部分。

#     如果 max_obs 不为 None，则最多只返回前 max_obs 条。
#     """
#     lines = [ln.strip() for ln in text.splitlines()]
#     obs = []
#     for ln in lines:
#         m = re.match(r'^(\d+[\.\)]\s+)(.+)$', ln)
#         if m:
#             obs.append(m.group(2).strip())
#             if max_obs is not None and len(obs) >= max_obs:
#                 break
#     return obs



def parse_observations(text: str, max_obs: int | None = None):
    """
    解析观察列表，支持新旧格式，并自动处理行首行尾空格。
    格式支持：
      1. [Observation 1]: 内容 (最稳健，允许括号内有空格)
      2. 1. 内容 或 1) 内容
    """
    # 1. splitlines() 切分行
    # 2. strip() 去除每一行前后的所有空白字符（空格、Tab等）
    lines = [ln.strip() for ln in text.splitlines()]
    
    obs = []
    
    # 优化后的正则：
    # \[\s*       -> 匹配 '[' 及其后可能的空格
    # Observation -> 匹配单词
    # \s+         -> 匹配中间的空格
    # \d+         -> 匹配数字
    # \s*\]       -> 匹配 ']' 及其前可能的空格
    # :\s*        -> 匹配冒号及其后可能的空格
    pattern_new = r'^\[\s*Observation\s+\d+\s*\]:\s*(.+)$'
    pattern_old = r'^\d+[\.\)]\s+(.+)$'

    for ln in lines:
        if not ln:
            continue

        content = None
        
        # 优先尝试新格式 (忽略大小写)
        m_new = re.match(pattern_new, ln, re.IGNORECASE)
        if m_new:
            content = m_new.group(1).strip()
        else:
            # 尝试旧格式
            m_old = re.match(pattern_old, ln)
            if m_old:
                content = m_old.group(1).strip()
        
        if content:
            obs.append(content)
            if max_obs is not None and len(obs) >= max_obs:
                break
                
    return obs

def parse_observations_stage2(text: str):
    """
    解析观察列表，支持新旧格式，并自动处理行首行尾空格。
    格式支持：
      1. [Observation 1]: 内容 (最稳健，允许括号内有空格)
      2. 1. 内容 或 1) 内容
    """
    # 1. splitlines() 切分行
    # 2. strip() 去除每一行前后的所有空白字符（空格、Tab等）
    lines = [ln.strip() for ln in text.splitlines()]
    
    obs = []
    
    # 优化后的正则：
    # \[\s*       -> 匹配 '[' 及其后可能的空格
    # Observation -> 匹配单词
    # \s+         -> 匹配中间的空格
    # \d+         -> 匹配数字
    # \s*\]       -> 匹配 ']' 及其前可能的空格
    # :\s*        -> 匹配冒号及其后可能的空格
    pattern_new = r'^\[\s*Observation\s+\d+\s*\]:\s*(.+)$'
    pattern_old = r'^\d+[\.\)]\s+(.+)$'

    for ln in lines:
        if not ln:
            continue

        content = None
        
        # 优先尝试新格式 (忽略大小写)
        m_new = re.match(pattern_new, ln, re.IGNORECASE)
        if m_new:
            content = m_new.group(1).strip()
        else:
            # 尝试旧格式
            m_old = re.match(pattern_old, ln)
            if m_old:
                content = m_old.group(1).strip()
        
        if content:
            obs.append(content)
                
    return obs

# 把一个 observation 子集 list 拼成一段可读文本，给后续 prompt 用
def format_observations(obs_list):
    """
    把一个 observation 子集 list 拼成一段可读文本，给后续 prompt 用。
    """
    if not obs_list:
        return "(no explicit observations provided; reason from the problem directly.)"
    return "\n".join(f"- {o}" for o in obs_list)

# 构造观察的子集
def build_observation_subsets(obs_list, max_subset_size=2, include_empty=True):
    """
    构造大小不超过 max_subset_size 的所有子集。
    按论文 PlanSearch：
      - 空集（可选）
      - 所有单元素
      - 所有二元组
    """
    subsets = []
    if include_empty:
        subsets.append([])

    n = len(obs_list)
    for size in range(1, min(max_subset_size, n) + 1):
        for combo in combinations(obs_list, size):
            subsets.append(list(combo))
    return subsets

# 打印每个阶段的prompt-output交互日志
def log_llm_interaction(stage_name, prompts, outputs, args, meta_info_list=None,):
    """
    用于打印清晰的 LLM 交互日志
    """
    # 是否开启打印交互日志
    if not args.verbose_logging:
        return
    # --------------------

    sep_thick = "=" * 80
    sep_thin = "-" * 80
    
    print(f"\n{sep_thick}", flush=True)
    print(f" [LOGGER] {stage_name} INTERACTION START | Batch Size: {len(prompts)}", flush=True)
    print(f"{sep_thick}\n", flush=True)

    for i, (p, o) in enumerate(zip(prompts, outputs)):
        # 获取题目 ID 以便追踪
        p_idx = "Unknown"
        if meta_info_list and i < len(meta_info_list):
            # 兼容 meta_info 是 dict (如 task) 或直接是 int (如 active_indices)
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

def run_generation_plansearch(
    data,
    case_generation_prompts,
    case_index,
    runner,
    tokenizer,
    args,
):
    """
    PlanSearch 分阶段代码生成（多题并行 + 筛选优化版）
    """
    
    num = len(data)
    usage_round_key = initial_round_key()

    print(
        "执行 PlanSearch 分阶段代码生成（多题并行版）: "
        "Stage1(一阶观察) → Stage2(二阶观察) → "
        "Stage3(解法) → Critique → [Shuffle & Select] → Stage4(代码)",
        flush=True,
    )

    # 每题的最终代码 / 自然语言解法 / 状态
    codes_per_problem = [[] for _ in range(num)]          # 存的是已经 wrap 过的 ```python 块
    solution_plans_per_problem = [[] for _ in range(num)] # 自然语言解法（包含批判解）

    # 记录每个题目下，所有 (观察子集 + 自然语言解法 + 对应代码) 的详细信息
    plansearch_records_per_problem = [[] for _ in range(num)]

    failed_stage1_rounds = [0] * num
    problem_halted = [False] * num 
    global_round = 0

    while True:
        # ================= 终止条件检查 =================
        if global_round >= args.max_global_rounds:
            print(
                f"\n[PlanSearch] 已达到最大轮数 max_global_rounds={args.max_global_rounds}，停止 PlanSearch",
                flush=True,
            )
            break

        # 筛选活跃题目：未被暂停 且 代码数量未达标
        active_indices = [
            i
            for i in range(num)
            if (not problem_halted[i]) and (len(codes_per_problem[i]) < args.k_code)
        ]

        if not active_indices:
            print("\n[PlanSearch] 所有题目已满足 k_code 需求或被停止，结束任务", flush=True)
            break

        global_round += 1
        print(
            f"\n[PlanSearch] Global Round {global_round}, 活跃题目数: {len(active_indices)}",
            flush=True,
        )

        # ==============================================================================
        # Stage 1: 生成一阶观察 (First-Order Observations)
        # ==============================================================================
        stage1_prompts = [
            data[idx]["code_generation_prompts"]["stage1"] for idx in active_indices
        ]
        print(f"  [Stage1] 为 {len(stage1_prompts)} 道题生成一阶观察...", flush=True)
        
        stage1_outputs = runner.generate(stage1_prompts)
        
        # --- LOGGING STAGE 1 ---
        log_llm_interaction(f"Stage 1 (Round {global_round})", stage1_prompts, stage1_outputs, args, active_indices)
        # -----------------------

        # 解析并存储一阶观察
        first_order_obs = {}  # idx -> list of obs_strings
        stage1_items_by_call = []
        
        for idx, out in zip(active_indices, stage1_outputs):
            data[idx]["stage1_observations_raw"] = out
            obs_list = parse_observations(out, max_obs=args.max_obs)
            data[idx]["stage1_observations_list"] = obs_list
            stage1_items_by_call.append(obs_list)
            
            if not obs_list:
                failed_stage1_rounds[idx] += 1
                print(
                    f"    [WARN] 题目 {idx} 本轮未解析出一阶观察 (fail count: {failed_stage1_rounds[idx]})",
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
            print("  [Stage1] 本轮全军覆没（无有效观察），进入下一轮重试", flush=True)
            continue
        
        if args.ablation != "only_stage1" and args.ablation != "only_stage2":
            # ==============================================================================
            # Stage 2: 构造子集 -> 生成二阶观察 (Second-Order Observations)
            # ==============================================================================
            stage2_tasks = []
            for idx, obs_list in first_order_obs.items():
                c1_subsets = build_observation_subsets(obs_list, max_subset_size=2, include_empty=args.is_empty)
                for c1 in c1_subsets:
                    stage2_tasks.append({
                        "idx": idx,
                        "first_obs_for_branch": c1,
                    })

            if not stage2_tasks:
                print("  [Stage2] 无法构造一阶子集，跳过本轮", flush=True)
                continue

            print(f"  [Stage2] 对 {len(stage2_tasks)} 条路径生成二阶观察...", flush=True)

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

            # --- LOGGING STAGE 2 ---
            log_llm_interaction(f"Stage 2 (Round {global_round})", stage2_prompts, stage2_outputs, args, stage2_tasks)
            # -----------------------

            # 解析二阶观察，构建叶子节点任务
            leaf_tasks = [] 
            # 记录本轮已生成的任务数，避免生成过多
            tasks_count_per_problem = {idx: 0 for idx in active_indices}
            stage2_items_by_call = []
            
            for task, out in zip(stage2_tasks, stage2_outputs):
                idx = task["idx"]
                if args.use_all_second_order_obs:
                    second_order_obs_list = parse_observations_stage2(out)
                else:
                    second_order_obs_list = parse_observations(out, max_obs=args.max_obs)
                stage2_items_by_call.append(second_order_obs_list)
                
                # 计算还需要多少个代码
                needed = args.k_code - len(codes_per_problem[idx])
                # 如果本轮已经积攒了足够的任务，就跳过后续的
                if tasks_count_per_problem[idx] >= needed:
                    continue

                if args.use_all_second_order_obs:
                    # 直接使用每个二阶观察，不构建子集
                    if not second_order_obs_list:
                        continue 
                    for second_obs in second_order_obs_list:
                        if tasks_count_per_problem[idx] >= needed:
                            break
                            
                        leaf_tasks.append({
                            "idx": idx,
                            "first_obs_for_branch": task["first_obs_for_branch"],
                            "second_obs_for_leaf": second_obs, 
                            "first_obs_for_leaf_str": task["first_subset_text"], 
                            "second_obs_for_leaf_str":second_obs  # 直接使用单个观察
                        })
                        tasks_count_per_problem[idx] += 1
                else:
                    # 原始逻辑：构建二阶观察子集
                    if not second_order_obs_list:
                        continue 
                
                    c2_subsets = build_observation_subsets(second_order_obs_list, max_subset_size=2)
                    
                    for c2 in c2_subsets:
                        if tasks_count_per_problem[idx] >= needed:
                            break
                            
                        leaf_tasks.append({
                            "idx": idx,
                            "first_obs_for_branch": task["first_obs_for_branch"],
                            "second_obs_for_leaf": c2,
                            "first_obs_for_leaf_str": task["first_subset_text"], 
                            "second_obs_for_leaf_str": format_observations(c2)
                        })
                        tasks_count_per_problem[idx] += 1
            reserve_item_usage(
                args,
                [task["idx"] for task in stage2_tasks],
                stage2_prompts,
                stage2_outputs,
                "stage2_observation",
                usage_round_key,
                stage2_items_by_call,
            )

            if not leaf_tasks:
                print("  [WARN] 本轮未生成任何有效的叶子路径 (C1->C2)，跳过本轮", flush=True)
                continue

            # ==============================================================================
            # Stage 3: 生成自然语言解法 (Solution Plans)
            # ==============================================================================
            print(f"  [Stage3] 对 {len(leaf_tasks)} 个叶子生成 Solution Plans...", flush=True)

            stage3_prompts = []
            for leaf in leaf_tasks:
                idx = leaf["idx"]
                problem_template = data[idx]["code_generation_prompts"]["stage3_template"]
                
                if args.use_first_order_obs:
                    prompt_obs1_str = leaf["first_obs_for_leaf_str"]
                else:
                    prompt_obs1_str = ""

                p = get_stage3_prompt(
                    problem_template,
                    prompt_obs1_str,
                    leaf["second_obs_for_leaf_str"],
                    args.system_prompts_stage3,
                )
                stage3_prompts.append(p)

            stage3_outputs = runner.generate(stage3_prompts)
            record_direct_usage(
                args,
                [leaf["idx"] for leaf in leaf_tasks],
                stage3_prompts,
                stage3_outputs,
                "stage3_solution_plan",
                usage_round_key,
            )
            
            # --- LOGGING STAGE 3 ---
            log_llm_interaction(f"Stage 3 - Solution (Round {global_round})", stage3_prompts, stage3_outputs, args, leaf_tasks)
            # -----------------------

            for leaf, sol in zip(leaf_tasks, stage3_outputs):
                leaf["solution_plan"] = sol
                idx = leaf["idx"]
                solution_plans_per_problem[idx].append(sol)

                record_solution = {
                    "global_round": global_round,
                    "first_obs_for_branch": leaf["first_obs_for_branch"],
                    "second_obs_for_leaf": leaf["second_obs_for_leaf"],
                    "first_obs_for_leaf_str": leaf["first_obs_for_leaf_str"],
                    "second_obs_for_leaf_str": leaf["second_obs_for_leaf_str"],
                    "plan_type": "solution",
                    "plan_text": sol,
                    "pseudocode": None,
                    "codes": [],
                    "use_pseudocode_module": args.use_pseudocode_module,
                    "reflection_visibility": args.reflection_visibility,
                    "use_first_order_obs": args.use_first_order_obs,
                }
                plansearch_records_per_problem[idx].append(record_solution)
                leaf["solution_record"] = record_solution

            # ==============================================================================
            # Stage 3.5: 生成批判与替代解法 (Critique Plans)
            # ==============================================================================
            if args.use_critique_plan:
                print(f"  [Stage3.5] 对 {len(leaf_tasks)} 个叶子生成 Critique Plans...", flush=True)

                stage3_crit_prompts = []
                for leaf in leaf_tasks:
                    idx = leaf["idx"]
                    problem_template = data[idx]["code_generation_prompts"]["stage3_template"]

                    if args.use_first_order_obs:
                        prompt_obs1_str = leaf["first_obs_for_leaf_str"]
                    else:
                        prompt_obs1_str = ""

                    if args.reflection_visibility:
                        obs1_for_critique = prompt_obs1_str
                        obs2_for_critique = leaf["second_obs_for_leaf_str"]
                    else:
                        obs1_for_critique = ""
                        obs2_for_critique = ""

                    p = get_stage3_critique_prompt(
                        problem_template,
                        obs1_for_critique,
                        obs2_for_critique,
                        leaf["solution_plan"],
                        args.system_prompts_stage3_critique,
                    )
                    stage3_crit_prompts.append(p)

                stage3_crit_outputs = runner.generate(stage3_crit_prompts)
                record_direct_usage(
                    args,
                    [leaf["idx"] for leaf in leaf_tasks],
                    stage3_crit_prompts,
                    stage3_crit_outputs,
                    "stage3_critique_plan",
                    usage_round_key,
                )
                
                # --- LOGGING STAGE 3.5 ---
                log_llm_interaction(f"Stage 3.5 - Critique (Round {global_round})", stage3_crit_prompts, stage3_crit_outputs, args, leaf_tasks)
                # -------------------------

                for leaf, crit in zip(leaf_tasks, stage3_crit_outputs):
                    leaf["crit_plan"] = crit
                    idx = leaf["idx"]
                    solution_plans_per_problem[idx].append(crit)

                    record_crit = {
                        "global_round": global_round,
                        "first_obs_for_branch": leaf["first_obs_for_branch"],
                        "second_obs_for_leaf": leaf["second_obs_for_leaf"],
                        "first_obs_for_leaf_str": leaf["first_obs_for_leaf_str"],
                        "second_obs_for_leaf_str": leaf["second_obs_for_leaf_str"],
                        "plan_type": "critique",
                        "plan_text": crit,
                        "pseudocode": None,
                        "codes": [],
                        "use_pseudocode_module": args.use_pseudocode_module,
                        "reflection_visibility": args.reflection_visibility,
                        "use_first_order_obs": args.use_first_order_obs,
                    }
                    plansearch_records_per_problem[idx].append(record_crit)
                    leaf["critique_record"] = record_crit
            else:
                print(f"  [Stage3.5] Critique Plan 已禁用，跳过本阶段", flush=True)

            # ==============================================================================
            # Stage 3.75:生成伪代码 (Solution Plan -> Pseudocode)

            if args.use_pseudocode_module:
                print(f"  [Stage3.75] 准备生成伪代码: 收集所有 自然语言解法 ...", flush=True)
                stage3_75_solution_prompts = []
                stage3_75_critique_prompts = []

                for leaf in leaf_tasks:
                    idx = leaf["idx"]
                    problem_template = data[idx]["code_generation_prompts"]["stage3_75_template"]

                    if "solution_plan" in leaf:
                        p = get_stage3_75_prompt(
                            problem_template,
                            leaf["solution_plan"],
                            args.system_prompts_stage3_75,
                        )
                        stage3_75_solution_prompts.append((leaf, p))

                    if "crit_plan" in leaf:
                        p = get_stage3_75_prompt(
                            problem_template,
                            leaf["crit_plan"],
                            args.system_prompts_stage3_75,
                        )
                        stage3_75_critique_prompts.append((leaf, p))

                solution_count = len(stage3_75_solution_prompts)
                all_pseudocode_tasks_meta = [l[0] for l in stage3_75_solution_prompts] + [l[0] for l in stage3_75_critique_prompts]
                all_pseudocode_prompts = [p for _, p in stage3_75_solution_prompts] + [p for _, p in stage3_75_critique_prompts]
                
                pseudocode_outputs = runner.generate(all_pseudocode_prompts)
                record_direct_usage(
                    args,
                    [leaf["idx"] for leaf in all_pseudocode_tasks_meta],
                    all_pseudocode_prompts,
                    pseudocode_outputs,
                    "stage3_75_pseudocode",
                    usage_round_key,
                )
                
                # --- LOGGING STAGE 3.75 ---
                log_llm_interaction(f"Stage 3.75 - Pseudocode (Round {global_round})", all_pseudocode_prompts, pseudocode_outputs, args, all_pseudocode_tasks_meta)
                # --------------------------

                for i, (leaf, _) in enumerate(stage3_75_solution_prompts):
                    pseudocode_text = pseudocode_outputs[i]
                    leaf["pseudocode"] = pseudocode_text
                    if "solution_record" in leaf:
                        leaf["solution_record"]["pseudocode"] = pseudocode_text
                
                for i, (leaf, _) in enumerate(stage3_75_critique_prompts):
                    pseudocode_text = pseudocode_outputs[solution_count + i]
                    leaf["crit_pseudocode"] = pseudocode_text
                    if "critique_record" in leaf:
                        leaf["critique_record"]["pseudocode"] = pseudocode_text
                
                print(f"  [Stage3.75]  生成了 {len(pseudocode_outputs)} 段伪代码", flush=True)

            # ==============================================================================
            # Stage 4: 翻译为代码 (Plan -> Code)
            # ==============================================================================
            # 1. 收集所有 伪代码 并进行随机筛选
            if args.use_pseudocode_module:
                print(f"  [Stage4] 准备生成代码: 收集所有 伪代码 并进行随机筛选...", flush=True)
                all_stage4_tasks = []
                for leaf in leaf_tasks:
                    idx = leaf["idx"]
                    problem_template = data[idx]["code_generation_prompts"]["stage2_template"]

                    if "pseudocode" in leaf:
                        p = get_stage4_prompt(
                            problem_template,
                            leaf["pseudocode"],
                            args.system_prompts_stage4,
                            args.special_requirements,
                        )
                        all_stage4_tasks.append({
                            "idx": idx,
                            "prompt": p,
                            "record": leaf.get("solution_record")
                        })

                    if "crit_pseudocode" in leaf:
                        p = get_stage4_prompt(
                            problem_template,
                            leaf["crit_pseudocode"],
                            args.system_prompts_stage4,
                            args.special_requirements,
                        )
                        all_stage4_tasks.append({
                            "idx": idx,
                            "prompt": p,
                            "record": leaf.get("critique_record")
                        })
            # 使用自然语言生成
            else:
                print(f"  [Stage4] 准备生成代码: 收集所有 自然语言解法 并进行随机筛选...", flush=True)
                all_stage4_tasks = []
                for leaf in leaf_tasks:
                    idx = leaf["idx"]
                    problem_template = data[idx]["code_generation_prompts"]["stage3_template"]

                    if "solution_plan" in leaf:
                        p = get_stage4_prompt(
                            problem_template,
                            leaf["solution_plan"],
                            args.system_prompts_stage4_ablation_only_stage,
                            args.special_requirements,
                        )
                        all_stage4_tasks.append({
                            "idx": idx,
                            "prompt": p,
                            "record": leaf.get("solution_record")
                        })
                    
                    if "crit_plan" in leaf:
                        p = get_stage4_prompt(
                            problem_template,
                            leaf["crit_plan"],
                            args.system_prompts_stage4_ablation_only_stage,
                            args.special_requirements,
                        )
                        all_stage4_tasks.append({
                            "idx": idx,
                            "prompt": p,
                            "record": leaf.get("critique_record")
                        })
            # 2. 按题目分组
            tasks_by_problem = {idx: [] for idx in active_indices}
            for task in all_stage4_tasks:
                if task["idx"] in tasks_by_problem:
                    tasks_by_problem[task["idx"]].append(task)

            # 3. 随机筛选
            final_stage4_prompts = []
            final_stage4_meta = []        # 记录 idx
            final_stage4_records = []     # 记录 record

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
                
                print(f"    - 题目 {idx}: 现有 {len(available_tasks)} 个想法, 选取 {len(selected_tasks)} 个生成代码", flush=True)

            if not final_stage4_prompts:
                print("  [Info] 没有任务需要执行 Stage 4 (可能已满足 k_code)，跳过", flush=True)
                continue
        # 只使用观察生成
        else:

            if args.ablation == "only_stage1":
                print(f"  [Stage4] 准备生成代码: 使用一阶观察直接生成（构造子集）...", flush=True)
                
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
                            "pseudocode": None,
                            "codes": [],
                            "use_pseudocode_module": args.use_pseudocode_module,
                            "reflection_visibility": args.reflection_visibility,
                            "use_first_order_obs": True,
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
                        
                        # 把记录加到 plansearch_records
                        if t["record"] is not None:
                            plansearch_records_per_problem[t["idx"]].append(t["record"])
                    
                    print(f"    - 题目 {idx}: 使用一阶观察子集, 现有 {len(available_tasks)} 个想法, 选取 {len(selected_tasks)} 个生成代码", flush=True)
                
                if not final_stage4_prompts:
                    print("  [Info] 没有任务需要执行 Stage 4 (可能已满足 k_code)，跳过", flush=True)
                    continue
            
            elif args.ablation == "only_stage2":
                print(f"  [Stage4] 准备生成代码: 使用二阶观察直接生成（构造子集）...", flush=True)
                
                stage2_tasks = []
                for idx, obs_list in first_order_obs.items():
                    c1_subsets = build_observation_subsets(obs_list, max_subset_size=2)
                    for c1 in c1_subsets:
                        stage2_tasks.append({
                            "idx": idx,
                            "first_obs_for_branch": c1,
                        })

                if not stage2_tasks:
                    print("  [WARN] 无法构造一阶子集，跳过本轮", flush=True)
                    continue

                print(f"  [Stage2临时] 对 {len(stage2_tasks)} 条路径生成二阶观察...", flush=True)

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
                
                log_llm_interaction(f"Stage 2 临时 (Round {global_round})", stage2_prompts, stage2_outputs, args, stage2_tasks)

                all_stage4_tasks = []
                
                for task, out in zip(stage2_tasks, stage2_outputs):
                    idx = task["idx"]
                    second_order_obs_list = parse_observations_stage2(out)
                    stage2_items_by_call.append(second_order_obs_list)

                    # Save observations to data for UT generation
                    if data[idx].get("stage2_observations") is None:
                        data[idx]["stage2_observations"] = ""
                    if data[idx].get("stage2_observations_list") is None:
                        data[idx]["stage2_observations_list"] = []
                    
                    # Aggregate the list form for granular UT generation
                    data[idx]["stage2_observations_list"].extend(second_order_obs_list)

                    for obs in second_order_obs_list:
                        data[idx]["stage2_observations"] += f"- {obs}\n"
                    
                    if not second_order_obs_list:
                        continue
                    
                    # 使用所有二阶观察，而不是构建子集
                    
                    problem_template = data[idx]["code_generation_prompts"]["stage2_template"]
                    
                    for second_order_obs in second_order_obs_list:
                        second_obs_text = second_order_obs
                        record = {
                            "global_round": global_round,
                            "first_obs_for_branch": task["first_obs_for_branch"],
                            "second_obs_for_leaf": second_order_obs,  # 保存所有二阶观察
                            "first_obs_for_leaf_str": "",
                            "second_obs_for_leaf_str": second_obs_text,
                            "plan_type": "only_stage2",
                            "plan_text": second_obs_text,
                            "pseudocode": None,
                            "codes": [],
                            "use_pseudocode_module": args.use_pseudocode_module,
                            "reflection_visibility": args.reflection_visibility,
                            "use_first_order_obs": False,
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
                    
                    print(f"    - 题目 {idx}: 使用二阶观察子集, 现有 {len(available_tasks)} 个想法, 选取 {len(selected_tasks)} 个生成代码", flush=True)
                
                if not final_stage4_prompts:
                    print("  [Info] 没有任务需要执行 Stage 4 (可能已满足 k_code)，跳过", flush=True)
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
        # --- LOGGING STAGE 4 ---
        log_llm_interaction(f"Stage 4 - Code Gen (Round {global_round})", final_stage4_prompts, stage4_outputs, args, final_stage4_meta)
        # -----------------------
        
        # 5. 解析结果
        for full_output, idx, plan_record in zip(stage4_outputs, final_stage4_meta, final_stage4_records):
            code_text = extract_code(full_output)
            codes_per_problem[idx].append(code_text)
            if plan_record is not None:
                plan_record["codes"].append(code_text)
        
            

    # ==============================================================================
    # 汇总结果
    # ==============================================================================
    all_code_full_outputs = []
    
    for idx in range(num):
        data[idx]["solution_plans"] = solution_plans_per_problem[idx]
        data[idx]["num_codes_generated"] = len(codes_per_problem[idx])
        data[idx]["plansearch_plans_and_observations"] = plansearch_records_per_problem[idx]
        
        data[idx]["experiment_config"] = {
            "use_pseudocode_module": args.use_pseudocode_module,
            "reflection_visibility": args.reflection_visibility,
            "use_first_order_obs": args.use_first_order_obs,
            "max_obs": args.max_obs,
            "max_global_rounds": args.max_global_rounds,
            "prompt_role_mode": args.prompt_role_mode,
        }

        status = "OK" if len(codes_per_problem[idx]) >= args.k_code else "WARN"
        print(f"  [{status}] 题目 {idx} 最终代码数: {len(codes_per_problem[idx])}", flush=True)

        for code_str in codes_per_problem[idx]:
            all_code_full_outputs.append(code_str) 
            data[idx]["full_code_generation"].append(code_str)
            data[idx]["generated_code"].append(code_str)

    code_generation_result = all_code_full_outputs

    print(f"\n✓ PlanSearch 结束，总共生成 {len(code_generation_result)} 段代码。", flush=True)

    code_response_length = get_token_lengths(code_generation_result, tokenizer)
    mean_code = sum(code_response_length) / len(code_response_length) if code_response_length else 0

    case_generation_result = []
    mean_case = 0

    if args.eval_pass_at_k_only:
        print("\n[PlanSearch] eval_pass_at_k_only=True，跳过测试用例生成", flush=True)
    elif args.eval_bon:
        print("\n[PlanSearch] 开始为所有题目生成测试用例（由 k_case 控制数量）", flush=True)
        data, case_generation_result, mean_case = generate_unit_tests_for_dataset(
            data,
            case_generation_prompts,
            case_index,
            runner,
            tokenizer,
            args,
        )

    return data, code_generation_result, case_generation_result, mean_code, mean_case



# ======================= 一体化生成逻辑（原逻辑） =======================

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
    """
    一体化提示词模式,即原本的代码生成逻辑
    """
    print("开始推理 (一体化模式)...", flush=True)
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
        f"✓ 生成代码 {len(code_generation_result)} 条, 生成测试 {len(case_generation_result)} 条",
        flush=True,
    )

    # ========= 计算 response length =========
    code_response_length = get_token_lengths(code_generation_result, tokenizer)
    case_response_length = (
        get_token_lengths(case_generation_result, tokenizer)
        if len(case_generation_result) > 0
        else []
    )
    mean_code = sum(code_response_length) / len(code_response_length)
    mean_case = sum(case_response_length) / len(case_response_length) if case_response_length else 0

    # ========= 把生成结果挂回 data =========
    # process generated codes
    i = 0
    for full_output in code_generation_result:
        code_output = extract_code(full_output)
        index_i = code_index[i]
        data[index_i]["full_code_generation"].append(full_output)
        data[index_i]["generated_code"].append(code_output)
        i += 1

    # process generated unit tests
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

def generate_unit_tests_for_dataset(
    data,
    case_generation_prompts,
    case_index,
    runner,
    tokenizer,
    args,
):
    """
    单独负责：
      - 根据 case_generation_prompts 调模型生成 UT
      - 把结果写回 data[i]["full_case_generation"] / ["case_input"] / ["case_output"] / ["case_text"]
      - 打印和之前完全一样的日志
      - 返回 case_generation_result 和 mean_case
    """
    usage_round_key = initial_round_key()
    # ---------- 测试用例生成（和原逻辑一致） ----------
    # 如果仅评估 pass@k (args.eval_pass_at_k_only 为 True)，则跳过测试用例生成步骤
    if args.use_idea_attack_ut == False:
        if args.eval_pass_at_k_only:
            case_generation_result = []
            print("\n=== pass@k-only 模式：跳过生成测试用例 ===", flush=True)
            mean_case = 0
            return data, case_generation_result, mean_case
        print("\n=== 生成测试用例 ===", flush=True)  # 打印开始生成测试用例的日志，并强制刷新缓冲区
        all_case_prompts = case_generation_prompts  # 获取所有待生成的测试用例提示词
        N_case = len(all_case_prompts)  # 计算提示词的总数量
        
        # 为了避免模型生成时的顺序偏差（或仅仅是为了随机化批处理顺序），对提示词进行打乱
        indices_case = list(range(N_case))  # 创建一个从 0 到 N_case-1 的索引列表
        shuffled_idx_case = indices_case[:]  # 复制一份索引列表，避免修改原列表
        random.shuffle(shuffled_idx_case)  # 随机打乱索引列表的顺序
        shuffled_case_prompts = [all_case_prompts[i] for i in shuffled_idx_case]  # 根据打乱后的索引重新排列提示词列表

        # 调用模型生成测试用例
        shuffled_case_outputs = runner.generate(shuffled_case_prompts)  # 使用模型运行器批量生成测试用例结果
        record_direct_usage(
            args,
            [case_index[i] for i in shuffled_idx_case],
            shuffled_case_prompts,
            shuffled_case_outputs,
            "ut_generation_one_shot",
            usage_round_key,
        )
        print("第一个prompt: ", shuffled_case_prompts[0])
        print("第一个输出：", shuffled_case_outputs[0])

        # 将打乱顺序生成的结果恢复到原始顺序
        restored_case_outputs = [None] * N_case  # 初始化一个长度为 N_case 的列表，用于存放恢复顺序后的结果
        for out, idx2 in zip(shuffled_case_outputs, shuffled_idx_case):  # 遍历打乱后的输出和对应的原始索引
            restored_case_outputs[idx2] = out  # 将输出放回其原始索引对应的位置
        case_generation_result = restored_case_outputs  # 将恢复顺序后的结果赋值给 case_generation_result
        print("✓ 测试生成完成", flush=True)  # 打印测试用例生成完成的日志，并强制刷新缓冲区

        # ========= 把 UT 结果挂回 data =========
        # 将生成的测试用例结果解析并回填到 data 结构中
        i = 0  # 初始化索引计数器
        for full_output in case_generation_result:  # 遍历每一个生成的测试用例完整输出
            # 提取测试输入、输出和示例文本
            test_input, test_output, example_text = extract_test_cases(full_output)  # 从模型输出中提取结构化的测试用例数据
            index_i = case_index[i]  # 获取当前测试用例对应的题目索引
            
            # 将完整生成结果存入 data
            data[index_i]["full_case_generation"].append(full_output)  # 将原始的完整输出保存到对应题目的数据中
            # 将解析出的输入输出追加到对应字段
            data[index_i]["case_input"] += test_input  # 将提取的测试输入追加到对应题目的输入列表中
            data[index_i]["case_output"] += test_output  # 将提取的测试输出追加到对应题目的输出列表中
            data[index_i]["case_text"] += example_text  # 将提取的示例文本追加到对应题目的文本列表中
            
            # 在标准模式下，默认所有提取出来的 UT 都是有效的
            data[index_i]["case_is_valid"] += [True] * len(test_input)
            
            i += 1  # 索引计数器加 1

        # ========= 计算 mean_case =========
        # 计算生成测试用例的平均 token 长度，用于统计
        case_response_length = get_token_lengths(case_generation_result, tokenizer)  # 计算所有生成结果的 token 长度
        mean_case = (  # 计算平均长度
            sum(case_response_length) / len(case_response_length)  # 总长度除以数量
            if case_response_length  # 如果列表不为空
            else 0  # 如果列表为空，则平均长度为 0
        )
    elif args.use_idea_attack_ut:
        print("\n=== 生成测试用例 (Idea Attack Mode) ===", flush=True)
        total_k_case = getattr(args, "k_case", 0)
        if total_k_case <= 0:
            print("  [WARN] k_case=0，在 Idea Attack 模式下无需生成 UT", flush=True)
            case_generation_result = []
            mean_case = 0
            return data, case_generation_result, mean_case

        idea_target = total_k_case // 2
        random_target = total_k_case - idea_target
        idea_candidates_per_problem = idea_target * 2 if idea_target > 0 else 0
        random_candidates_per_problem = random_target * 2 if random_target > 0 else 0
        num_problems = len(data)

        print(
            f"    - [Idea Attack] 总计 {total_k_case} 个 UT: Idea-based={idea_target}, Random={random_target}",
            flush=True,
        )
        print(
            f"    - [Candidates] Idea 候选={idea_candidates_per_problem}/题, Random 候选={random_candidates_per_problem}/题",
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
                # 标记 UT 来源：random / idea
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
                    f"    - [Self-Consistency] 采样数量: {args.self_consistency_num}, 正在生成并投票...",
                    flush=True,
                )
                
                # 初始化 UT 状态追踪
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
                
                # 初始化 resample 统计（如果不存在）
                for idx in set(input_prompt_idx_to_case_idx):
                    data[idx].setdefault("ut_resample_stats", {"generator": 0, "self_play": 0})
                
                for attempt in range(max_resample_attempts):
                    # 统计每个题目已通过验证的 UT 数量
                    problem_success_count = {}
                    for i, st in ut_states.items():
                        if st["success"]:
                            prob_idx = st["prob_idx"]
                            problem_success_count[prob_idx] = problem_success_count.get(prob_idx, 0) + 1
                    
                    # 找出尚未成功的 UT，但如果所属题目已经凑够数量则跳过
                    pending_indices = []
                    for i, st in ut_states.items():
                        if st["success"]:
                            continue  # 已经成功的跳过
                        prob_idx = st["prob_idx"]
                        if problem_success_count.get(prob_idx, 0) >= target_per_problem:
                            # 该题目已经凑够了，不再为该题目的其他候选 UT 尝试
                            continue
                        pending_indices.append(i)
                    
                    if not pending_indices:
                        print(f"    - [Resample] 第 {attempt + 1} 轮：所有题目已凑够目标 UT 数量", flush=True)
                        break
                    
                    print(f"    - [Resample] 第 {attempt + 1}/{max_resample_attempts} 轮：尝试生成 {len(pending_indices)} 个低一致性 UT...", flush=True)
                    
                    # 构造扩展的 prompts（每个 UT 生成 self_consistency_num 次）
                    expanded_prompts = []
                    for i in pending_indices:
                        expanded_prompts.extend([ut_states[i]["prompt"]] * args.self_consistency_num)
                        ut_states[i]["attempts"] += 1
                    
                    # 批量生成
                    batch_outputs = runner.generate(expanded_prompts)
                    record_direct_usage(
                        args,
                        [ut_states[i]["prob_idx"] for i in pending_indices for _ in range(args.self_consistency_num)],
                        expanded_prompts,
                        batch_outputs,
                        "ut_output_generation",
                        usage_round_key,
                    )
                    
                    # 处理每个 UT 的生成结果
                    for idx_in_batch, i in enumerate(pending_indices):
                        start_idx = idx_in_batch * args.self_consistency_num
                        end_idx = start_idx + args.self_consistency_num
                        samples_raw = batch_outputs[start_idx:end_idx]
                        samples_extracted = [extract_ut_output(s) for s in samples_raw]
                        
                        # 保存样本
                        ut_states[i]["samples_raw"] = samples_raw
                        ut_states[i]["samples_extracted"] = samples_extracted
                        
                        # 计算一致性
                        valid_samples = [
                            s for s in samples_extracted if s.strip() and "We can not extract" not in s
                        ]
                        if valid_samples:
                            counts = Counter(valid_samples)
                            winner_extracted, unique_count = counts.most_common(1)[0]
                        else:
                            counts = Counter(samples_extracted)
                            winner_extracted, unique_count = counts.most_common(1)[0]
                        
                        # 找到 winner 对应的原始输出
                        try:
                            winner_idx = samples_extracted.index(winner_extracted)
                            winner_raw = samples_raw[winner_idx]
                        except ValueError:
                            winner_raw = samples_raw[0]
                        
                        ut_states[i]["consistency"] = unique_count
                        ut_states[i]["fallback"] = winner_raw
                        
                        # 检查是否通过一致性验证
                        if (
                            unique_count >= min_consistency_threshold
                            and winner_extracted
                            and "We can not extract" not in winner_extracted
                        ):
                            ut_states[i]["success"] = True
                            ut_states[i]["final_output"] = winner_raw
                            count_same[unique_count - 1] += 1
                            # 记录该 UT 在 generator 阶段的 resample 次数
                            prob_idx = ut_states[i]["prob_idx"]
                            data[prob_idx]["ut_resample_stats"]["generator"] += ut_states[i]["attempts"]
                        else:
                            # 记录统计，但不标记为成功
                            count_same[unique_count - 1] += 1
                    
                    # 统计本轮成功数量
                    newly_succeeded = sum(1 for i in pending_indices if ut_states[i]["success"])
                    print(f"      > 第 {attempt + 1} 轮：{newly_succeeded}/{len(pending_indices)} 个 UT 通过验证", flush=True)
                
                # 最后处理：标记失败的 UT 为无效
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
                        # 经过最大重试次数后仍未通过，标记为 None（无效）
                        final_outputs.append(None)
                        prob_idx = st["prob_idx"]
                        # print(f"      > [屏蔽] 题目 {prob_idx} 的 UT 在 {max_resample_attempts} 次尝试后仍未通过一致性验证，已屏蔽", flush=True)

                print("count_same=", count_same)
                print("全部一致的个数是", count_same[-1])
                print("全部一致的百分比是", count_same[-1] / len(output_prompts))
                print("全部不一致的个数是", count_same[0])
                print("全部不一致的百分比是", count_same[0] / len(output_prompts))
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
                # 若输入解析失败，直接标记无效（避免占位符被当作有效 UT）
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

            # 不再进行低一致性补齐，因为已经在 resample 阶段尝试过多次
            # 直接统计每个题目的有效 UT 数量
            for idx, record_list in records_by_problem.items():
                valid_count = sum(1 for rec in record_list if rec["is_valid"])
                if valid_count < target_per_problem:
                    if max_resample_attempts > 0:
                        retry_note = f"已经过 {max_resample_attempts} 轮 resample"
                    else:
                        retry_note = "未启用 self-consistency resample"
                    print(
                        f"    - [WARN] 题目 {idx}: 有效 UT 仅 {valid_count}/{target_per_problem}（{retry_note}）",
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

                        # print(
                        #     f"    - 题目 {idx}: 使用二阶观察子集, 现有 {num_obs} 个观察, 目标生成 {total_ideas} 个 Attack Ideas (本次使用观察索引: {list(ideas_per_obs_map.keys())})",
                        #     flush=True,
                        # )
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
                    elif mode == "natural_desciption":
                        input_mode = data_i.get("solution_plan", "")
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

                # print(
                #     f"    - 题目 {idx}: 从输出中解析出 {len(current_ideas)} 个 Attack Ideas",
                #     flush=True,
                # )
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
                # print(f"    - 题目 {idx}: 解析出 UT Input, 准备生成 Output", flush=True)

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
            # 和 idea-based 保持一致：生成更多候选以应对自一致性验证的失败
            case_index_for_generation = build_case_index(random_candidates_per_problem)
            if not case_index_for_generation:
                return []

            print("--- Random UT: Generating Inputs ---", flush=True)
            placeholder = getattr(  # random UT 占位符（用于过滤无效输入）
                args, "random_ut_placeholder", "We can not extract the input in the output. "  # 默认占位符
            )  # 占位符获取结束
            random_prompts = []  # 随机 UT 输入 prompt 列表
            random_prompt_idx_to_case_idx = []  # prompt 对应题目索引
            for idx in case_index_for_generation:
                prompt_list = get_ut_input_random_generation_prompt(
                    problem=data[idx]["question"],
                    num_cases=1,
                )
                prompt_str = get_full_prompt(prompt_list)
                random_prompts.append(prompt_str)  # 收集 prompt
                random_prompt_idx_to_case_idx.append(idx)  # 记录对应题目

            random_outputs = runner.generate(random_prompts)  # 按原顺序批量生成随机输入
            record_direct_usage(
                args,
                random_prompt_idx_to_case_idx,
                random_prompts,
                random_outputs,
                "random_ut_input_generation",
                usage_round_key,
            )
            parsed_inputs = []  # 保存随机 UT 输入
            parsed_ideas = []  # 保存随机 UT 想法标签
            input_prompt_raw_outputs = []  # 保存输入阶段原始输出
            input_prompt_idx_to_case_idx = []  # 记录输入对应的题目索引

            for output, idx in zip(random_outputs, random_prompt_idx_to_case_idx):  # 按原顺序回填
                candidates = parse_random_case_inputs(output)  # 解析候选输入
                ut_input = candidates[0] if candidates else extract_ut_input(output)  # 解析输入
                ut_input = _strip_case_prefix(ut_input)  # 去 CASE| 前缀
                parsed_inputs.append(ut_input)  # 写入输入列表
                parsed_ideas.append("[Random Range Sampling]")  # 写入标签
                input_prompt_raw_outputs.append(output)  # 写入原始输出
                input_prompt_idx_to_case_idx.append(idx)  # 记录题目索引
                # print(f"    - 题目 {idx}: 随机采样得到 UT Input", flush=True)

            for idx in range(num_problems):  # 清理旧的 random_case_input
                data[idx]["random_case_input"] = []  # 重置为仅用于 BoN 聚类的输入池
            total_k_case = int(getattr(args, "k_case", 0))  # 目标 UT 数
            target_count = total_k_case if total_k_case > 0 else int(getattr(args, "random_ut_batch", 16))
            bon_candidates = (total_k_case * 2) if total_k_case > 0 else (target_count * 2)
            if bon_candidates > 0:  # 只有正数才生成
                bon_prompts = []  # BoN 随机输入 prompt 列表
                bon_prompt_idx_to_case_idx = []  # BoN prompt 对应题目索引
                for idx in range(num_problems):  # 每题生成 bon_candidates 条
                    for _ in range(bon_candidates):  # 生成固定数量
                        prompt_list = get_ut_input_random_generation_prompt(  # 构造随机输入 prompt
                            problem=data[idx]["question"],  # 题目文本
                            num_cases=1,  # 一次 1 条
                        )  # prompt 列表结束
                        prompt_str = get_full_prompt(prompt_list)  # 拼成完整 prompt
                        bon_prompts.append(prompt_str)  # 收集 prompt
                        bon_prompt_idx_to_case_idx.append(idx)  # 记录对应题目
                bon_outputs = runner.generate(bon_prompts)  # 按原顺序批量生成 BoN 随机输入
                record_direct_usage(
                    args,
                    bon_prompt_idx_to_case_idx,
                    bon_prompts,
                    bon_outputs,
                    "random_bon_input_generation",
                    usage_round_key,
                )
                random_inputs_by_problem = [[] for _ in range(num_problems)]  # 按题目收集候选输入
                for output, idx in zip(bon_outputs, bon_prompt_idx_to_case_idx):  # 回填 BoN 输入
                    candidates = parse_random_case_inputs(output)  # 解析候选输入
                    ut_input = candidates[0] if candidates else extract_ut_input(output)  # 解析输入
                    ut_input = _strip_case_prefix(ut_input)  # 去 CASE| 前缀
                    if not ut_input:  # 空输入跳过
                        continue  # 直接跳过
                    key = str(ut_input).strip()  # 去重 key
                    if not key or key == placeholder:  # 空或占位符跳过
                        continue  # 直接跳过
                    random_inputs_by_problem[idx].append(ut_input)  # 暂存候选输入
                # 去重后选 16（不足补占位符），与 self_play 保持一致
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
            print("    - [Idea Attack] Idea-based UT 数量为 0，跳过", flush=True)

        if random_target > 0:
            case_generation_result.extend(generate_random_cases())
        else:
            print("    - [Idea Attack] Random UT 数量为 0，跳过", flush=True)

        print("✓ 测试生成完成 (Idea Attack Mode)", flush=True)
        case_response_length = get_token_lengths(case_generation_result, tokenizer)
        mean_case = (
            sum(case_response_length) / len(case_response_length)
            if case_response_length
            else 0
        )

    return data, case_generation_result, mean_case  # 返回更新后的数据、生成结果和平均长度


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
    """
    统一入口，只做分发：
      - use_multi_stage_generation=True: 调用 run_generation_plansearch
      - use_multi_stage_generation=False: 调用 run_generation_original
    """
    if args.use_multi_stage_generation:
        # PlanSearch 不需要 code_generation_prompts / code_index，这里保持签名兼容
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
