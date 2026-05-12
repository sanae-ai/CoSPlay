# metrics.py
# 计算 one-shot / pass@k-only / BoN 三种评测模式的指标

import os  # 文件路径与目录
import math  # 组合数与数学计算
import json  # JSON 序列化

import numpy as np  # 数组计算
from termcolor import cprint  # 终端彩色输出

try:  # 优先走包内导入
    from evaluation import execution  # 执行脚本与沙箱
except Exception:  # 兼容脚本直跑
    import execution  # 执行脚本与沙箱


def compute_pass_at_k_for_task(all_test_table_i, pass_at_k_values):
    """
    计算单个任务的 pass@k 指标。
    
    Args:
        all_test_table_i: shape = (num_codes, num_tests)，布尔矩阵，表示每个代码是否通过了每个测试用例
        pass_at_k_values: 需要计算的 k 值列表
        
    Returns:
        dict: {k: pass@k_i}，该任务在不同 k 值下的 pass@k
    """
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


def _is_error_output(text):  # 判断输出是否报错
    if text is None:  # None 直接非 error
        return False  # 返回非 error
    raw = str(text).lower()  # 统一小写
    return "traceback" in raw or "error" in raw or "exception" in raw  # 关键字匹配


def _normalize_output(text):  # 归一化输出文本
    if text is None:  # 空值处理
        return ""  # 归一为空串
    return " ".join(str(text).split())  # 折叠多余空白


def _build_error_masks(outputs):  # 构造 error 位置掩码
    return [[_is_error_output(item) for item in row] for row in outputs]  # 行级掩码


def _hamming_distance(a, b, mask_a=None, mask_b=None):  # 计算忽略 error 的距离
    count = 0  # 距离计数
    for i, (x, y) in enumerate(zip(a, b)):  # 遍历位置
        if mask_a is not None and mask_b is not None:  # 有掩码才判断
            if (i < len(mask_a) and mask_a[i]) or (i < len(mask_b) and mask_b[i]):  # error 位置
                continue  # 跳过 error
        if x != y:  # 输出不一致
            count += 1  # 增加距离
    return count  # 返回距离


def _non_error_match_count(a, b, mask_a=None, mask_b=None):  # 统计非 error 的一致数
    count = 0  # 匹配计数
    for i, (x, y) in enumerate(zip(a, b)):  # 遍历位置
        if mask_a is not None and mask_b is not None:  # 有掩码才判断
            if (i < len(mask_a) and mask_a[i]) or (i < len(mask_b) and mask_b[i]):  # error 位置
                continue  # 跳过 error
        if x == y:  # 一致输出
            count += 1  # 增加匹配数
    return count  # 返回匹配数


def _cluster_outputs(outputs, max_diff, error_masks=None):  # 输出聚类
    if not outputs:  # 无输出直接返回
        return []  # 空簇
    # strict 模式（max_diff<=0）:
    # 仅比较双方都非 error 的位置；任一方为 error 则该位置跳过。
    # 为避免 AB、BC 连通后把 A、C 传递并簇，采用“簇内全成员兼容”分组，
    # 而不是并查集连通分量。
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
    parent = list(range(len(outputs)))  # 并查集父节点
    def find(x):  # 查找根
        while parent[x] != x:  # 路径压缩
            parent[x] = parent[parent[x]]  # 压缩路径
            x = parent[x]  # 迭代到根
        return x  # 返回根
    def union(a, b):  # 合并集合
        ra, rb = find(a), find(b)  # 取根
        if ra != rb:  # 根不同才合并
            parent[rb] = ra  # 合并
    for i in range(len(outputs)):  # 枚举对
        for j in range(i + 1, len(outputs)):  # 只看上三角
            if (  # 距离满足阈值
                _hamming_distance(  # 计算距离
                    outputs[i],  # 输出 i
                    outputs[j],  # 输出 j
                    None if error_masks is None else error_masks[i],  # error 掩码 i
                    None if error_masks is None else error_masks[j],  # error 掩码 j
                )  # 距离计算结束
                <= max_diff  # 阈值判断
            ):  # 满足条件
                union(i, j)  # 连边合并
    clusters = {}  # 根到簇
    for idx in range(len(outputs)):  # 汇总簇
        root = find(idx)  # 查根
        clusters.setdefault(root, []).append(idx)  # 归类
    return list(clusters.values())  # 返回簇列表


def _pick_cluster_and_center(clusters, outputs, candidate_indices, pass_counts, error_masks=None):  # 选簇与中心
    if not clusters:  # 无簇直接返回
        return [], -1, 0  # 空结果
    def cluster_match_score(cluster):  # 计算簇一致性得分
        score = 0  # 初始化得分
        for i in range(len(cluster)):  # 遍历成员
            for j in range(i + 1, len(cluster)):  # 遍历成员对
                ii = cluster[i]  # 成员 i
                jj = cluster[j]  # 成员 j
                score += _non_error_match_count(  # 累计非 error 一致数
                    outputs[ii],  # 输出 ii
                    outputs[jj],  # 输出 jj
                    None if error_masks is None else error_masks[ii],  # 掩码 ii
                    None if error_masks is None else error_masks[jj],  # 掩码 jj
                )  # 统计结束
        return score  # 返回得分
    def cluster_key(cluster):  # 簇排序关键字
        match_score = cluster_match_score(cluster)  # 一致性分
        pass_sum = sum(pass_counts[candidate_indices[i]] for i in cluster)  # case 通过数总和
        min_idx = min(candidate_indices[i] for i in cluster)  # 最小索引
        return (match_score, pass_sum, -min_idx)  # 越大越好
    best_cluster = max(clusters, key=cluster_key)  # 选最佳簇
    if len(best_cluster) == 1:  # 单点簇直接返回
        return best_cluster, best_cluster[0], 0  # 返回单点中心
    best_center = best_cluster[0]  # 默认中心
    best_key = None  # 最佳 key
    best_dist = 0  # 中心距离
    for i in best_cluster:  # 遍历候选中心
        dist_sum = sum(  # 距离求和
            _hamming_distance(  # 计算距离
                outputs[i],  # 输出 i
                outputs[j],  # 输出 j
                None if error_masks is None else error_masks[i],  # 掩码 i
                None if error_masks is None else error_masks[j],  # 掩码 j
            )  # 距离计算结束
            for j in best_cluster  # 遍历簇成员
        )  # 距离总和结束
        match_sum = sum(  # 非 error 匹配总和
            _non_error_match_count(  # 统计一致数
                outputs[i],  # 输出 i
                outputs[j],  # 输出 j
                None if error_masks is None else error_masks[i],  # 掩码 i
                None if error_masks is None else error_masks[j],  # 掩码 j
            )  # 统计结束
            for j in best_cluster  # 遍历簇成员
            if j != i  # 排除自身
        )  # 匹配总和结束
        err_count = sum(error_masks[i]) if error_masks is not None else 0  # error 数量
        key = (-match_sum, err_count, -pass_counts[candidate_indices[i]], candidate_indices[i])  # 排序 key
        if best_key is None or key < best_key:  # 更新最佳
            best_key = key  # 保存 key
            best_center = i  # 保存中心
            best_dist = dist_sum  # 保存距离
    return best_cluster, best_center, best_dist  # 返回簇与中心


def _format_bool_matrix_rows(mat):  # 将 bool 矩阵转成字符串行
    if mat is None:  # 空矩阵处理
        return []  # 返回空
    rows = []  # 行列表
    for row in mat:  # 遍历每行
        rows.append("".join("1" if bool(x) else "0" for x in row))  # 转为 01 字符串
    return rows  # 返回行列表


def _run_random_ut_batch(code_list, ut_inputs, time_limit, args):  # 执行 random UT 批次
    if not code_list or not ut_inputs:  # 无数据直接返回
        return [], []  # 空结果
    code_list_expanded = []  # 展开的 code 列表
    input_list = []  # 展开的输入列表
    time_limit_list = []  # 展开的时限列表
    for code in code_list:  # 遍历代码
        for ut_input in ut_inputs:  # 遍历输入
            code_list_expanded.append(code)  # 追加代码
            input_list.append(ut_input)  # 追加输入
            time_limit_list.append(time_limit)  # 追加时限
    outputs = execution.run_scripts_with_chunk(  # 调执行器
        code_list_expanded,  # 代码列表
        input_list,  # 输入列表
        time_limit_list,  # 时限列表
        args.num_chunks,  # 分块数
        args.exe_verbose,  # 是否打印进度
    )  # 执行结束
    n_col = len(ut_inputs)  # 每行 UT 数
    exe_matrix = []  # 输出矩阵
    bool_matrix = []  # 非 error 矩阵
    for i in range(len(code_list)):  # 组装矩阵
        row = outputs[i * n_col : (i + 1) * n_col]  # 取行输出
        exe_matrix.append(row)  # 保存输出行
        bool_matrix.append([not _is_error_output(x) for x in row])  # 标记非 error
    return exe_matrix, bool_matrix  # 返回矩阵


def _select_random_inputs(inputs, target_count, placeholder):  # 选取固定数量 random 输入
    if target_count <= 0:  # 无需选择
        return []  # 返回空
    if len(inputs) >= target_count:  # 已足够
        return list(inputs[:target_count])  # 直接截断
    if not inputs:  # 完全没有有效输入
        return [placeholder] * target_count  # 用占位符填满
    padded = list(inputs)  # 复制已有输入
    while len(padded) < target_count:  # 不足则补齐
        padded.append(placeholder)  # 用占位符补齐
    return padded  # 返回补齐结果


def _execute_random_ut_batches(  # 批量执行 random UT（全局合并后再分块）
    data,  # 数据列表
    args,  # 执行参数
    random_ut_max_attempts,  # 最大轮数
    random_ut_batch,  # 每轮数量
    random_ut_total,  # 总量上限
    max_scale_code,  # 最大代码数
    random_inputs_key="random_case_input",  # random 输入字段名
):  # 函数结束标记
    random_exec_cache = {}  # 缓存执行结果
    if random_ut_max_attempts <= 0 or random_ut_batch <= 0:  # 参数非法直接返回
        return random_exec_cache  # 返回空缓存
    for attempt in range(random_ut_max_attempts):  # 逐轮执行
        code_list = []  # 展开的代码列表
        input_list = []  # 展开的输入列表
        time_limit_list = []  # 展开的超时列表
        index_list = []  # 题目索引列表
        position_list = []  # (code_idx, ut_idx) 列表
        batch_len_by_task = {}  # 每题本轮 UT 数
        for i in range(len(data)):  # 遍历题目
            if data[i].get("case_exe_results") is None or data[i].get("test_exe_results") is None:  # 缺结果跳过
                continue  # 跳过该题
            random_inputs = data[i].get(random_inputs_key, [])  # 读取 random 输入
            if random_ut_total > 0:  # 有总量限制
                random_inputs = random_inputs[:random_ut_total]  # 截断到总量
            start = attempt * random_ut_batch  # 本轮起始
            end = start + random_ut_batch  # 本轮结束
            batch_inputs = random_inputs[start:end]  # 本轮输入
            if not batch_inputs:  # 本轮没有输入
                continue  # 跳过该题
            max_code = min(max_scale_code, len(data[i].get("generated_code", [])))  # 代码上限
            if max_code <= 0:  # 没有代码
                continue  # 跳过该题
            data[i][f"random_case_exe_results_{attempt + 1}"] = [  # 初始化输出矩阵
                ["" for _ in range(len(batch_inputs))]  # 每行填空
                for _ in range(max_code)  # 行数为代码数
            ]  # 初始化结束
            batch_len_by_task[i] = len(batch_inputs)  # 记录本轮 UT 数
            for code_idx in range(max_code):  # 遍历代码
                code = data[i]["generated_code"][code_idx]  # 取代码
                for ut_idx, ut_input in enumerate(batch_inputs):  # 遍历 UT
                    code_list.append(code)  # 追加代码
                    input_list.append(ut_input)  # 追加输入
                    time_limit_list.append(data[i].get("test_time_limit", 1))  # 追加时限
                    index_list.append(i)  # 记录题目索引
                    position_list.append((code_idx, ut_idx))  # 记录位置
        if code_list:  # 有任务才执行
            outputs = execution.run_scripts_with_chunk(  # 调执行器
                code_list,  # 代码列表
                input_list,  # 输入列表
                time_limit_list,  # 时限列表
                args.num_chunks,  # 分块数
                args.exe_verbose,  # 打印进度
            )  # 执行结束
            for idx, output in enumerate(outputs):  # 回填输出
                task_i = index_list[idx]  # 题目索引
                code_idx, ut_idx = position_list[idx]  # 位置索引
                data[task_i][f"random_case_exe_results_{attempt + 1}"][code_idx][ut_idx] = output  # 写入输出
        for task_i, _ in batch_len_by_task.items():  # 组装 bool 行
            exe_matrix = data[task_i].get(f"random_case_exe_results_{attempt + 1}", [])  # 输出矩阵
            if not exe_matrix:  # 空矩阵跳过
                continue  # 跳过该题
            bool_matrix = [  # 构造非 error 矩阵
                [not _is_error_output(x) for x in row]  # 行内转换
                for row in exe_matrix  # 遍历行
            ]  # 构造结束
            bool_rows = _format_bool_matrix_rows(bool_matrix)  # 转为 01 行
            data[task_i][f"random_case_bool_table_rows_{attempt + 1}"] = bool_rows  # 保存矩阵
            random_exec_cache[(task_i, attempt)] = (exe_matrix, bool_rows)  # 写入缓存
    return random_exec_cache  # 返回缓存


def _get_outputs_result_name(args, outputs_name):
    """
    统一生成结果文件路径，和你原来代码保持一致。
    """
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
    """
    打开结果文件并返回 (file_object, path)
    """
    outputs_result_name = _get_outputs_result_name(args, outputs_name)
    os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
    f = open(outputs_result_name, "a")
    return f, outputs_result_name


def _safe_divide(d1, d2):
    return d1 / d2 if d2 != 0 else 0


def _ut_input_rank(case_inputs):
    """
    构造相等矩阵（相同=1，不同=0），返回其秩与样本数。
    """
    if not case_inputs:
        return 0, 0
    n = len(case_inputs)
    eq = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            eq[i, j] = 1.0 if case_inputs[i] == case_inputs[j] else 0.0
    rank = int(np.linalg.matrix_rank(eq))
    return rank, n


# ======================== 模式 1：single_eval=True → one-shot 代码生成能力 ========================


def _evaluate_single_eval_mode(data, outputs_name, mean_code, mean_case, args):
    """
    single_eval = True 模式下的评估逻辑：
      - 仅评估 one-shot 代码生成能力
      - 不计算 pass@k
      - 不评估 LLM 生成的测试用例或 BoN 性能
      - 只关注在真实单元测试下的 per-code 成功率
    """
    code_score = 0
    code_num = 0
    code_acc_score = 0
    code_acc_num = 0

    for i in range(len(data)):
        if data[i]["test_exe_results"] is None or data[i]["test_bool_table"] is None:
            continue
        all_test_table_i = np.array(data[i]["test_bool_table"]).copy()

        # 完全通过所有真实 UT 的代码
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
        # one-shot 模式只关心代码，不关心 UT 相关指标
        save_and_print(f"code average response length: {mean_code}")

    return data


# ======================== 模式 2：pass@k-only（single_eval=False & eval_pass_at_k_only=True） ========================


def _evaluate_passatk_mode(data, outputs_name, mean_code, mean_case, args):
    """
    single_eval = False, eval_pass_at_k_only = True 模式下的评估逻辑：
      - 进行多次代码采样
      - 仅使用真实单元测试计算 pass@k 和 per-code 成功率
      - 不生成或评估 LLM 生成的测试用例，不进行 BoN
    """
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

        # 完全通过所有真实 UT 的代码
        correct_code_list = np.where(all_test_table_i.all(axis=1))[0].tolist()
        code_score += len(correct_code_list)
        code_num += all_test_table_i.shape[0]

        code_acc_score += np.sum(all_test_table_i).item()
        code_acc_num += all_test_table_i.shape[0] * all_test_table_i.shape[1]

        # 单题 pass@k
        if pass_at_k_values:
            task_passk = compute_pass_at_k_for_task(all_test_table_i, pass_at_k_values)
            for k, v in task_passk.items():
                pass_at_k_scores[k].append(v)

    code_acc = _safe_divide(code_score, code_num)
    code_acc_acc = _safe_divide(code_acc_score, code_acc_num)

    # 聚合多题的 pass@k
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

        # pass@k-only 模式一般没有 LLM 生成的 UT，因此不打印 UT 相关指标
        save_and_print(f"code average response length: {mean_code}")

    return data


# ======================== 模式 3：BoN (混合增强版) ========================

def _evaluate_bon_mode(data, outputs_name, mean_code, mean_case, args):
    """
    BoN 混合模式：
    1. 计算生成的测试用例质量 (Case Acc, p_00, p_01)。
    2. 计算 BoN 选优后的准确率。
    3. (新增) 如果提供了 pass_at_k_list，同时利用全量代码计算 Pass@k。
    """
    
    # --- 1. 准备 Pass@k 统计 ---
    pass_at_k_values = sorted(set(args.pass_at_k_list)) if args.pass_at_k_list else []
    pass_at_k_scores = {k: [] for k in pass_at_k_values}

    # --- 2. 准备 BoN 统计 ---
    stats_single = {
        "BoN_score": 0,
        "BoN_num": 0,
        "BoN_acc_score": 0,
        "BoN_acc_num": 0,
    }
    # BoN 统计列表（按 scale_tuple_list 初始化，避免 index 越界）
    stats = []  # BoN 统计列表
    stats_all = []  # 记录 BoN_all（并列最佳全测）的统计
    for tpl in (args.scale_tuple_list or []):  # 遍历每个尺度配置
        stats_i = {  # BoN 结构
            "tuple": tpl,  # 当前 (k_code, k_case)
            "BoN_score": 0,  # BoN 通过题数
            "BoN_num": 0,  # BoN 题目总数
            "BoN_acc_score": 0,  # BoN 累计通过数
            "BoN_acc_num": 0,  # BoN 累计总数
            "passed_tasks": [],  # 记录通过题目索引
        }  # BoN 结构结束
        stats.append(stats_i)  # 写入 BoN 列表
        stats_all_i = {  # BoN_all 结构
            "tuple": tpl,  # 当前 (k_code, k_case)
            "BoN_all_score": 0,  # BoN_all 通过题数
            "BoN_all_num": 0,  # BoN_all 题目总数
            "BoN_all_acc_score": 0,  # BoN_all 累计通过数
            "BoN_all_acc_num": 0,  # BoN_all 累计总数
            "tie_counts": [],  # 并列候选数量记录
        }  # BoN_all 结构结束
        stats_all.append(stats_all_i)  # 写入 BoN_all 列表

    skip_bon_log = bool(getattr(args, "skip_bon_log", False))
    # --- 3. 基础指标 accumulators ---
    code_score = 0; code_num = 0
    code_acc_score = 0; code_acc_num = 0
    case_score = 0; case_num = 0
    case_acc_score = 0; case_acc_num = 0
    p_01_score = 0; p_01_num = 0
    p_00_score = 0; p_00_num = 0
    ut_rank_sum = 0; ut_rank_norm_sum = 0; ut_rank_num = 0
    bon_debug_records = []  # BoN debug 记录
    random_exec_cache = {}  # random UT 执行缓存
    use_random_ut_cluster = bool(getattr(args, "use_random_ut_cluster", True))  # 开关
    random_ut_total = int(getattr(args, "random_ut_total", 80))  # random UT 总数
    random_ut_batch = int(getattr(args, "random_ut_batch", 16))  # 每批数量
    random_ut_max_attempts = int(getattr(args, "random_ut_max_attempts", 5))  # 最大轮数
    random_ut_min_top_count = int(getattr(args, "random_ut_min_top_count", 2))  # 触发阈值
    random_ut_cluster_max_diff = int(getattr(args, "random_ut_cluster_max_diff", 0))  # 距离阈值
    random_ut_placeholder = getattr(  # random UT 占位符
        args, "random_ut_placeholder", "We can not extract the input in the output. "  # 默认占位符
    )  # 占位符获取结束
    random_ut_select_count = random_ut_batch  # 去重后选取数量
    random_ut_exec_rounds = 1  # 只执行 1 轮聚类
    max_scale_code = max([tpl[0] for tpl in args.scale_tuple_list], default=0)  # 最大 code 数
    if use_random_ut_cluster:  # 启用 random UT 聚类时准备字段
        for i in range(len(data)):  # 遍历题目
            for attempt in range(random_ut_max_attempts):  # 最多 5 轮
                data[i].setdefault(f"random_case_bool_table_rows_{attempt + 1}", [])  # 执行矩阵行
                data[i].setdefault(f"random_case_exe_results_{attempt + 1}", [])  # 执行输出矩阵
            raw_inputs = data[i].get("random_case_input", [])  # 原始 random 输入
            if random_ut_total > 0:  # 有总量上限
                raw_inputs = raw_inputs[:random_ut_total]  # 截断到总量
            data[i]["random_case_input_selected"] = _select_random_inputs(  # 直接选 16 个
                raw_inputs,  # 原始输入
                random_ut_select_count,  # 选取数量
                random_ut_placeholder,  # 占位符
            )  # 选取结束
        if random_ut_select_count > 0 and max_scale_code > 0:  # 有效配置才执行
            random_exec_cache = _execute_random_ut_batches(  # 批量执行 random UT
                data,  # 数据列表
                args,  # 执行参数
                random_ut_exec_rounds,  # 只执行 1 轮
                random_ut_select_count,  # 每轮数量
                random_ut_select_count,  # 总量上限
                max_scale_code,  # 代码上限
                random_inputs_key="random_case_input_selected",  # 使用去重后的输入
            )  # 执行结束

    for i in range(len(data)):
        # 必须同时存在生成的 UT 和真实的 UT 结果
        if data[i]["case_exe_results"] is None or data[i]["test_exe_results"] is None:
            continue

        all_case_table_i = np.array(data[i]["case_bool_table"]).copy()
        all_test_table_i = np.array(data[i]["test_bool_table"]).copy()
        case_is_valid = None
        valid_indices = None

        # [Feature 0] 应用 UT 掩码 (只保留通过一致性筛选的 UT)
        # 只有在 plansearch 模式下，且使用了投票，且设置了自洽性数量 > 1 才应用 UT 掩码
        if args.generation_mode == "plansearch":
            case_is_valid = data[i].get("case_is_valid", [True] * all_case_table_i.shape[1])
            valid_indices = [j for j, valid in enumerate(case_is_valid) if valid]
            # 过滤掉无效的列
            all_case_table_i = all_case_table_i[:, valid_indices].copy()

        # [Feature 0.5] 生成 UT 输入重复度（用相等矩阵秩做归一化）
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

        # [Feature 1] 计算 Pass@k (全量代码)
        # 只要配置了参数，就对当前所有 k_code 个代码算 Pass@k
        if pass_at_k_values:
            task_passk = compute_pass_at_k_for_task(all_test_table_i, pass_at_k_values)
            for k, v in task_passk.items():
                pass_at_k_scores[k].append(v)

        # [Feature 2] 基础 Code Acc 统计
        correct_code_list = np.where(all_test_table_i.all(axis=1))[0].tolist()
        code_score += len(correct_code_list)
        code_num += all_test_table_i.shape[0]
        code_acc_score += np.sum(all_test_table_i).item()
        code_acc_num += all_test_table_i.shape[0] * all_test_table_i.shape[1]

        # [Feature 3] 计算生成 UT 的质量 (UT Accuracy / p_00 / p_01)
        # 只看 AC 的代码：UT 是否也全对？
        sub_case_table_i = all_case_table_i[correct_code_list, :].copy()
        
        if len(correct_code_list) > 0 and all_case_table_i.shape[1] > 0:
            correct_case_list = np.where(sub_case_table_i.all(axis=0))[0].tolist()
            case_score += len(correct_case_list)
            case_num += sub_case_table_i.shape[1]
            case_acc_score += np.sum(sub_case_table_i).item()
            case_acc_num += sub_case_table_i.shape[0] * sub_case_table_i.shape[1]

            # 计算 p_01 (False Rejection): 正确代码被 UT 判错
            # 计算 p_00 (False Acceptance): 错误代码被 UT 判对
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

        # [Feature 4] BoN 计算 (取前 N 个切片)
        if len(args.scale_tuple_list) > 0:
            index_id = 0
            for scale_num_code, scale_num_case in args.scale_tuple_list:
                # 截取前 N 个代码和前 M 个测试用例
                # 注意：这里的 scale_num_case 是原始请求的数量，但我们现在只有过滤后的列
                actual_num_case = min(scale_num_case, all_case_table_i.shape[1])
                case_table_i = all_case_table_i[:scale_num_code, :actual_num_case].copy()
                test_table_i = all_test_table_i[:scale_num_code, :].copy()
                
                pass_counts = np.sum(case_table_i, 1) if case_table_i.shape[0] > 0 else np.array([])  # case 通过数
                # 如果没有有效的 UT，则无法选优，默认选第一个代码  # 规则说明
                if case_table_i.shape[1] > 0 and pass_counts.size > 0:  # 有 UT 才能比较
                    best_code_index = int(pass_counts.argmax())  # 选通过数最多的代码
                else:  # 没有有效 UT
                    best_code_index = 0  # 默认选第一个
                # 验证：看该代码在真实 UT 上的表现  # 验证说明
                sub_test_table_i = test_table_i[best_code_index, :].copy()  # 取该代码 test 行
                stats[index_id]["BoN_score"] += int(all(sub_test_table_i))  # BoN 通过数
                stats[index_id]["BoN_num"] += 1  # BoN 总数
                stats[index_id]["BoN_acc_score"] += np.sum(sub_test_table_i).item()  # BoN 累计通过
                stats[index_id]["BoN_acc_num"] += len(sub_test_table_i)  # BoN 累计总数
                if all(sub_test_table_i):  # 若完全通过
                    stats[index_id]["passed_tasks"].append(i)  # 记录通过题目
                # BoN_all: 对所有并列“通过 UT 数量最多”的代码都算一次，取表现最好的  # BoN_all 说明
                if pass_counts.size > 0:  # 有通过数才计算
                    max_pass = pass_counts.max()  # 最大通过数
                    top_candidates = np.where(pass_counts == max_pass)[0].tolist()  # 并列候选
                    tie_count = len(top_candidates)  # 并列数量
                    if tie_count == 0:  # 没候选兜底
                        tie_count = 1  # 兜底为 1
                        top_candidates = [best_code_index]  # 兜底候选
                    best_top_idx = top_candidates[0]  # 默认最优
                    best_top_score = -1  # 最优分数
                    for cand in top_candidates:  # 遍历候选
                        cand_row = test_table_i[cand, :].copy()  # 取 test 行
                        cand_score = cand_row.mean() if len(cand_row) > 0 else 0  # 平均通过率
                        if cand_score > best_top_score:  # 更新最优
                            best_top_score = cand_score  # 更新分数
                            best_top_idx = cand  # 更新索引
                    best_row = test_table_i[best_top_idx, :].copy()  # 最优行
                    stats_all[index_id]["BoN_all_score"] += int(all(best_row))  # BoN_all 通过数
                    stats_all[index_id]["BoN_all_num"] += 1  # BoN_all 总数
                    stats_all[index_id]["BoN_all_acc_score"] += np.sum(best_row).item()  # BoN_all 累计通过
                    stats_all[index_id]["BoN_all_acc_num"] += len(best_row)  # BoN_all 累计总数
                    stats_all[index_id]["tie_counts"].append(tie_count)  # 记录并列数
                debug_entry["bon_slices"].append(  # 写入 debug
                    {  # debug 内容
                        "tuple": [scale_num_code, scale_num_case],  # 切片配置
                        "case_slice_shape": list(case_table_i.shape),  # case 形状
                        "test_slice_shape": list(test_table_i[:scale_num_code, :].shape),  # test 形状
                        "case_slice": case_table_i.tolist(),  # case 切片
                        "test_slice": test_table_i[:scale_num_code, :].tolist(),  # test 切片
                        "best_code_index": int(best_code_index),  # BoN 选码索引
                        "best_code_test_row": sub_test_table_i.tolist(),  # BoN test 行
                    }  # debug 结构结束
                )  # debug 追加结束
                index_id += 1  # 下一个尺度索引
        bon_debug_records.append(debug_entry)

    # --- 4. 汇总结果 ---
    code_acc = _safe_divide(code_score, code_num)
    code_acc_acc = _safe_divide(code_acc_score, code_acc_num)
    
    # UT 质量指标
    case_acc = _safe_divide(case_score, case_num)
    case_acc_acc = _safe_divide(case_acc_score, case_acc_num)
    p_01 = _safe_divide(p_01_score, p_01_num)
    p_00 = _safe_divide(p_00_score, p_00_num)

    # Pass@k 汇总
    overall_pass_at_k = {}
    for k in pass_at_k_values:
        vals = pass_at_k_scores.get(k, [])
        overall_pass_at_k[k] = sum(vals) / len(vals) if vals else 0.0

    # --- 5. 打印并写入文件 ---
    f, outputs_result_name = _open_result_file(args, outputs_name)
    with f:
        def save_and_print(text):
            cprint(text, color="green")
            f.write(text + "\n")

        save_and_print("=== Evaluation Report ===")
        
        # 1. 打印基础代码准确率
        save_and_print(
            f"code acc (average proportion of tasks the generated code can pass): {code_acc}\n"
            f"code accumulate acc (average proportion of unit tests the generated code can pass): {code_acc_acc}"
        )
        
        # 3. 打印正确的题目序号
        correct_task_indices = [i for i, d in enumerate(data) if d.get("is_correct", False)]
        save_and_print(f"Correct task indices: {correct_task_indices}")
        
        # 2. 打印 Pass@k (如果计算了)
        if overall_pass_at_k:
            save_and_print("--- Pass@k Metrics ---")
            for k in sorted(overall_pass_at_k.keys()):
                save_and_print(f"pass@{k}: {overall_pass_at_k[k]}")

        # 3. 打印生成的 UT 质量 (这就是你问的 UT 准确率)
        save_and_print("--- Generated Unit Test Quality ---")
        save_and_print(
            f"estimated unit test acc (average proportion of tasks that the generated unit test can pass all correct code): {case_acc}\n"
            f"estimated unit test accumulate acc (average proportion of correct code that the generated unit test can pass): {case_acc_acc}"
        )
        save_and_print(f"estimated p_01 (False Rejection Rate): {1 - p_01}")
        save_and_print(f"estimated p_00 (False Acceptance Rate): {p_00}")

        # 生成 UT 输入重复度（秩归一化）
        if ut_rank_num > 0:
            ut_rank_avg = ut_rank_sum / ut_rank_num
            ut_rank_norm_avg = ut_rank_norm_sum / ut_rank_num
            save_and_print("--- UT Input Redundancy ---")
            save_and_print(f"ut input rank avg: {ut_rank_avg}")
            save_and_print(f"ut input rank norm avg: {ut_rank_norm_avg}")

        # 4. 打印 BoN 结果（允许通过 skip_bon_log 跳过）
        if (not skip_bon_log) and (args.scale_tuple_list or []):
            save_and_print("--- Best-of-N Metrics ---")  # 分组标题
            # 只读取 execution 中缓存的 New_BoN，避免重复执行 random UT
            hist = getattr(args, "_new_bon_cached", None)
            for st, st_all in zip(stats, stats_all):  # 遍历三种统计
                tuple_name = st["tuple"]  # 当前配置
                if st["BoN_num"] == 0 or st["BoN_acc_num"] == 0:  # 无数据跳过
                    continue  # 直接跳过
                acc = st["BoN_score"] / st["BoN_num"]  # BoN acc
                acc_acc = st["BoN_acc_score"] / st["BoN_acc_num"]  # BoN 累计 acc
                save_and_print(f"BoN setting {tuple_name}:")  # 打印 BoN 配置
                save_and_print(f"acc: {acc}, accumulate acc: {acc_acc}")  # 打印 BoN 结果
                passed = st.get("passed_tasks", [])  # 通过题列表
                passed_str = ", ".join(str(x) for x in passed) if passed else "None"  # 格式化列表
                save_and_print(f"passed task indices: {passed_str}")  # 打印通过题
                # BoN_all 汇总（并列最佳全测）
                tuple_name = st_all.get("tuple", st.get("tuple"))  # BoN_all 配置（兜底避免 KeyError）
                if st_all.get("BoN_all_num", 0) == 0 or st_all.get("BoN_all_acc_num", 0) == 0:
                    pass
                else:
                    acc_all = st_all["BoN_all_score"] / st_all["BoN_all_num"]  # BoN_all acc
                    acc_acc_all = st_all["BoN_all_acc_score"] / st_all["BoN_all_acc_num"]  # BoN_all 累计 acc
                    tie_counts = st_all.get("tie_counts", [])
                    tie_info = (
                        f"max_top_candidates={max(tie_counts)}, "
                        f"avg_top_candidates={sum(tie_counts)/len(tie_counts):.2f}, "
                        f"ties_gt1={sum(1 for t in tie_counts if t>1)}"
                    ) if tie_counts else "no data"
                    save_and_print(f"BoN_all setting {tuple_name}:")
                    save_and_print(f"acc: {acc_all}, accumulate acc: {acc_acc_all}")
                    save_and_print(f"top-candidate tie summary: {tie_info}")

                # New_BoN（增量式历史池）：使用 execution 内部逻辑输出
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

    # --- 6. 将真正参与 BoN 的布尔矩阵和截取信息写入 JSON，便于复核 ---
    if not skip_bon_log:
        debug_outputs_result_name = _get_outputs_result_name(args, outputs_name)
        debug_json_path = os.path.splitext(debug_outputs_result_name)[0] + "_bon_tables.json"
        os.makedirs(os.path.dirname(debug_json_path), exist_ok=True)
        with open(debug_json_path, "w") as df:
            json.dump(bon_debug_records, df, indent=2)
        cprint(f"[BoN Debug] Saved BoN boolean tables to {debug_json_path}", color="yellow")

    return data



# ======================== 统一入口：根据 flag 选择模式 ========================


def compute_and_log_metrics(data, outputs_name, mean_code, mean_case, args):
    """
    统一的指标计算和日志记录入口。
    根据 args 中的配置选择三种互斥模式之一进行评估：

    1. if args.single_eval:
        → One-shot 模式：只评估 one-shot 代码生成能力
          （不计算 pass@k，不计算 BoN，不使用 LLM 生成的 UT）

    2. elif args.eval_pass_at_k_only:
        → Pass@k-only 模式
          （多次采样 + 真实 UT + 计算 pass@k，不计算 BoN，不使用 LLM 生成的 UT）

    3. elif args.eval_bon:
        → BoN 模式
          （多次采样 + 模型生成 UT + 计算 BoN + 评估 UT 质量指标）

    如果以上三个标志都为 False，则抛出异常，提示配置错误。
    """

    if args.single_eval:
        # one-shot 模式
        data = _evaluate_single_eval_mode(data, outputs_name, mean_code, mean_case, args)

    elif args.eval_pass_at_k_only:
        # 多采样 + pass@k-only
        data = _evaluate_passatk_mode(data, outputs_name, mean_code, mean_case, args)

    elif args.eval_bon:
        # BoN 模式（需要 LLM 生成的 UT）
        data = _evaluate_bon_mode(data, outputs_name, mean_code, mean_case, args)

    else:
        # 理论上不应该出现，如果出现说明 config / args 写错了
        raise ValueError(
            "No evaluation mode selected. "
            "Please set exactly ONE of {single_eval, eval_pass_at_k_only, eval_bon} to True."
        )

    # --------- 最后一步：把 numpy.bool_ 转成 list，方便写回 JSON ---------
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
