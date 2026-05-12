# prompts.py
# 只负责“生成 prompt”
from jinja2 import Template
from itertools import combinations
import re
# 生成每一段prompt
# 生成每一段prompt
def get_scaling_prompt(
    data_i,
    method,
    *,
    k_case,
    system_prompts,
    system_case_prompts,
    special_requirements,
    no_example
):
    """
    生成扩展（Scaling）相关的 Prompt。
    根据 method 参数选择生成代码生成 Prompt 还是测试用例生成 Prompt。

    Args:
        data_i (dict): 单个题目数据
        method (str): "sample" (代码生成) 或 "case" (用例生成)
        k_case (int): 生成用例的数量
        system_prompts (str): 代码生成的系统提示词模板
        system_case_prompts (str): 用例生成的系统提示词模板
        special_requirements (str): 特殊要求字符串
        no_example (bool): 是否不包含示例

    Returns:
        str: 渲染后的 Prompt 字符串
    """
    """
    复用你原来的 get_scaling_prompt，只是把依赖的 config 参数变成函数参数传进来。
    """
    problem = data_i["question"]

    if method == "sample":
        # 代码生成的整体 prompt
        return Template(system_prompts).render(
            language="python",
            special_requirements=special_requirements,
            problem=problem,
        )

    if method == "case":
        # UT 生成的 prompt
        n_example = min(k_case, len(data_i["example_input"]))
        example_input = ", ".join([repr(item) for item in data_i["example_input"]])
        example_output = ", ".join([repr(item) for item in data_i["example_output"]])

        if n_example == 0:
            example_intro = " "
        elif n_example == 1:
            example_intro = """We already have one test sample:
            Its input is {{example_input}}. Its output is {{example_output}}.
            """
            example_intro = Template(example_intro).render(
                example_input=example_input, example_output=example_output
            )
        else:
            example_intro = """We already have {{n_sample}} test samples:
            The inputs are, respectively, {{example_input}}.
            The corresponding outputs are {{example_output}}.
            """
            example_intro = Template(example_intro).render(
                n_sample=n_example,
                example_input=example_input,
                example_output=example_output,
            )

        return Template(system_case_prompts).render(
            problem=problem,
            example_intro=example_intro,
        )

    raise ValueError(f"Unknown method: {method}")


def get_stage1_prompt(problem, tmpl):
    """
    生成第一阶段 Prompt：一阶观察。
    """
    return Template(tmpl).render(problem=problem)


def get_stage2_prompt(problem, first_order_observations, tmpl):
    """
    生成第二阶段 Prompt：二阶观察。
    """
    return Template(tmpl).render(
        problem=problem,
        first_order_observations=first_order_observations,
    )


def init_eval_fields_for_data(data):
    """
    把每道题里用到的公共字段初始化一下，避免在多个函数里重复写。
    """
    for i in range(len(data)):
        data[i]["full_code_generation"] = []
        data[i]["full_case_generation"] = []
        data[i]["generated_code"] = []
        data[i]["case_input"] = []
        data[i]["case_output"] = []
        data[i]["case_text"] = []
        
        # Original outputs for debugging/analysis
        data[i]["idea_generation_outputs"] = []
        data[i]["attack_ideas"] = []
        data[i]["case_input_original"] = []
        data[i]["case_output_original"] = []
        data[i]["case_is_valid"] = []

        # PlanSearch / multi-stage intermediate observations.
        data[i]["stage1_observations"] = None
        data[i]["stage2_observations"] = None


def build_code_prompts_original_eval(data, args):
    """
    原始评测（非 PlanSearch）的代码生成 prompt 构建。
    返回:
      - code_generation_prompts: 按照 k_code 展开后的 prompt 列表
      - code_index: 和上面一一对应，每个 prompt 属于哪一道题
    """
    init_eval_fields_for_data(data)

    code_generation_prompts = []
    code_index = []

    num = len(data)
    print("使用一体化生成模式（original eval）", flush=True)

    for i in range(num):
        data_i = data[i].copy()
        problem = data_i["question"]

        # 代码生成 prompt（sample 模式）
        prompt_sample = get_scaling_prompt(
            data_i,
            "sample",
            k_case=args.k_case,  # 这个参数在 sample 里其实没用，但保持签名一致
            system_prompts=args.system_prompts,
            system_case_prompts=args.system_case_prompts,
            special_requirements=args.special_requirements,
            no_example=args.no_example,
        )
        data[i]["code_generation_prompt"] = prompt_sample

        # 和你之前一样：为每题重复 k_code 次
        code_generation_prompts += [prompt_sample] * args.k_code
        code_index += [i] * args.k_code

    print(
        f"✓ Prompt 生成完成: {len(code_generation_prompts)} 个代码生成 prompt",
        flush=True,
    )

    return code_generation_prompts, code_index


def build_code_prompts_plansearch(data, args):
    """
    PlanSearch 评测的代码部分：只负责为每道题构建 Stage1 prompt 和模板信息。
    具体的 Stage2/3/4 prompt 在 PlanSearch 流程里按观察子集动态拼。
    
    返回:
      - stage1_prompts: 长度 = 题目数，每题一个 Stage1 prompt（方便你一次性 batch 给 LLM）
    """
    init_eval_fields_for_data(data)

    num = len(data)
    stage1_prompts = []

    print("使用 PlanSearch 分阶段生成模式", flush=True)

    for i in range(num):
        problem = data[i]["question"]

        # Stage1 prompt：一阶观察
        prompt_stage1 = get_stage1_prompt(problem, args.system_prompts_stage1)

        # 保存模板信息，后面 PlanSearch 会用到 question 来拼 Stage2/3/4
        data[i]["code_generation_prompts"] = {
            "stage1": prompt_stage1,
            "stage2_template": problem,
        }

        stage1_prompts.append(prompt_stage1)

    print(
        f"✓ PlanSearch: 为 {num} 题构建 Stage1 prompt",
        flush=True,
    )

    return stage1_prompts

def build_case_prompts(data, args):
    """
    原始用例（unit test）生成 prompt 函数。

    返回:
      - case_generation_prompts: 展开后的 prompt 列表（每道题重复 k_case 次）
      - case_index: 和上面一一对应，每个 prompt 属于哪一道题
    """
    case_generation_prompts = []
    case_index = []

    num = len(data)

    for i in range(num):
        data_i = data[i].copy()

        # 是否在 prompt 里提供公开样例
        if args.no_example:
            data_i["example_input"] = []
            data_i["example_output"] = []

        prompt_case = get_scaling_prompt(
            data_i,
            "case",
            k_case=args.k_case,
            system_prompts=args.system_prompts,
            system_case_prompts=args.system_case_prompts,
            special_requirements=args.special_requirements,
            no_example=args.no_example,
        )
        data[i]["case_generation_prompt"] = prompt_case

        case_generation_prompts += [prompt_case] * args.k_case
        case_index += [i] * args.k_case

    print(
        f"✓ 用例 Prompt 生成完成: {len(case_generation_prompts)} 个测试生成 prompt",
        flush=True,
    )

    return case_generation_prompts, case_index


def build_prompts_for_dataset(data, args):
    """
    统一封装：
      - 根据 use_multi_stage_generation 选择：
          * 原始一体化模式：build_code_prompts_original_eval
          * PlanSearch 分阶段模式：build_code_prompts_plansearch
      - 总是构建 case_generation_prompts（当 k_case=0 时自然会得到空列表）

    返回:
      code_generation_prompts, code_index, case_generation_prompts, case_index

    注意：
      - PlanSearch 模式下，真正用来做 Stage1 的 prompt
        已经存进了 data[i]["code_generation_prompts"]["stage1"]，
        run_generation_plansearch() 只看 data 里的这些字段，
        不会用这里返回的 code_generation_prompts / code_index。
    """
    # 1) 代码生成 prompt
    if args.use_multi_stage_generation:
        # PlanSearch：只需要在 data 里写好 stage1/template 信息即可
        # build_code_prompts_plansearch 会负责:
        #   - init_eval_fields_for_data(data)
        #   - 为每题构建 stage1 prompt 并存到 data[i]["code_generation_prompts"]
        _ = build_code_prompts_plansearch(data, args)

        # 对于 PlanSearch，run_generation_pipeline 里不会用下面两个返回值，
        # 但为了 main.py 的统一接口，这里返回空列表占位即可。
        code_generation_prompts = []
        code_index = []
    else:
        # 原始一体化模式：需要真正返回展开后的 code_generation_prompts 和 code_index
        code_generation_prompts, code_index = build_code_prompts_original_eval(data, args)

    # 2) 用例（UT）生成 prompt
    # single_eval / pass@k-only 模式下，parse_args 已经把 k_case=0，
    # 这里依然可以统一调用 build_case_prompts：
    #   - 当 k_case=0 时，每题虽然会构造一个 prompt_case，但不会被重复，
    #     最终 case_generation_prompts 长度为 0，不会真的调模型生成 UT。
    case_generation_prompts, case_index = build_case_prompts(data, args)

    return code_generation_prompts, code_index, case_generation_prompts, case_index
