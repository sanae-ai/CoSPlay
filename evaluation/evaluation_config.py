# ========================  This is the config for evaluation  =========================
# ======================================================================================


# set True if you want to evaluate API model, set False for vllm inference model
use_api = True


# dataset you can directly use for evaluation: CodeContests_test, LiveBench, LiveCodeBench, Codeforces, MBPP
dataset = "CodeContests"

# number of codes (k_code) and unit tests (k_case) generated for each task; 
# 每个任务生成的代码数量 (k_code) 和单元测试数量 (k_case)
k_code = 8
k_case = 8

# the BoN setting (num_code, num_case) you want to test, num_code <= k_code and num_case <= k_case. The running time is totally decided by k_code and k_case, so you can add as many settings as you like
scale_tuple_list = [(4, 4), (8, 8)]

# how many parts the execution tasks are divided into (too small may get stuck), should be proportion to k_code * k_case
# 执行任务被分成的块数（太小可能会卡住），应该与 k_code * k_case 成比例
num_chunks = 512

# if provide public test example in unit test generation prompt
no_example = True

# max ground-truth unit test we can use for evaluation here
# 这里用于评估的最大真实单元测试数量
max_test = 8

# if output process for execution (sometimes execution may take a very long time, such like LiveCodeBench, given its long time limit)
exe_verbose = True

# set True by default here, no need to change
is_final_eval = True

# set False by default, unless for some specific deepseek models (like coder models)
trust_remote_code = False

# 不同模式,用于控制生成的结果和日志文件的名字
mode = "plan_search_code_only_pass_at_k_debug"

# 生成模式：original / plansearch / original_resample
generation_mode = "plansearch"   #  "original", "original_resample", "plansearch"

# 评测模式：oneshot / passatk / bon
eval_mode = "bon"                # "passatk", "oneshot", "bon" 

# 是否使用 PlanSearch 分阶段模式
use_multi_stage_generation = (generation_mode == "plansearch")

# 是否使用原始一体化重复采样模式（你新加的那个）
use_original_resample = (generation_mode == "original_resample")

# 是否是 one-shot
single_eval = (eval_mode == "oneshot")

# 是否是 pass@k-only 模式
eval_pass_at_k_only = (eval_mode == "passatk")

# 是否是 BoN 模式（用 LLM UT + 真实 UT）
eval_bon = (eval_mode == "bon")

# 要计算哪些 k 的 pass@k
# 想同时测pass@k和bon需要传入这个参数，同时设置eval_mode为bon
pass_at_k_list = [10, 50, 100]

# ======================== Random UT clustering for BoN ========================  # random UT 聚类配置分组
# 是否启用 random UT 聚类选码（仅在 eval_mode=bon 时生效）  # 开关说明
use_random_ut_cluster = True  # 是否启用 random UT 聚类
# 每题 random UT input 总数（用于聚类选码）  # 总量说明
random_ut_total = 80  # 每题 random UT input 总数
# 每轮使用的 random UT 数量（分 5 轮）  # 分批说明
random_ut_batch = 16  # 每轮使用的 random UT 数
# 生成 random UT input 的最大重试轮数  # 重试说明
random_ut_max_attempts = 5  # 最多重试轮数
# top-pass 代码数量达到该值才执行聚类  # 触发阈值说明
random_ut_min_top_count = 1  # top-pass 最小数量
# 聚类距离阈值（0 表示输出完全一致才连边）  # 聚类距离说明
random_ut_cluster_max_diff = 0  # 距离阈值
# 兜底占位符（解析失败且重试耗尽时填充）  # 兜底说明
random_ut_placeholder = "We can not extract the input in the output. "  # 占位符文本

# plansearch最大轮数
max_global_rounds = 50

# 一阶和二阶观察最大值
max_obs = 4

# 在生成自然语言解法时是否能看到一阶观察
use_first_order_obs = False

# 是否使用伪代码模块 plansearch:"False" or ours:"True"
use_pseudocode_module = True

# prompt 使用模式（角色设定在system：system，角色设定在user：user）
prompt_role_mode = 2  # ours:"user" or plansearch:"system" or "system+add"

# 反思所能看到的东西（只能看到前面的想法，不能看到观察：False；都能看到：True）
reflection_visibility = True  # plansearch:"False" or ours:"True"

# 是否启用 Stage 3.5 Critique Plan 生成（批判性思考和替代解法）
use_critique_plan = False  # True: 启用 critique plan，False: 禁用

# 是否使用所有二阶观察分别生成自然语言解法（不使用子集）
use_all_second_order_obs = True  # True: 直接用每个二阶观察生成，False: 使用子集

# 消融实验
ablation = "only_stage2" # only_stage1: 只用一阶观察生成代码 only_stage2: 只用二阶观察生成代码 natural_description: 使用自然语言生成代码

# 是否生成想法层面攻击UT
use_idea_attack_ut = True

# 想法层面攻击UT数量
num_ideas = 1

# 是否打印出大模型prompt和输出做调试
verbose_logging = False

# 自一致性验证重复采样数量
self_consistency_num = 4

# 自博弈轮数
self_play_round = 1

# UT 投票模式
ut_vote_by_code = False

# 是否使用自博弈模块
use_self_play = False

# ======================== 自博弈参数 ========================
# ut通过率
ut_accuracy_target = 0.5

# 全否ut通过率
ut_regen_max_attempts = 5

skip_attack_when_all_pass = True
# ======================== config for vllm inference model (use_api = False) ========================


# vllm model name
# vLLM 模型名称
pretrained_model = "Qwen/Qwen2.5-0.5B-Instruct"

# maximum number of tokens the vLLM engine can handle in a single sequence
# vLLM 引擎在单个序列中可以处理的最大 token 数量
max_model_len = 8192 # 20000

# max token model can generate for each quiry
# 模型对每个查询生成的最大 token 数量
max_generation_token = 10000

# inference temperature
# 推理温度
temp = 0.8

# GPU usage for vllm inference, [[0]] represents only one engine with one GPU; [[0, 1], [2, 3]] represents two engines each with 2 GPUs
# each engine loads a model, so for <=7B model, you can only set GPU numbers for each engine <= 2
# vLLM 推理的 GPU 使用情况，[[0]] 表示只有一个引擎使用一个 GPU；[[0, 1], [2, 3]] 表示两个引擎，每个使用 2 个 GPU
# 每个引擎加载一个模型，所以对于 <=7B 的模型，你只能为每个引擎设置 <= 2 个 GPU
gpu_groups = [[0,1],[2,3]]














# ======================== config for API inference model (use_api = True) ========================


# api_key and base_url
api_key = "sk-2ff631128d994a6c97b6c26ed5f022e1"
# base_url 可以带或不带 /v1/chat/completions，代码会自动处理
# OpenAI:   "https://api.openai.com" 或 "https://api.openai.com/v1/chat/completions"
# DeepSeek: "https://api.deepseek.com" 或 "https://api.deepseek.com/v1/chat/completions"
base_url = "https://api.deepseek.com"

# api model name, such like "gpt-4o", "deepseek-chat"
api_model_name = "deepseek-chat"

# temperature
api_temperature = 0.8

# max inquiries submitted at one time
# max inquiries submitted at one time
# 一次提交的最大查询数量
max_workers = 50

# if it's OpenAI's model, and your account is available for batch inference, recommend setting this to be True, it's cheaper
use_openai_batch_api = False

# max token can generate for each task
max_tokens = 2500

# the request per minute limit for your API
# the request per minute limit for your API
# 你的 API 每分钟请求限制
rpm_limit = 250


# evaluation_config.py (底部)

# ======================= Prompt Registry (JSON-like Structure) ============================
# 结构说明:
# Keys: prompt_role_mode (int)
# Values: 具体 Prompt 的字典
# 对于受 bool 开关影响的 Prompt (如 stage3_critique, stage4)，使用 {True: "...", False: "..."} 存储

PROMPT_REGISTRY = {
    0: {
        "stage1": """<|im_start|>system
  You are an expert Python programmer.
  <|im_end|>
  <|im_start|>user
  You will be given a competitive programming question.
  This is the problem:

  {{problem}}

  You will return several useful, non-obvious, and correct observations about the problem,
  like hints to solve the problem.
  List them as numbered points.
  You will NOT return any code. Be creative and go beyond obvious insights.

  Output format (VERY IMPORTANT):
  - Write one observation per line.
  - Start each line with a number and a dot, like:
    1. first observation
    2. second observation
    3. third observation
  - Do NOT add any extra text, headings, or explanations before or after the list.
  Just output the numbered observations, nothing else.
  <|im_end|>
  <|im_start|>assistant
  """,
        "stage2": """<|im_start|>system
  You are an expert Python programmer.
  <|im_end|>
  <|im_start|>user
  You will be given an competitive programming question (problem specification) and several correct observations about the problem.
  This is the problem:
  {{problem}}

  Here are the first-order observations:
  {{first_order_observations}}

  You will brainstorm several new, useful, and correct observations about the problem, derived from the given observations. 
  You will NOT return any code. Be as creative as possible, going beyond what you think is intuitively correct.

  Output format (VERY IMPORTANT):
  - Write one derived observation per line.
  - Start each line with a number and a dot, like:
    1. first derived observation
    2. second derived observation
    3. third derived observation
  - Do NOT add any extra text, headings, or explanations before or after the list.
  Just output the numbered observations, nothing else.<|im_end|>
  <|im_start|>assistant: """,
        "stage3": """<|im_start|>system
  You are an expert Python programmer designing a solution to a competitive programming question..
  <|im_end|>
  <|im_start|>user
  This is the problem:
  {{problem}}
  Here are the intelligent observations to help solve the problem:
  {{first_order_observations}}

  {{second_order_observations}}

  Use these observations above to brainstorm a natural language solution to the problem above. 
  Note that your intuition may lead you astray, so come up with simple, creative ideas that go beyond what you would usually come up with and exceeds your narrow intuition. 
  Quote relevant parts of the observations EXACTLY before each step of the solution. QUOTING IS CRUCIAL.
  <|im_end|>
  <|im_start|>assistant: """,
        # 对应 reflection_visibility
        "stage3_critique": {
            True: """<|im_start|>system
  You are an expert competitive programming analyst.
  Your job is to critique reasoning and propose alternative solution directions.
  <|im_end|>
  <|im_start|>user
  We are solving the following competitive programming problem:
  {{problem}}

  The following observations were used to propose a solution:
  {{first_order_observations}}

  {{second_order_observations}}

  Here is a proposed natural-language solution plan:
  {{solution_plan}}

  Your task:
  1. **Assume the above solution is incorrect or suboptimal.**
  2. Provide a detailed critique explaining potential flaws, logical errors, missing cases, or inefficiencies.
  3. Based on your criticism, generate a **new alternative natural-language solution idea**.
  4. This new idea must differ meaningfully from the original and explore a different reasoning direction.
  5. DO NOT provide code.

  Be creative, constructive, and insightful.
  <|im_end|>
  <|im_start|>assistant
  """,
            False: """<|im_start|>system
  You are an expert competitive programming analyst.
  Your job is to critique reasoning and propose alternative solution directions.
  <|im_end|>
  <|im_start|>user
  We are solving the following competitive programming problem:
  {{problem}}

  Here is a proposed natural-language solution plan:
  {{solution_plan}}

  Your task:
  1. **Assume the above solution is incorrect or suboptimal.**
  2. Provide a detailed critique explaining potential flaws, logical errors, missing cases, or inefficiencies.
  3. Based on your criticism, generate a **new alternative natural-language solution idea**.
  4. This new idea must differ meaningfully from the original and explore a different reasoning direction.
  5. DO NOT provide code.

  Be creative, constructive, and insightful.
  <|im_end|>
  <|im_start|>assistant
  """
        },
        "stage3_75": """<|im_start|>system
  You are an expert Python programmer.
  <|im_end|>
  <|im_start|>user
  You will be given a question (problem specification) and a natural language solution plan.
  You will generate correct pseudocode that follows the above solution exactly.
  You will NOT return anything for the pseudocode inside Markdown codeblocks
  We are solving the following competitive programming problem:
  {{problem}}
  Here is the natural-language solution plan:
  {{solution_plan}}
  Your task: 
  1. Carefully write pseudocode that follows the above solution EXACTLY and solves the problem. 
  2. The pseudocode does not need to go into low-level details, and should be high-level.
  <|im_end|>
  <|im_start|>assistant
  """,
        # 对应 use_pseudocode_module
        "stage4": {
            True: """<|im_start|>You are a helpful assistant help user solve problems. \
  <|im_end|>
  <|im_start|>User: You need to write {{language}} script. {{special_requirements}}
  This is the problem:
  {{problem}}

  Here is the pseudocode:
  {{solution_plan}}

  Your task: Now write the final {{language}} script that implements this pseudocode exactly.
  Keep it clean, efficient, and consistent with {{special_requirements}}.
  Include minimal inline comments when necessary.
  <|im_end|>
  <|im_start|>Assistant: """,
            False: """<|im_start|>You are a helpful assistant help user solve problems. \
  <|im_end|>
  <|im_start|>User: You need to write {{language}} script. {{special_requirements}}
  This is the problem:
  {{problem}}

  Here is the natural-language solution plan:
  {{solution_plan}}

  Your task: Now write the final {{language}} script that implements this plan exactly. 
  Keep it clean, efficient, and consistent with {{special_requirements}}.
  Include minimal inline comments that map to the plan when relevant.
  <|im_end|>
  <|im_start|>Assistant: """
        },
        "original": """<|im_start|>You are a helpful assistant help user solve problems. \
  <|im_end|>\n<|im_start|>User: You need to think first then write {{language}} script. {{special_requirements}}
  This is the problem:\n{{problem}} <|im_end|>\n<|im_start|>Assistant: """,
        "case": """<|im_start|>You are a helpful assistant help user generate test examples for coding tasks. \
  <|im_end|>\n<|im_start|>User: Given a coding task, instead of providing the final script, your task is to generate a new test example (both input, output and explanation).
  This is the problem:\n{{problem}}\n
  {{example_intro}}
  You need to provide a new test example. A good test example should be completely accurate and conform to the problem’s format requirements, while also possessing enough discriminative power to distinguish correct code from incorrect code.
  Before providing a test example, you must think carefully and reason step by step to derive an input and output you are very confident are correct. For example, start by designing an input you can reliably handle, then compute the output step by step. If you're unsure about the output, revise or re-design the input to ensure accuracy. Directly providing input/output pairs without this process is discouraged, as it often results in low accuracy.
  Finally, after completing these previous thinking and derivation steps (you should not write the final test example unless you have gone through these steps very thoroughly), you MUST put your final test example in the following format:\n
  **Test Input:**\n```input here```\n\n**Test Output:**\n```output here```\n\n**Explanation:**\n\nexplanation here.\n <|im_end|>\n<|im_start|>Assistant: """,
        "special_requirements": """You should use input() to input and print() to output in your script. Your code should output the results based on the input read in, rather than generating the given test example."""
    },

    # ========================== Mode 2 ==========================
    2: {
        "stage1": """<|im_start|>system
  You are an expert Python programmer.
  You will be given a competitive programming question.
  You will return several useful, non-obvious, and correct observations about the problem,
  like hints to solve the problem.
  List them as numbered points.
  You will NOT return any code. Be creative and go beyond obvious insights.
  <|im_end|>
  <|im_start|>user
  This is the problem:

  {{problem}}

  Output format (VERY IMPORTANT):
  - Write one observation per line.
  - Start each line with a number and a dot, like:
    1. first observation
    2. second observation
    3. third observation
  - Do NOT add any extra text, headings, or explanations before or after the list.
  Just output the numbered observations, nothing else.
  <|im_end|>
  <|im_start|>assistant
  """,
        "stage2": """<|im_start|>system
  You are an expert Python programmer.
  You will be given an competitive programming question (problem specification) and several correct observations about the problem.
  You will brainstorm several new, useful, and correct observations about the problem, derived from the given observations. 
  You will NOT return any code. Be as creative as possible, going beyond what you think is intuitively correct.

  <|im_end|>
  <|im_start|>user
  This is the problem:
  {{problem}}

  Here are the first-order observations:
  {{first_order_observations}}

  Output format (VERY IMPORTANT):
  - Write one derived observation per line.
  - Start each line with a number and a dot, like:
    1. first derived observation
    2. second derived observation
    3. third derived observation
  - Do NOT add any extra text, headings, or explanations before or after the list.
  Just output the numbered observations, nothing else.<|im_end|>
  <|im_start|>assistant: """,
        "stage3": """<|im_start|>system
  You are an expert Python programmer designing a solution to a competitive programming question..
  You will be given an algorithmic question (problem specification) and some observations which are necessary to solve the problem. You will return a high-level, natural language solution to the question,
  like an editorial, which uses the observations given. You will NOT return any code. Be as creative as possible, going beyond what you think is intuitively correct.
  <|im_end|>
  <|im_start|>user
  This is the problem:
  {{problem}}
  Here are the intelligent observations to help solve the problem:
  {{first_order_observations}}

  {{second_order_observations}}

  Use these observations above to brainstorm a natural language solution to the problem above. 
  Note that your intuition may lead you astray, so come up with simple, creative ideas that go beyond what you would usually come up with and exceeds your narrow intuition. 
  Quote relevant parts of the observations EXACTLY before each step of the solution. QUOTING IS CRUCIAL.
  <|im_end|>
  <|im_start|>assistant: """,
        "stage3_critique": {
            True: """<|im_start|>system
  You are an expert competitive programming analyst.
  Your job is to critique reasoning and propose alternative solution directions.
  You will be given a competitive programming question (problem specification).
  A student has come up with a proposed idea to solve the problem, but it is incorrect.
  You will also be given correct fixes to the idea. 
  Incorporate the fixes into the original idea to make it correct.
  MAKE SURE not to output any code.
  <|im_end|>
  <|im_start|>user
  We are solving the following competitive programming problem:
  {{problem}}

  The following observations were used to propose a solution:
  {{first_order_observations}}

  {{second_order_observations}}

  Here is a proposed natural-language solution plan:
  {{solution_plan}}

  Your task:
  1. **Assume the above solution is incorrect or suboptimal.**
  2. Provide a detailed critique explaining potential flaws, logical errors, missing cases, or inefficiencies.
  3. Based on your criticism, generate a **new alternative natural-language solution idea**.
  4. This new idea must differ meaningfully from the original and explore a different reasoning direction.
  5. DO NOT provide code.

  Be creative, constructive, and insightful.
  <|im_end|>
  <|im_start|>assistant
  """,
            False: """<|im_start|>system
  You are an expert competitive programming analyst.
  Your job is to critique reasoning and propose alternative solution directions.
  You will be given a competitive programming question (problem specification).
  A student has come up with a proposed idea to solve the problem, but it is incorrect.
  You will also be given correct fixes to the idea. 
  Incorporate the fixes into the original idea to make it correct.
  MAKE SURE not to output any code.
  <|im_end|>
  <|im_start|>user
  We are solving the following competitive programming problem:
  {{problem}}

  Here is a proposed natural-language solution plan:
  {{solution_plan}}

  Your task:
  1. **Assume the above solution is incorrect or suboptimal.**
  2. Provide a detailed critique explaining potential flaws, logical errors, missing cases, or inefficiencies.
  3. Based on your criticism, generate a **new alternative natural-language solution idea**.
  4. This new idea must differ meaningfully from the original and explore a different reasoning direction.
  5. DO NOT provide code.

  Be creative, constructive, and insightful.
  <|im_end|>
  <|im_start|>assistant
  """
        },
        "stage3_75": """<|im_start|>system
  You are an expert Python programmer.
  You will be given a question (problem specification) and a natural language solution plan.
  You will generate correct pseudocode that follows the above solution exactly.
  You will NOT return anything for the pseudocode inside Markdown codeblocks
  <|im_end|>
  <|im_start|>user
  We are solving the following competitive programming problem:
  {{problem}}
  Here is the natural-language solution plan:
  {{solution_plan}}
  Your task: 
  1. Carefully write pseudocode that follows the above solution EXACTLY and solves the problem. 
  2. The pseudocode does not need to go into low-level details, and should be high-level.
  <|im_end|>
  <|im_start|>assistant
  """,
        "stage4": {
            True: """<|im_start|>You are a helpful assistant help user solve problems. \
      You will be given a question (problem specification) and high-level pseudocode that describes how to solve the problem. 
      You will generate a correct Python program that matches said pseudocode and passes all tests. 
      You will NOT return anything except for the program inside Markdown codeblocks.
  <|im_end|>
  <|im_start|>User: You need to write {{language}} script. {{special_requirements}}
  This is the problem:
  {{problem}}

  Here is the pseudocode:
  {{solution_plan}}

  Your task: Now write the final {{language}} script that implements this pseudocode exactly.
  Keep it clean, efficient, and consistent with {{special_requirements}}.
  Include minimal inline comments when necessary.
  <|im_end|>
  <|im_start|>Assistant: """,
            False: """<|im_start|>You are a helpful assistant help user solve problems. \
      You are an expert Python programmer. 
      You will be given a question (problem specification) and a natural language solution/tutorial that describes how to solve the problem. 
      You will generate a correct Python program that matches said specification and tutorial and passes all tests. 
      You will NOT return anything except for the program inside markdown codeblocks.
  <|im_end|>
  <|im_start|>User: You need to write {{language}} script. {{special_requirements}}
  This is the problem:
  {{problem}}

  Here is the natural-language solution plan:
  {{solution_plan}}

  Your task: Now write the final {{language}} script that implements this plan exactly. 
  Keep it clean, efficient, and consistent with {{special_requirements}}.
  Include minimal inline comments that map to the plan when relevant.
  <|im_end|>
  <|im_start|>Assistant: """
        },
        "original": """<|im_start|>You are a helpful assistant help user solve problems. \
  <|im_end|>\n<|im_start|>User: You need to think first then write {{language}} script. {{special_requirements}}
  This is the problem:\n{{problem}} <|im_end|>\n<|im_start|>Assistant: """,
        "case": """<|im_start|>You are a helpful assistant help user generate test examples for coding tasks. \
  <|im_end|>\n<|im_start|>User: Given a coding task, instead of providing the final script, your task is to generate a new test example (both input, output and explanation).
  This is the problem:\n{{problem}}\n
  {{example_intro}}
  You need to provide a new test example. A good test example should be completely accurate and conform to the problem’s format requirements, while also possessing enough discriminative power to distinguish correct code from incorrect code.
  Before providing a test example, you must think carefully and reason step by step to derive an input and output you are very confident are correct. For example, start by designing an input you can reliably handle, then compute the output step by step. If you're unsure about the output, revise or re-design the input to ensure accuracy. Directly providing input/output pairs without this process is discouraged, as it often results in low accuracy.
  Finally, after completing these previous thinking and derivation steps (you should not write the final test example unless you have gone through these steps very thoroughly), you MUST put your final test example in the following format:\n
  **Test Input:**\n```input here```\n\n**Test Output:**\n```output here```\n\n**Explanation:**\n\nexplanation here.\n <|im_end|>\n<|im_start|>Assistant: """,
        "special_requirements": """You should use input() to input and print() to output in your script. Your code should output the results based on the input read in, rather than generating the given test example."""
    },

    # ========================== Mode 1 (Same as Mode 0/2 hybrid in your input) ==========================
    1: {
         "stage1": """<|im_start|>system
  You are an expert Python programmer.
  You will be given a competitive programming question.
  You will return several useful, non-obvious, and correct observations about the problem,
  like hints to solve the problem.
  List them as numbered points.
  You will NOT return any code. Be creative and go beyond obvious insights.
  <|im_end|>
  <|im_start|>user
  This is the problem:

  {{problem}}

  Output format (VERY IMPORTANT):
  - Write one observation per line.
  - Start each line with a number and a dot, like:
    1. first observation
    2. second observation
    3. third observation
  - Do NOT add any extra text, headings, or explanations before or after the list.
  Just output the numbered observations, nothing else.
  <|im_end|>
  <|im_start|>assistant
  """,
        "stage2": """<|im_start|>system
  You are an expert Python programmer.
  You will be given an competitive programming question (problem specification) and several correct observations about the problem.
  You will brainstorm several new, useful, and correct observations about the problem, derived from the given observations. 
  You will NOT return any code. Be as creative as possible, going beyond what you think is intuitively correct.

  <|im_end|>
  <|im_start|>user
  This is the problem:
  {{problem}}

  Here are the first-order observations:
  {{first_order_observations}}

  Output format (VERY IMPORTANT):
  - Write one derived observation per line.
  - Start each line with a number and a dot, like:
    1. first derived observation
    2. second derived observation
    3. third derived observation
  - Do NOT add any extra text, headings, or explanations before or after the list.
  Just output the numbered observations, nothing else.<|im_end|>
  <|im_start|>assistant: """,
        "stage3": """<|im_start|>system
  You are an expert Python programmer designing a solution to a competitive programming question..
  <|im_end|>
  <|im_start|>user
  This is the problem:
  {{problem}}
  Here are the intelligent observations to help solve the problem:
  {{first_order_observations}}

  {{second_order_observations}}

  Use these observations above to brainstorm a natural language solution to the problem above. 
  Note that your intuition may lead you astray, so come up with simple, creative ideas that go beyond what you would usually come up with and exceeds your narrow intuition. 
  Quote relevant parts of the observations EXACTLY before each step of the solution. QUOTING IS CRUCIAL.
  <|im_end|>
  <|im_start|>assistant: """,
        "stage3_critique": {
            True: """<|im_start|>system
  You are an expert competitive programming analyst.
  Your job is to critique reasoning and propose alternative solution directions.
  <|im_end|>
  <|im_start|>user
  We are solving the following competitive programming problem:
  {{problem}}

  The following observations were used to propose a solution:
  {{first_order_observations}}

  {{second_order_observations}}

  Here is a proposed natural-language solution plan:
  {{solution_plan}}

  Your task:
  1. **Assume the above solution is incorrect or suboptimal.**
  2. Provide a detailed critique explaining potential flaws, logical errors, missing cases, or inefficiencies.
  3. Based on your criticism, generate a **new alternative natural-language solution idea**.
  4. This new idea must differ meaningfully from the original and explore a different reasoning direction.
  5. DO NOT provide code.

  Be creative, constructive, and insightful.
  <|im_end|>
  <|im_start|>assistant
  """,
            False: """<|im_start|>system
  You are an expert competitive programming analyst.
  Your job is to critique reasoning and propose alternative solution directions.
  <|im_end|>
  <|im_start|>user
  We are solving the following competitive programming problem:
  {{problem}}

  Here is a proposed natural-language solution plan:
  {{solution_plan}}

  Your task:
  1. **Assume the above solution is incorrect or suboptimal.**
  2. Provide a detailed critique explaining potential flaws, logical errors, missing cases, or inefficiencies.
  3. Based on your criticism, generate a **new alternative natural-language solution idea**.
  4. This new idea must differ meaningfully from the original and explore a different reasoning direction.
  5. DO NOT provide code.

  Be creative, constructive, and insightful.
  <|im_end|>
  <|im_start|>assistant
  """
        },
        "stage3_75": """<|im_start|>system
  You are an expert Python programmer.
  <|im_end|>
  <|im_start|>user
  You will be given a question (problem specification) and a natural language solution plan.
  You will generate correct pseudocode that follows the above solution exactly.
  You will NOT return anything for the pseudocode inside Markdown codeblocks
  We are solving the following competitive programming problem:
  {{problem}}
  Here is the natural-language solution plan:
  {{solution_plan}}
  Your task: 
  1. Carefully write pseudocode that follows the above solution EXACTLY and solves the problem. 
  2. The pseudocode does not need to go into low-level details, and should be high-level.
  <|im_end|>
  <|im_start|>assistant
  """,
        "stage4": {
            True: """<|im_start|>You are a helpful assistant help user solve problems. \
  <|im_end|>
  <|im_start|>User: You need to write {{language}} script. {{special_requirements}}
  This is the problem:
  {{problem}}

  Here is the pseudocode:
  {{solution_plan}}

  Your task: Now write the final {{language}} script that implements this pseudocode exactly.
  Keep it clean, efficient, and consistent with {{special_requirements}}.
  Include minimal inline comments when necessary.
  <|im_end|>
  <|im_start|>Assistant: """,
            False: """<|im_start|>You are a helpful assistant help user solve problems. \
  <|im_end|>
  <|im_start|>User: You need to write {{language}} script. {{special_requirements}}
  This is the problem:
  {{problem}}

  Here is the natural-language solution plan:
  {{solution_plan}}

  Your task: Now write the final {{language}} script that implements this plan exactly. 
  Keep it clean, efficient, and consistent with {{special_requirements}}.
  Include minimal inline comments that map to the plan when relevant.
  <|im_end|>
  <|im_start|>Assistant: """
        },
        "original": """<|im_start|>You are a helpful assistant help user solve problems. \
  <|im_end|>\n<|im_start|>User: You need to think first then write {{language}} script. {{special_requirements}}
  This is the problem:\n{{problem}} <|im_end|>\n<|im_start|>Assistant: """,
        "case": """<|im_start|>You are a helpful assistant help user generate test examples for coding tasks. \
  <|im_end|>\n<|im_start|>User: Given a coding task, instead of providing the final script, your task is to generate a new test example (both input, output and explanation).
  This is the problem:\n{{problem}}\n
  {{example_intro}}
  You need to provide a new test example. A good test example should be completely accurate and conform to the problem’s format requirements, while also possessing enough discriminative power to distinguish correct code from incorrect code.
  Before providing a test example, you must think carefully and reason step by step to derive an input and output you are very confident are correct. For example, start by designing an input you can reliably handle, then compute the output step by step. If you're unsure about the output, revise or re-design the input to ensure accuracy. Directly providing input/output pairs without this process is discouraged, as it often results in low accuracy.
  Finally, after completing these previous thinking and derivation steps (you should not write the final test example unless you have gone through these steps very thoroughly), you MUST put your final test example in the following format:\n
  **Test Input:**\n```input here```\n\n**Test Output:**\n```output here```\n\n**Explanation:**\n\nexplanation here.\n <|im_end|>\n<|im_start|>Assistant: """,
        "special_requirements": """You should use input() to input and print() to output in your script. Your code should output the results based on the input read in, rather than generating the given test example."""
    },
    
    
    
    
    
    
    
    
    
    
    
###### ========================== Mode Stage1_refine (New) ========================== ######
      3:{
            
###### stage1 
       "stage1": """<|im_start|>system
  You are an expert Python programmer.
  
  You will be given a competitive programming question.
  You will return several useful, non-obvious, and correct observations about the problem,
  like hints to solve the problem.
  List them as numbered points.
  You will NOT return any code. Be creative and go beyond obvious insights.
  <|im_end|>
  <|im_start|>user
  This is the problem:

  {{problem}}

  Output format (VERY IMPORTANT):
  - Write one observation per line.
  - Start each line with a number and a dot, like:
    1. first observation
    2. second observation
    3. third observation
  - Do NOT add any extra text, headings, or explanations before or after the list.
  Just output the numbered observations, nothing else.
<|im_end|>
<|im_start|>assistant
  """,
  
  
  
###### stage2 ######
        "stage2": """<|im_start|>system
  You are an expert Python programmer.
  <|im_end|>
  <|im_start|>user
  You will be given an competitive programming question (problem specification) and several correct observations about the problem.
  This is the problem:
  {{problem}}

  Here are the first-order observations:
  {{first_order_observations}}

  You will brainstorm several new, useful, and correct observations about the problem, derived from the given observations. 
  You will NOT return any code. Be as creative as possible, going beyond what you think is intuitively correct.

  Output format (VERY IMPORTANT):
  - Write one derived observation per line.
  - Start each line with a number and a dot, like:
    1. first derived observation
    2. second derived observation
    3. third derived observation
  - Do NOT add any extra text, headings, or explanations before or after the list.
  Just output the numbered observations, nothing else.<|im_end|>
  <|im_start|>assistant: """,




###### stage3 ######
        "stage3": """<|im_start|>system
  You are an expert Python programmer designing a solution to a competitive programming question.
  <|im_end|>
  <|im_start|>user
  Use following observations to brainstorm a step by step natural language solution plan to the problem. 
  You should NOT output any code.
  This is the problem:
  {{problem}}
  Here are the intelligent observations to help solve the problem:
  {{second_order_observations}}

  <|im_end|>
  <|im_start|>assistant: """,
        # 对应 reflection_visibility




###### stage3_critique ######
        "stage3_critique": {
            True: """<|im_start|>system
  You are an expert competitive programming analyst.
  Your job is to critique reasoning and propose alternative solution directions.
  <|im_end|>
  <|im_start|>user
  We are solving the following competitive programming problem:
  {{problem}}

  The following observations were used to propose a solution:
  {{first_order_observations}}

  {{second_order_observations}}

  Here is a proposed natural-language solution plan:
  {{solution_plan}}

  Your task:
  1. **Assume the above solution is incorrect or suboptimal.**
  2. Provide a detailed critique explaining potential flaws, logical errors, missing cases, or inefficiencies.
  3. Based on your criticism, generate a **new alternative natural-language solution idea**.
  4. This new idea must differ meaningfully from the original and explore a different reasoning direction.
  5. DO NOT provide code.

  Be creative, constructive, and insightful.
  <|im_end|>
  <|im_start|>assistant
  """,
            False: """<|im_start|>system
  You are an expert competitive programming analyst.
  Your job is to critique reasoning and propose alternative solution directions.
  <|im_end|>
  <|im_start|>user
  We are solving the following competitive programming problem:
  {{problem}}

  Here is a proposed natural-language solution plan:
  {{solution_plan}}

  Your task:
  1. **Assume the above solution is incorrect or suboptimal.**
  2. Provide a detailed critique explaining potential flaws, logical errors, missing cases, or inefficiencies.
  3. Based on your criticism, generate a **new alternative natural-language solution idea**.
  4. This new idea must differ meaningfully from the original and explore a different reasoning direction.
  5. DO NOT provide code.

  Be creative, constructive, and insightful.
  <|im_end|>
  <|im_start|>assistant
  """
        },




###### stage3_75 ######
        "stage3_75": """<|im_start|>system
  You are an expert Python programmer.
  <|im_end|>
  <|im_start|>user
  You will be given a question (problem specification) and a natural language solution plan.
  You will generate correct pseudocode that follows the above solution exactly.
  You will NOT return anything for the pseudocode inside Markdown codeblocks
  We are solving the following competitive programming problem:
  {{problem}}
  Here is the natural-language solution plan:
  {{solution_plan}}
  Your task: 
  1. Carefully write pseudocode that follows the above solution EXACTLY and solves the problem. 
  2. The pseudocode does not need to go into low-level details, and should be high-level.
  <|im_end|>
  <|im_start|>assistant
  """,
  
  
  
  
###### stage4 ######
        # 对应 use_pseudocode_module
        "stage4": {
            True: """<|im_start|>You are a helpful assistant help user solve problems. \
  <|im_end|>
  <|im_start|>User: You need to write {{language}} script. {{special_requirements}}
  This is the problem:
  {{problem}}

  Here is the pseudocode:
  {{solution_plan}}

  Your task: Now write the final {{language}} script that implements this pseudocode exactly.
  Keep it clean, efficient, and consistent with {{special_requirements}}.
  Include minimal inline comments when necessary.
  <|im_end|>
  <|im_start|>Assistant: """,
            False: {


###### 自然语言描述生成代码 ######
                  "natural_desciption": """<|im_start|>You are a helpful assistant help user solve problems. \
  <|im_end|>
  <|im_start|>User: You need to write Python script that implements this plan exactly. Your code should output the results based on the input read in, rather than generating the given test example.
  This is the problem:
  {{problem}}

  Here is the natural-language solution plan:
  {{solution_plan}}

  <|im_end|>
  <|im_start|>Assistant: """,

###### 只用一阶观察生成代码 ######
                  "only_stage1": """<|im_start|>You are a helpful assistant help user solve problems. \
  <|im_end|>
  <|im_start|>User: You need to think first and write the final Python script based on the observations. 
  This is the problem:
  {{problem}}

  Here are the observations:
  {{first_order_observations}}

  You should use input() to input and print() to output in your script. Your code should output the results based on the input read in, rather than generating the given test example.
  <|im_end|>
  <|im_start|>Assistant: """,

  ###### 只用二阶观察生成代码 ######
                  "only_stage2": """<|im_start|>You are a helpful assistant help user solve problems. \
  <|im_end|>
  <|im_start|>User: You need to write Python script based on the observations. 
  You should use input() to input and print() to output in your script. Your code should output the results based on the input read in, rather than generating the given test example.
  This is the problem:
  {{problem}}

  Here are intelligent observations:
  {{first_order_observations}}

  <|im_end|>
  <|im_start|>Assistant: """},
        },
        
        
        
###### original ######
        "original": """<|im_start|>You are a helpful assistant help user solve problems. \
  <|im_end|>\n<|im_start|>User: You need to think first then write {{language}} script. {{special_requirements}}
  This is the problem:\n{{problem}} <|im_end|>\n<|im_start|>Assistant: """,
#         "original": """
# A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: 
# # Task: You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. Your final answer should be wrapped in ```python``` tags.

# Question: {{problem}}

# Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.
# ```python
# # YOUR CODE HERE
# ```

# Assistant: <think>
# """,

        
###### case ######
        "case": """<|im_start|>You are a helpful assistant help user generate test examples for coding tasks. \
  <|im_end|>\n<|im_start|>User: Given a coding task, instead of providing the final script, your task is to generate a new test example (both input, output and explanation).
  This is the problem:\n{{problem}}\n
  {{example_intro}}
  You need to provide a new test example. A good test example should be completely accurate and conform to the problem’s format requirements, while also possessing enough discriminative power to distinguish correct code from incorrect code.
  Before providing a test example, you must think carefully and reason step by step to derive an input and output you are very confident are correct. For example, start by designing an input you can reliably handle, then compute the output step by step. If you're unsure about the output, revise or re-design the input to ensure accuracy. Directly providing input/output pairs without this process is discouraged, as it often results in low accuracy.
  Finally, after completing these previous thinking and derivation steps (you should not write the final test example unless you have gone through these steps very thoroughly), you MUST put your final test example in the following format:\n
  **Test Input:**\n```input here```\n\n**Test Output:**\n```output here```\n\n**Explanation:**\n\nexplanation here.\n <|im_end|>\n<|im_start|>Assistant: """,
        "special_requirements": """You should use input() to input and print() to output in your script. Your code should output the results based on the input read in, rather than generating the given test example."""
    }
}




