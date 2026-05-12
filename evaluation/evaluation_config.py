# ======================== Evaluation configuration ========================

# Set True to evaluate an API model, False to evaluate a local/vLLM model.
use_api = True

# Dataset name is resolved as ../CURE_data/<dataset>.json.
dataset = "CodeContests"

# Number of candidate programs and generated unit tests per task.
k_code = 8
k_case = 8

# BoN settings. Runtime is controlled by k_code and k_case.
scale_tuple_list = [(4, 4), (8, 8)]

# Execution settings.
num_chunks = 512
no_example = True
max_test = 8
exe_verbose = True
is_final_eval = True
trust_remote_code = False
verbose_logging = False

# Run mode.
mode = "plan_search_code_only_pass_at_k_debug"
generation_mode = "plansearch"  # original / original_resample / plansearch
eval_mode = "bon"  # oneshot / passatk / bon
use_multi_stage_generation = generation_mode == "plansearch"
use_original_resample = generation_mode == "original_resample"
single_eval = eval_mode == "oneshot"
eval_pass_at_k_only = eval_mode == "passatk"
eval_bon = eval_mode == "bon"
pass_at_k_list = [10, 50, 100]

# Random UT clustering for BoN.
use_random_ut_cluster = True
random_ut_total = 80
random_ut_batch = 16
random_ut_max_attempts = 5
random_ut_min_top_count = 1
random_ut_cluster_max_diff = 0
random_ut_placeholder = "We can not extract the input in the output. "

# Final CosPlay method settings.
max_global_rounds = 50
max_obs = 4
prompt_role_mode = 3
use_all_second_order_obs = True
ablation = "only_stage2"  # only_stage1 / only_stage2
use_idea_attack_ut = True
num_ideas = 1
self_consistency_num = 4
self_play_round = 1
ut_vote_by_code = False
use_self_play = False
ut_accuracy_target = 0.5
ut_regen_max_attempts = 5
skip_attack_when_all_pass = True
is_empty = True

# ======================== vLLM model configuration ========================
pretrained_model = "Qwen/Qwen2.5-0.5B-Instruct"
max_model_len = 8192
max_generation_token = 10000
temp = 0.8
gpu_groups = [[0, 1], [2, 3]]

# ======================== API model configuration ========================
# Keep these as placeholders for open-source release.
api_key = "TODO"
base_url = "https://api.deepseek.com"
api_model_name = "deepseek-chat"
api_temperature = 0.8
max_workers = 50
use_openai_batch_api = False
max_tokens = 2500
rpm_limit = 250


STAGE1_PROMPT = """<|im_start|>system
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
"""


STAGE2_PROMPT = """<|im_start|>system
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
<|im_start|>assistant: """


ONLY_STAGE1_CODE_PROMPT = """<|im_start|>You are a helpful assistant help user solve problems. \
<|im_end|>
<|im_start|>User: You need to think first and write the final Python script based on the observations.
This is the problem:
{{problem}}

Here are the observations:
{{first_order_observations}}

You should use input() to input and print() to output in your script. Your code should output the results based on the input read in, rather than generating the given test example.
<|im_end|>
<|im_start|>Assistant: """


ONLY_STAGE2_CODE_PROMPT = """<|im_start|>You are a helpful assistant help user solve problems. \
<|im_end|>
<|im_start|>User: You need to write Python script based on the observations.
You should use input() to input and print() to output in your script. Your code should output the results based on the input read in, rather than generating the given test example.
This is the problem:
{{problem}}

Here are intelligent observations:
{{first_order_observations}}

<|im_end|>
<|im_start|>Assistant: """


ORIGINAL_CODE_PROMPT = """<|im_start|>You are a helpful assistant help user solve problems. \
<|im_end|>
<|im_start|>User: You need to think first then write {{language}} script. {{special_requirements}}
This is the problem:
{{problem}} <|im_end|>
<|im_start|>Assistant: """


CASE_PROMPT = """<|im_start|>You are a helpful assistant help user generate test examples for coding tasks. \
<|im_end|>
<|im_start|>User: Given a coding task, instead of providing the final script, your task is to generate a new test example (both input, output and explanation).
This is the problem:
{{problem}}

{{example_intro}}
You need to provide a new test example. A good test example should be completely accurate and conform to the problem's format requirements, while also possessing enough discriminative power to distinguish correct code from incorrect code.
Before providing a test example, you must think carefully and reason step by step to derive an input and output you are very confident are correct. For example, start by designing an input you can reliably handle, then compute the output step by step. If you're unsure about the output, revise or re-design the input to ensure accuracy. Directly providing input/output pairs without this process is discouraged, as it often results in low accuracy.
Finally, after completing these previous thinking and derivation steps (you should not write the final test example unless you have gone through these steps very thoroughly), you MUST put your final test example in the following format:

**Test Input:**
```input here```

**Test Output:**
```output here```

**Explanation:**

explanation here.
<|im_end|>
<|im_start|>Assistant: """


SPECIAL_REQUIREMENTS = "You should use input() to input and print() to output in your script. Your code should output the results based on the input read in, rather than generating the given test example."


FINAL_PROMPTS = {
    "stage1": STAGE1_PROMPT,
    "stage2": STAGE2_PROMPT,
    "stage4": {
        "only_stage1": ONLY_STAGE1_CODE_PROMPT,
        "only_stage2": ONLY_STAGE2_CODE_PROMPT,
    },
    "original": ORIGINAL_CODE_PROMPT,
    "case": CASE_PROMPT,
    "special_requirements": SPECIAL_REQUIREMENTS,
}

# Keep numeric prompt modes accepted for compatibility, but point all of them
# to the same compact final-method prompt set.
PROMPT_REGISTRY = {
    0: FINAL_PROMPTS,
    1: FINAL_PROMPTS,
    2: FINAL_PROMPTS,
    3: FINAL_PROMPTS,
}
