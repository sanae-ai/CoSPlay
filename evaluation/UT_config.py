# Prompt constants and builders shared by unit-test generation and self-play.
SYSTEM = "<|im_start|>system\n"

USER = "<|im_start|>user\n"

ASSISTANT = "<|im_start|>assistant\n"

END = "\n<|im_end|>\n"

SYSTEM_PROMPT_UT_IDEA = "You are an expert Software Engineering Tester designing unit test to uncover the potential bugs."

SYSTEM_PROMPT_UT_INPUT = "You are an expert Software Engineering Tester designing unit test to uncover the potential bugs."

SYSTEM_PROMPT_UT_OUTPUT = "You are an expert Software Engineering Tester designing unit test to uncover the potential bugs."

SYSTEM_PROMPT_FIX_CODE = "You are an expert programmer designing code for competitive programming question."

def get_full_prompt(list_prompt: list) -> str:
    return SYSTEM + list_prompt[0]["content"] + END + USER + list_prompt[1]["content"] + END + ASSISTANT


def splicing_mode(mode: str) -> str:
    if mode == "only_stage2":
        return "is the intelligent observation:"
    elif mode == "only_stage1":
        return "is the observation:"
    raise ValueError(f"Unsupported UT idea mode: {mode}")

def splicing_text(mode: str) -> str:
    if mode == "only_stage2":
        return "an observation"
    elif mode == "only_stage1":
        return "an observation"
    raise ValueError(f"Unsupported UT idea mode: {mode}")


# Idea-level attack prompts.
def get_ut_idea_user_content(mode: str, num_ideas: int, problem: str, input_mode: str) -> str:
    return f"""You will be given a competitive programming question and """ + splicing_text(mode) + f""" about the problem, which will be used to generate code to solve the problem,
you should brainstorm several attack ideas at the same idea level, which will unveil potential design flaws, security risks, and edge case predictions.
This is the problem:
{problem}

Here """ + splicing_mode(mode) + f""" 
{input_mode}

Output format (VERY IMPORTANT):
  - Write one attack idea per line.
  - Start each line with a number and a dot, like:
    1. first attack idea
    2. second attack idea
    3. third attack idea
  - Do NOT add any extra text, headings, or explanations before or after the list.
  Just output the numbered observations, nothing else."""

def get_ut_idea_prompt(mode: str, num_ideas: int, problem: str, input_mode: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT_UT_IDEA},
        {"role": "user", "content": get_ut_idea_user_content(mode, num_ideas, problem, input_mode)}
    ]


# Unit-test input/output prompts.
def get_ut_input_user_content(problem: str, attact_idea: str) -> str:
    return f"""
# Role
You are an expert Quality Assurance Engineer and Competitive Programmer.\
Your task is to generate a strict "Unit Test Input" based on a provided "Attack Idea".

# Problem Description
```
{problem}
```

# Attack Idea
```
{attact_idea}
```

# Instructions (Chain of Thought)
To ensure the generated input is valid and effective, follow these steps strictly:

1.  **Analyze Constraints & Hidden Rules**:
    *   Identify all explicit constraints (e.g., $1 \\le N \\le 10^5$, sum of weights < $10^9$, the string should be re).
    *   **Crucial**: Identify *implicit* constraints in the text (e.g., "distinct integers", "connected graph", "permutation", "tree structure").
    *   List these constraints clearly in your explanation.

2.  **Map Attack to Constraints**:
    *   How can you implement the `attack_idea` while strictly obeying ALL constraints listed above?
    *   If the attack requires a large number, ensure it fits within the variable limits.

3.  **Construct & Verify**:
    *   Construct the test case input step-by-step.
    *   **Self-Correction**: Check the generated input against the constraints one last time. Does the number of elements match $N$? Are the formats (spaces/newlines) correct?

4.  **Final Output**:
    *   Produce the explanation, and finally the raw input block.
After constructing the input, explicitly show how it can be decomposed back into the components described in the problem. If you cannot show this, discard the input.


# Output Format
You must structure your response exactly as follows:

**Explanation:**
[Step 1: List constraints found in the problem...]
[Step 2: Reasoning on how to apply the attack idea within these constraints...]
[Step 3: Verification of the constructed input...]

**Test Input:**
[The raw input data ONLY. No comments. Strictly follow formatting.]
**Let's think step by step.**
"""

def get_ut_input_generation_prompt(problem: str, attact_idea: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT_UT_INPUT},
        {"role": "user", "content": get_ut_input_user_content(problem, attact_idea)}
    ]


def get_ut_output_user_content(problem: str, ut_input: str) -> str:
    return f"""
# Role
You are an expert in Competitive Programming. Your specific task is to generate the correct output for a programming problem based on a unit test.

# Problem
```
{problem}
```

# Unit Test Input
```
{ut_input}
```

# IMPORTANT INSTRUCTIONS
1. **Roleplay**: Act as a code execution engine.
2. **Chain of Thought**: You MUST write an "Explanation" section first. Trace the code logic step-by-step with the provided input.
3. **Format**: After the explanation, output the result inside a strict block.
4. Use only the given unit test input; if it seems mismatched to the problem format, do not invent missing data.
5. Match the EXAMPLE's output format exactly (spacing/line breaks/order); no brackets/commas unless shown; empty output -> blank line. 

# Response Format
**Explanation:**
[Your step-by-step logic tracing here]

**Test output:**
[Raw Output Data ONLY]
**Let's think step by step.**
"""

def get_ut_output_generation_prompt(problem: str, ut_input: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT_UT_OUTPUT},
        {"role": "user", "content": get_ut_output_user_content(problem, ut_input)} 
    ]


def get_ut_output_refine_user_content(problem: str, ut_input: str, previous_ut: str, previous_code: str) -> str:
    return f"""
# Role
You are an expert in Competitive Programming. Your task is to regenerate the correct output for a given unit test input.

# Problem
```
{problem}
```

# Unit Test Input
```
{ut_input}
```

# Previous Attempt (for reference, may contain format issues or logical errors, please improve)
```
{previous_code}
```

```
{previous_ut}
```

# IMPORTANT INSTRUCTIONS
1. **Roleplay**: Act as a code execution engine.
2. **Chain of Thought**: You MUST write an "Explanation" section first. Trace the code logic step-by-step with the provided input.
3. **Format**: After the explanation, output the result inside a strict block.
4. Recompute strictly from the given input; if the input format seems off, do not invent missing data. Do NOT copy the previous attempt.
5. Match the EXAMPLE's output format exactly (spacing/line breaks/order); no brackets/commas unless shown; empty output -> blank line. 

# Response Format
**Explanation:**
[Your step-by-step logic tracing here]

**Test output:**
[Raw Output Data ONLY]
**Let's think step by step.**
"""

def get_ut_output_refine_prompt(problem: str, ut_input: str, previous_ut: str, previous_code: str = "") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT_UT_OUTPUT},
        {"role": "user", "content": get_ut_output_refine_user_content(problem, ut_input, previous_ut, previous_code)},
    ]


import random

# Repair prompts used during self-play.
def code_fix_user_content_with_ut(
    problem: str, 
    failed_code: str, 
    attack_ut_input: list, 
    attack_ut_output: list, 
    exe_output: list, 
    num_to_include: int = None
) -> str:
    

    total_uts = len(attack_ut_input)
    indices = list(range(total_uts))
    

    if num_to_include is not None and 0 < num_to_include < total_uts:
        selected_indices = random.sample(indices, num_to_include)
    else:
        selected_indices = indices
    

    ut_details_list = []
    for i in selected_indices:
        inp = attack_ut_input[i]
        exp = attack_ut_output[i]
        exe = exe_output[i]
        
        detail = (
            f"--- Test Case {i+1} ---\n"
            f"Input:\n{inp}\n"
            f"Expected Output:\n{exp}\n"
            f"Actual Execution Output / Error Trace:\n{exe}\n"
        )
        ut_details_list.append(detail)
    

    failed_uts_text = "\n".join(ut_details_list)


    return f"""
# Role
You are an expert Python Debugger and Algorithm Engineer. Your task is to fix the provided code which failed specific unit tests.

# Problem Description
```
{problem}
```

# The Buggy Code
```python
{failed_code}
```
Failure Report
The code failed the following test case(s). Note that the "Actual Execution Output" might contain a runtime error (Traceback) or simply an incorrect calculation result.
```
{failed_uts_text}
```
**Debugging Instructions (Chain of Thought)**
To fix the code correctly, you MUST follow these steps:
1. **Analyze the Failure Type:**
    * Case A: Runtime Error (Traceback): If the output contains words like Error, Exception, or Traceback, identify strictly WHICH line caused the crash and WHY (e.g., index out of bounds, division by zero, recursion limit).
    * Case B: Logical Error (Wrong Answer): If the code ran but produced different output than expected, compare the Expected vs Actual. Trace the logic to find where the calculation diverged.
2. **Identify the Root Cause:**
    * Don't just look at the input values. Ask yourself: "What logical flaw in the algorithm allows this specific input to break the code?"
    * Is it an edge case (e.g., N=1, empty list)?
    * Is it a variable initialization issue?
3. **Refine the Code:**
    * Fix the logic to handle the general case, not just the specific failed input.
    * DO NOT simply add if input == ...: return ... (Hardcoding is strictly forbidden).
    * Ensure all necessary libraries (e.g., sys, math) are imported.

**Output Format**
**Analysis:**
[Step 1 & 2: Explain the error type (Runtime/Logic) and the specific bug location]
**Refined Code:**
```
[The fully fixed, runnable code]
```
**Let's think step by step to find the bug.**
"""


def get_fix_code_prompt_with_ut(
    problem: str, 
    failed_code: str, 
    attack_ut_input: list, 
    attack_ut_output: list, 
    exe_output: list, 
    num_to_include: int) -> list[dict[str, str]]:

    return [
        {"role": "system", "content": SYSTEM_PROMPT_FIX_CODE},
        {"role": "user", "content": code_fix_user_content_with_ut(problem, failed_code, attack_ut_input, attack_ut_output, exe_output, num_to_include)}
    ]


def code_fix_user_content_with_ut_refine(
    problem: str,
    failed_code: str,
    attack_ut_input: list,
    attack_ut_output: list,
    exe_output: list,
    previous_fix_output: str,
    num_to_include: int = None,
) -> str:
    total_uts = len(attack_ut_input)
    indices = list(range(total_uts))
    if num_to_include is not None and 0 < num_to_include < total_uts:
        selected_indices = random.sample(indices, num_to_include)
    else:
        selected_indices = indices

    ut_details_list = []
    for i in selected_indices:
        inp = attack_ut_input[i]
        exp = attack_ut_output[i]
        exe = exe_output[i]
        detail = (
            f"--- Test Case {i+1} ---\n"
            f"Input:\n{inp}\n"
            f"Expected Output:\n{exp}\n"
            f"Actual Execution Output / Error Trace:\n{exe}\n"
        )
        ut_details_list.append(detail)
    failed_uts_text = "\n".join(ut_details_list)

    prev_output_text = previous_fix_output if previous_fix_output else "[not available]"

    return f"""
# Role
You are an expert Python Debugger and Algorithm Engineer. Your task is to fix the provided code which failed specific unit tests.

# Problem Description
```
{problem}
```

# The Buggy Code
```python
{failed_code}
```
Failure Report
The code failed the following test cases. Note that the "Actual Execution Output" might contain a runtime error (Traceback) or simply an incorrect calculation result.
```
{failed_uts_text}
```

# Previous Fix Attempt (FAILED)
Note: The previous fix attempt failed. Use it as reference and avoid regressions.
Previous Raw Output (full text):
```
{prev_output_text}
```

**Debugging Instructions (Chain of Thought)**
To fix the code correctly, you MUST follow these steps:
1. **Analyze the Failure Type:**
    * Case A: Runtime Error (Traceback): If the output contains words like Error, Exception, or Traceback, identify strictly WHICH line caused the crash and WHY (e.g., index out of bounds, division by zero, recursion limit).
    * Case B: Logical Error (Wrong Answer): If the code ran but produced different output than expected, compare the Expected vs Actual. Trace the logic to find where the calculation diverged.
2. **Identify the Root Cause:**
    * Don't just look at the input values. Ask yourself: "What logical flaw in the algorithm allows this specific input to break the code?"
    * Is it an edge case (e.g., N=1, empty list)?
    * Is it a variable initialization issue?
3. **Refine the Code:**
    * Fix the logic to handle the general case, not just the specific failed input.
    * DO NOT simply add if input == ...: return ... (Hardcoding is strictly forbidden).
    * Ensure all necessary libraries (e.g., sys, math) are imported.
4. **Preserve Previously-Passing Behavior:**
    * Avoid breaking unit tests that already passed before the fix attempt.

**Output Format**
**Analysis:**
[Step 1 & 2: Explain the error type (Runtime/Logic) and the specific bug location]
**Refined Code:**
```
[The fully fixed, runnable code]
```
**Let's think step by step to find the bug.**
"""


def get_fix_code_prompt_with_ut_refine(
    problem: str,
    failed_code: str,
    attack_ut_input: list,
    attack_ut_output: list,
    exe_output: list,
    previous_fix_output: str,
    num_to_include: int,
) -> list[dict[str, str]]:

    return [
        {"role": "system", "content": SYSTEM_PROMPT_FIX_CODE},
        {
            "role": "user",
            "content": code_fix_user_content_with_ut_refine(
                problem,
                failed_code,
                attack_ut_input,
                attack_ut_output,
                exe_output,
                previous_fix_output,
                num_to_include,
            ),
        },
    ]

def ut_fix_user_content_with_code(
    problem: str, 
    failed_code: str, 
    attack_ut_input: list, 
    attack_ut_output: list, 
    exe_output: list, 
    num_to_include: int = None
) -> str:
    

    total_uts = len(attack_ut_input)
    indices = list(range(total_uts))
    

    if num_to_include is not None and 0 < num_to_include < total_uts:
        selected_indices = random.sample(indices, num_to_include)
    else:
        selected_indices = indices
    

    ut_details_list = []
    for i in selected_indices:

        inp = attack_ut_input[i]
        exp = attack_ut_output[i]
        exe = exe_output[i]
        
        detail = f"ut input: ```{inp}```, expected output: ```{exp}```, execution output and the Execution Trace: ```{exe}```"
        ut_details_list.append(detail)
    

    failed_uts_text = "\n".join(ut_details_list)


    return f"""
Your previous ut failed the best code, detailed information are as follows:
**Problem:**
{problem}

**Failed Code:**
```python
{failed_code}
Unit Test Case(s) that caused failure:
{failed_uts_text}
Output ONLY the new refined output inside a code block.
Refined Output:
```

```
"""


def get_fix_ut_prompt_with_code(
    problem: str, 
    failed_code: str, 
    attack_ut_input: list, 
    attack_ut_output: list, 
    exe_output: list, 
    num_to_include: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT_FIX_CODE},
        {"role": "user", "content": ut_fix_user_content_with_code(problem, failed_code, attack_ut_input, attack_ut_output, exe_output, num_to_include)}
    ]
    


# Random unit-test input prompts and validation prompts.
def get_ut_input_random_user_content(problem: str, num_cases: int) -> str:
    return f"""Your task is to generate EXACTLY {num_cases} NEW RANDOM and VALID test inputs for a competitive programming problem. 
You should not repeat any example inputs in the problem.

**Instructions (STRICT):**
1. Each line represents ONE independent test input.
2. Each test input must strictly follow the input format and constraints of the problem.
3. Each line MUST start with the exact prefix: CASE|
4. After CASE|, output ONLY the raw input values in correct order. Do NOT include parameter names, variable labels, code, or any explanatory text.


**Problem:**
{problem}

You MUST output in the following EXACT format:
CASE|```<input for test case>```
"""


def get_ut_input_random_generation_prompt(problem: str, num_cases: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT_UT_INPUT},
        {"role": "user", "content": get_ut_input_random_user_content(problem, num_cases)}
    ]


def check_input_format_user(problem: str, former_ut_input: list, ut_input: str) -> str:
    input_list_text = "\n".join([str(item) for item in former_ut_input])
    return f"""
Your task is to validate the provided unit test input for a specific programming problem. 

**Validation Checklist:**
1. **Uniqueness:** The input must NOT duplicate any entry in the "Former inputs" list.
2. **Compliance:** The input must strictly adhere to the problem's requirements. 
   - **Explicit Requirements:** Follow data types, variable counts, and value ranges specified in the text.
   - **Implicit Requirements:** Ensure logical consistency (e.g., graph connectivity, array lengths matching N) that may not be explicitly stated but are necessary for a valid test case.

**Instructions:**
- If the "New unit test input" fails any of the criteria above, explain exactly which requirement was violated. Then, modify the input to create a valid, unique test case input.
- If the input is already valid and unique, confirm its validity in the explanation and output the input unchanged.

**Output Format:**
**Explanation:**
```
(Explain why the input was rejected or accepted. If rejected, specify the violated rule.)
```
**New unit test input:**
```
(The corrected valid input, or the original input if it was already valid)
```
Problem Description:
{problem}
Former inputs:
{input_list_text}
New unit test input:
{ut_input}
"""
 

def get_check_input_format_prompt(problem: str, former_ut_input: list, ut_input: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT_UT_INPUT},
        {"role": "user", "content": check_input_format_user(problem, former_ut_input, ut_input)}
    ]


