# Prompt construction for original and PlanSearch generation modes.
from jinja2 import Template
from itertools import combinations
import re


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
    problem = data_i["question"]

    if method == "sample":

        return Template(system_prompts).render(
            language="python",
            special_requirements=special_requirements,
            problem=problem,
        )

    if method == "case":

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


# PlanSearch prompt stages.
def get_stage1_prompt(problem, tmpl):
    return Template(tmpl).render(problem=problem)


def get_stage2_prompt(problem, first_order_observations, tmpl):
    return Template(tmpl).render(
        problem=problem,
        first_order_observations=first_order_observations,
    )


def init_eval_fields_for_data(data):
    for i in range(len(data)):
        data[i]["full_code_generation"] = []
        data[i]["full_case_generation"] = []
        data[i]["generated_code"] = []
        data[i]["case_input"] = []
        data[i]["case_output"] = []
        data[i]["case_text"] = []
        

        data[i]["idea_generation_outputs"] = []
        data[i]["attack_ideas"] = []
        data[i]["case_input_original"] = []
        data[i]["case_output_original"] = []
        data[i]["case_is_valid"] = []


        data[i]["stage1_observations"] = None
        data[i]["stage2_observations"] = None


# Dataset-level prompt builders.
def build_code_prompts_original_eval(data, args):
    init_eval_fields_for_data(data)

    code_generation_prompts = []
    code_index = []

    num = len(data)
    print("Using original unified generation mode", flush=True)

    for i in range(num):
        data_i = data[i].copy()
        problem = data_i["question"]


        prompt_sample = get_scaling_prompt(
            data_i,
            "sample",
            k_case=args.k_case,
            system_prompts=args.system_prompts,
            system_case_prompts=args.system_case_prompts,
            special_requirements=args.special_requirements,
            no_example=args.no_example,
        )
        data[i]["code_generation_prompt"] = prompt_sample


        code_generation_prompts += [prompt_sample] * args.k_code
        code_index += [i] * args.k_code

    print(
        f"✓ Prompt generation complete: {len(code_generation_prompts)} code prompts",
        flush=True,
    )

    return code_generation_prompts, code_index


def build_code_prompts_plansearch(data, args):
    init_eval_fields_for_data(data)

    num = len(data)
    stage1_prompts = []

    print("Using PlanSearch staged generation mode", flush=True)

    for i in range(num):
        problem = data[i]["question"]


        prompt_stage1 = get_stage1_prompt(problem, args.system_prompts_stage1)


        data[i]["code_generation_prompts"] = {
            "stage1": prompt_stage1,
            "stage2_template": problem,
        }

        stage1_prompts.append(prompt_stage1)

    print(
        f"✓ PlanSearch: built Stage1 prompts for {num} problems",
        flush=True,
    )

    return stage1_prompts

def build_case_prompts(data, args):
    case_generation_prompts = []
    case_index = []

    num = len(data)

    for i in range(num):
        data_i = data[i].copy()


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
        f"✓ Test prompt generation complete: {len(case_generation_prompts)} test prompts",
        flush=True,
    )

    return case_generation_prompts, case_index


def build_prompts_for_dataset(data, args):

    if args.use_multi_stage_generation:


        _ = build_code_prompts_plansearch(data, args)


        code_generation_prompts = []
        code_index = []
    else:

        code_generation_prompts, code_index = build_code_prompts_original_eval(data, args)


    case_generation_prompts, case_index = build_case_prompts(data, args)

    return code_generation_prompts, code_index, case_generation_prompts, case_index
