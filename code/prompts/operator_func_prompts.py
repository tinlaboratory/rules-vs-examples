
from prompts.markdown_loader import render_markdown_prompt


HEADER = "You will be shown a problem that uses a newly defined operator function based on the following rules and need to calculate the answer for the problem.\n\n"


OP_FUNC_RULE_BASE = [
    "1. The new operator function f is composed of two arithmetic operators, and its inputs are integers.\n",
    "2. Let x, y, z be inputs of the function.\n",
    "3. To compute f(x,y,z), first multiply x and y.\n",
    "4. Then, add z to the result from the previous operation.\n"
]

OP_FUNC_RULE_LV2 = [
    "1. The new operator function g is composed of three arithmetic operators, and its inputs are integers.\n",
    "2. Let x, y, z, a be inputs of the function.\n",
    "3. To compute g(x,y,z,a), first subtract y from x.\n",
    "4. Second, multiply the result from the previous operation by z.\n",
    "5. Last, adds a to the result of the previous operation.\n"

]

OP_FUNC_RULE_LV3 = [
    "1. The new operator function h is composed of four arithmetic operators, and its inputs are integers.\n",
    "2. Let x, y, z, a, b be inputs of the function.\n",
    "3. To compute h(x,y,z,a,b), first add x and y.\n",
    "4. Second, multiply the result from the previous operation by z.\n",
    "5. Third, subtract a from the result of the previous operation.\n",
    "6. Last, take the remainder when the result is divided by b.\n"
]

def op_func(input, difficulty):
    if difficulty ==1:
        x,y,z = input
        output = f"({x},{y},{z}) → ({x*y},{z}) → {(x*y)+z}"
    elif difficulty ==2:
        x, y, z, a = input
        output = f"({x},{y},{z},{a}) → ({x-y},{z},{a}) → ({(x-y)*z},{a}) → {((x-y)*z)+a}"
    else:
        x, y, z, a, b = input
        output = f"({x},{y},{z},{a},{b}) → ({x+y},{z},{a},{b}) → ({(x+y)*z},{a},{b}) → ({(x+y)*z-a},{b}) → {((x+y)*z-a)%b}"
    return output

def build_problem_string(difficulty: int, inputs: list) -> str:
    """Build the problem operator string for display."""
    if difficulty == 1:
        problem = "f("+ ",".join(map(str, inputs)) + ")"
    elif difficulty == 2:
        problem = "g("+ ",".join(map(str, inputs)) + ")"
    else:
        problem = "h("+ ",".join(map(str, inputs)) + ")"
    return problem

def get_rule_based_prompt(difficulty, problem):
    prompt = HEADER
    prompt += "RULES:\n"
    if difficulty == 1:
        prompt += "".join(OP_FUNC_RULE_BASE)
    elif difficulty == 2:
        prompt += "".join(OP_FUNC_RULE_LV2)
    elif difficulty == 3:
        prompt += "".join(OP_FUNC_RULE_LV3)
    prompt += f"Problem: \({build_problem_string(difficulty, problem)}\)\n"
    prompt += "Answer (place the result in \\boxed{YOUR_ANSWER}):"

    return prompt
    return render_markdown_prompt(
        "operator_function.md",
        ("Rules", f"Difficulty {difficulty}"),
        problem=build_problem_string(difficulty, problem),
    )

def get_example_based_prompt(difficulty, problem, examples):
    prompt = ""
    if isinstance(examples, list):
        # Flatten all input_all entries across all example dicts
        all_examples = []
        for ex in examples:
            if isinstance(ex, dict) and "input_all" in ex:
                all_examples.extend(ex["input_all"])
            else:
                all_examples.append(ex)  # already a (inp, ans) tuple
    else:
        all_examples = examples["input_all"]

    for inp, ans in all_examples:
        prompt += f"\({build_problem_string(difficulty, inp)}\)={op_func(inp, difficulty)}=\(\\boxed{{{ans}}}\)\n"

    prompt += f"\({build_problem_string(difficulty, problem)}\)="
    return prompt


def get_combined_prompt(difficulty, problem, examples):
    prompt = HEADER
    prompt += "RULES:\n"
    if difficulty == 1:
        prompt += "".join(OP_FUNC_RULE_BASE)
    elif difficulty == 2:
        prompt += "".join(OP_FUNC_RULE_LV2)
    elif difficulty == 3:
        prompt += "".join(OP_FUNC_RULE_LV3)

    prompt += "\nFor example,\n"
    if isinstance(examples, list):
        # Flatten all input_all entries across all example dicts
        all_examples = []
        for ex in examples:
            if isinstance(ex, dict) and "input_all" in ex:
                all_examples.extend(ex["input_all"])
            else:
                all_examples.append(ex)  # already a (inp, ans) tuple
    else:
        all_examples = examples["input_all"]
        
    for inp, ans in all_examples:
        prompt += f"\({build_problem_string(difficulty, inp)}\)={op_func(inp, difficulty)}=\(\\boxed{{{ans}}}\)\n"

    prompt += f"Problem: \({build_problem_string(difficulty, problem)}\)\n"
    prompt += "Answer (place the result in \\boxed{}):\\boxed{}"
    return prompt
