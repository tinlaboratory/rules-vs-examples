
from prompts.markdown_loader import render_markdown_prompt


OP_FUNC_BASE = [
    "1. Operator function is composed of two operators, and inputs are integer.",
    "2. For the operator function @(x,y,z), x, y, z are the inputs of the function.",
    "3. First, multiply y to the x.",
    "4. Then, add z to the result from the previous operation."
]

OP_FUNC_LV2 = [
    "1. Operator function is composed of three operators, and inputs are integer. ",
    "2. For the operator function #(x,y,z,a), x,y,z,a are the inputs of the function.",
    "3. First, subtract y from x.",
    "4. Second, multiply the result from the previous operation by z.",
    "5. Last, adds a to the result of the previous operation."

]

OP_FUNC_LV3 = [
    "1. Operator function is composed of four operators, and inputs are integer.",
    "2. For the operator function $(x,y,z,a,b), x,y,z,a,b are the inputs of the function.",
    "3. First, adds y to x.",
    "4. Second, multiply the result from the previous operation by z.",
    "5. Third, subtract a from the result of the previous operation.",
    "6. Last, divide the result by b."
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
        output = f"({x},{y},{z},{a},{b}) → ({x+y},{z},{a},{b}) → ({(x+y)*z},{a},{b}) → ({(x+y)*z-a},{b}) → {((x+y)*z-a)/b}"
    return output

def build_problem_string(difficulty: int, inputs: list) -> str:
    """Build the problem operator string for display."""
    if difficulty == 1:
        problem = "@("+ ",".join(map(str, inputs)) + ")"
    elif difficulty == 2:
        problem = "#("+ ",".join(map(str, inputs)) + ")"
    else:
        problem = "$("+ ",".join(map(str, inputs)) + ")"
    return problem

def get_rule_based_prompt(difficulty, problem):
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
        #prompt += f"\({build_problem_string(difficulty, inp)}\)=\(\\boxed{{{ans}}}\)\n"
        prompt += f"\({build_problem_string(difficulty, inp)}\)={op_func(inp, difficulty)}=\(\\boxed{{{ans}}}\)\n"

    prompt += f"\({build_problem_string(difficulty, problem)}\)=\n"
    prompt += "Answer (place the result in \\boxed{}):"
    return prompt


def get_combined_prompt(difficulty, problem, examples):
    prompt = "You will be shown a problem using operator function and need to find the answer for the problem.\n\n"
    prompt += "RULES:\n"
    if difficulty == 1:
        prompt += "".join(OP_FUNC_BASE)
    elif difficulty == 2:
        prompt += "".join(OP_FUNC_LV2)
    elif difficulty == 3:
        prompt += "".join(OP_FUNC_LV3)

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
        prompt += f"\({build_problem_string(difficulty, inp)}\) = \(\\boxed{{{ans}}}\)\n"

    prompt += f"Problem: \({build_problem_string(difficulty, problem)}\)\n"
    prompt += "Answer (place the result in \\boxed{}):\\boxed{}"
    return prompt
