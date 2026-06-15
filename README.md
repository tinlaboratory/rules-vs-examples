<div align="center">

# LLMs Learn Better In-Context from Rules than from Examples

**Xiang Fu**¹*, **Seungmin Cho**¹*, **Yukyung Lee**¹, **Najoung Kim**¹

¹Boston University

\* Equal contribution

[Paper](#) | [Dataset](https://huggingface.co/datasets/tin-lab/rules_vs_examples)

</div>

This repository accompanies the paper "LLMs Learn Better In-Context from Rules than from Examples".

> **TL;DR:** LLMs learn better in-context from rules than from examples, but there are substantial model and task effects.

> **Abstract:** Large language models (LLMs) exhibit in-context learning capabilities, where they can learn new tasks from prompt contexts without weight updates. We compare the learning efficacies of two prominent modes of in-context learning: (1) learning from descriptions of *rules* (instruction following); and (2) learning from *examples* of input-output demonstrations (few-shot prompting). Through five learning tasks that cover diverse domains (games, arithmetic, linguistic inferences), we compare the learning efficacies of LLMs when the two modes of learning (rules vs. examples) specify the same underlying task. We furthermore explore model and task properties that modulate the learning efficacies. We find that models generally learn more reliably from rules than from examples alone, and additional examples on top of rules or simply scaling up the number of examples do not lead to consistent and significant gains. Instruction tuning amplifies the benefit of rule-based learning while keeping example-based learning capacities intact. Surprisingly, we find no privileged effect of example-based learning in base models, and rules still lead to gains in algebraic task domains. Overall, the comparative efficacy of rules over examples is larger when the task recruits algebraic abstractions and computations, and smaller when the task requires distributional sensitivity and/or recruits parametric knowledge.

### Repository Structure

```bash
.
├── code/                             # Experiment runners, tasks, prompts, and utilities
│   ├── main.py
│   ├── prompts/
│   ├── tasks/
│   └── utils/
├── data_generation/                  # Data generation scripts
│   ├── generate_lexical_category_inference.py
│   ├── generate_noun_class_agreement.py
│   ├── generate_operator_function.py
│   ├── generate_set_game.py
│   └── generate_tapatan.py
├── rules-mode-prompts/               # Markdown source files for rules-mode prompts
│   ├── lexical_category_inference.md
│   ├── noun_class_agreement.md
│   ├── operator_function.md
│   ├── set_game.md
│   └── tapatan.md
└── requirements.txt
```

The `rules-mode-prompts/` directory contains the markdown prompt sources used for rules-mode prompting. The executable task implementations live under `code/tasks/`.

### Included Tasks

* Set Game
* Tapatan
* Operator Function
* Noun Class Agreement
* Lexical Category Inference

Tapatan includes both move-sequence and final-board-state input variants. Lexical Category Inference includes the shared, AND, and OR logic conditions used in the paper.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset

The dataset is stored separately from this code repository. After cloning or downloading the dataset repository, point the code to its `data/` directory:

```bash
export RULES_VS_EXAMPLES_DATA_DIR=/path/to/rules-vs-examples-dataset/data
```

If the dataset repository is checked out next to this repository as `rules-vs-examples-dataset`, the code will also discover it automatically.

### Running Experiments

The experiment runner supports the three learning modes used in the paper: `rules`, `examples`, and `combined`.

For local or cluster runs with downloaded models served through vLLM:

```bash
export RULES_VS_EXAMPLES_DATA_DIR=/path/to/rules-vs-examples-dataset/data
export ROR_MODEL_ROOT=/path/to/downloaded/models

python code/main.py \
  --inference_backend vllm \
  --model qwen3-14b \
  --task tapatan \
  --mode combined \
  --tapatan_input_format final_board_state
```

For the OpenAI API:

```bash
export RULES_VS_EXAMPLES_DATA_DIR=/path/to/rules-vs-examples-dataset/data
export OPENAI_API_KEY=...

python code/main.py \
  --inference_backend api \
  --model gpt-5.4 \
  --api_token_param max_completion_tokens \
  --max_tokens 128 \
  --task tapatan \
  --mode combined \
  --tapatan_input_format final_board_state
```

For GPT-5.4, `--api_token_param max_completion_tokens` makes the runner send OpenAI's current Chat Completions token-budget parameter.

### Notes

* `combined` is the learning mode that provides both rules and examples.
* Tapatan includes two input formats: `move_sequence` and `final_board_state`.
* Lexical Category Inference includes the `shared`, `AND`, and `OR` logic conditions.
* The dataset is stored separately on Hugging Face rather than in this GitHub repository.
