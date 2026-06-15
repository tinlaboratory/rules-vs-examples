import random
from itertools import permutations
import argparse
from pathlib import Path
import json
from collections import defaultdict

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

def operator_func_base(input):
    """ mul -> plus """
    x,y,z = input
    output = (x * y) + z
    return output

def operator_func_lv2(input):
    """ minus -> mul -> plus """
    x,y,z,a = input
    output = ((x-y)*z) + a
    return output

def operator_func_lv3(input):
    """ plus -> mul -> minus -> mod """
    x,y,z,a,b = input
    output = (((x+y)*z)-a)%b
    return output


def example_generator(difficulty, train_needed, test_needed):
    train_examples = []
    test_examples = []
    existing_sets = set()

    input_range = list(range(-20, 0)) + list(range(1, 21))

    if difficulty == 1:
        input_num = 3
        op_func = operator_func_base
        
    elif difficulty == 2:
        input_num = 4
        op_func = operator_func_lv2
    else:
        input_num = 5
        op_func = operator_func_lv3
    
    
    total_needed = train_needed + test_needed
    attempts = 0
    max_attempts = total_needed * 100

    while len(existing_sets) < total_needed and attempts < max_attempts:
        attempts += 1
        inputs = tuple(sorted(random.sample(input_range, input_num)))
        if inputs in existing_sets:
            continue
        existing_sets.add(inputs)
    
    existing_sets = list(existing_sets)
    train_sets = existing_sets[:train_needed]
    test_sets = existing_sets[train_needed:]

    #train and test set formatting
    for t in train_sets:
        train_case = {"input_sorted": [],
                      "input_all":[]}
        train_case["input_sorted"] = t
        for inp in permutations(t):
            answer = op_func(inp)
            train_case["input_all"].append((inp, answer))
        train_examples.append(train_case)


    for test in test_sets:
        random.shuffle(list(test))
        answer = op_func(test)
        test_case = {"input":test, "answer":answer}
        test_examples.append(test_case)

    print("Total needed: ", total_needed)
    print("Train size: ", (len(train_examples)))
    print("Test size: ", (len(test_examples)))
    return train_examples, test_examples


def main():
    parser = argparse.ArgumentParser(description='Generate operator function examples')
    parser.add_argument('--train_samples', type=int, default=1000,
                      help='Number of training samples to generate')
    parser.add_argument('--test_samples', type=int, default=500,
                      help='Number of test samples to generate')
    parser.add_argument('--difficulty', type=int, default=1,
                      help='Difficulty of task')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument(
        '--output_root',
        type=str,
        default=str(WORKSPACE_ROOT / "data" / "op_func"),
        help='Output root directory. Defaults to <repo>/data/op_func.',
    )
    args = parser.parse_args()

    random.seed(args.seed)
    
    train_data, test_data = example_generator(args.difficulty,  args.train_samples, args.test_samples)

    # formatting train data file
    train_d = {
            'task': 'operator function',
            'difficulty': args.difficulty,
            'dataset_size': len(train_data),
            'data': train_data
        }

    # formatting test data file
    test_d = {
            'task': 'operator function',
            'difficulty': args.difficulty,
            'dataset_size': len(test_data),
            'data': test_data
        }

    # Create output directories
    output_root = Path(args.output_root)
    train_dir = output_root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    test_dir = output_root / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train file
    train_file = train_dir / f"op_func_train_{args.difficulty}.json"
    with open(train_file, 'w') as f:
        json.dump(train_d, f, indent=2)
    print(f"Created train file: {train_file}")
    
    # Save test file
    test_file = test_dir / f"op_func_test_{args.difficulty}.json"
    with open(test_file, 'w') as f:
        json.dump(test_d, f, indent=2)
    print(f"Created test file: {test_file}")

    print(f"Finished generating data.")


if __name__ == "__main__":
    main()
