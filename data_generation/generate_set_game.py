import random
from itertools import combinations
import argparse
from pathlib import Path
import json
from collections import defaultdict

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

ANIMALS = ["armadillo","axolotl","bat","cat","fox","horse","rabbit","sheep","turtle","wolf"]
BIOME = ["plain","savanna","desert","swamp","forest","jungle","taiga","hill","badland","tundra"]
FOOD = ["apple","potato","bread","cake","carrot","mutton","beef","cookie","pie","melon"]
NUMBERS = list(range(1,13))


def all_same(vals):
        return len(set(vals)) == 1
def all_diff(vals):
    return len(set(vals)) == 3

def is_set_base(cards):
    animal, biome = zip(*cards)

    #all animal same, biome different
    if all_same(animal) and all_diff(biome):
        return True, "animal_same_biome_diff"
    #all biome same, all animal different
    if all_same(biome) and all_diff(animal):
        return True, "animal_diff_biome_same"
    #all animal different, all biome different
    if all_diff(animal) and all_diff(biome):
        return True, "all_diff"
        
    return False, "Not a set"

def is_set_lv2(cards):
    animal, biome, food = zip(*cards)

    #all animal, biome same, food different
    if all_same(animal) and all_same(biome) and all_diff(food):
        return True, "other_same_food_diff"
    #all animal, food same, biome different
    if all_same(animal) and all_diff(biome) and all_same(food):
        return True, "other_same_biome_diff"
    #all biome, food same, animal different
    if all_diff(animal) and all_same(biome) and all_same(food):
        return True, "other_same_animal_diff"
    #all animal same, all biome, food different
    if all_same(animal) and all_diff(biome) and all_diff(food):
        return True, "other_diff_animal_same"
    #all biome same, all animal, food different
    if all_same(biome) and all_diff(animal) and all_diff(food):
        return True, "other_diff_biome_same"
    #all food same, all animal,biome different
    if all_same(food) and all_diff(animal) and all_diff(biome):
        return True, "other_diff_food_same"
    #all different for all attribute
    if all_diff(animal) and all_diff(biome) and all_diff(food):
        return True, "all_diff"

    return False, "Not a set"

def is_set_lv3(cards):
    number, animal, biome, food = zip(*cards)
    
    if len(set(number)) != 2:
        return False, "Not a set"
    
    lv2_cards = list(zip(animal, biome, food))

    return is_set_lv2(lv2_cards)

def make_board(difficulty):
        if difficulty == 1:
            return {
                "all_diff": [],
                "animal_same_biome_diff": [],
                "animal_diff_biome_same": [],
            }
        elif difficulty == 2:
            return {
            "all_diff": [],
            "other_same_food_diff": [],
            "other_same_biome_diff": [],
            "other_same_animal_diff": [],
            "other_diff_animal_same": [],
            "other_diff_biome_same": [],
            "other_diff_food_same": [],
        }
        elif difficulty == 3:
            return {
                "all_diff": [],
                "other_same_food_diff": [],
                "other_same_biome_diff": [],
                "other_same_animal_diff": [],
                "other_diff_animal_same": [],
                "other_diff_biome_same": [],
                "other_diff_food_same": [],
            }

def example_generator(difficulty, deck, total_needed):
    unique_cards_sets = set()
    unique_board_sets = set()
    board_size = 9
    meta_keys = {"valid_set", "p_set", "board"}

    is_set_fn = {1: is_set_base, 2: is_set_lv2, 3: is_set_lv3}[difficulty]

    # Determine category keys and per-category cap
    template = make_board(difficulty)
    category_keys = [k for k in template if k not in meta_keys]
    num_categories = len(category_keys)
    per_category_cap = total_needed // num_categories

    category_counts = {k: 0 for k in category_keys}

    examples = []
    attempts = 0
    max_attempts = total_needed * 200

    while len(examples) < total_needed and attempts < max_attempts:
        attempts += 1
        b = tuple(sorted(random.sample(deck, board_size)))

        if b in unique_board_sets:
            continue
        unique_board_sets.add(b)

        board_list = list(b)
        board = make_board(difficulty)
        board["valid_set"] = None
        board["board"] = board_list

        valid_sets = []
        all_triplets = list(combinations(board_list, 3))
        total = len(all_triplets)

        for cards in all_triplets:
            result, exp = is_set_fn(cards)
            if result:
                valid_sets.append(cards)
                board[exp].append(cards)

        if not valid_sets:
            continue

        board["p_set"] = len(valid_sets) / total
        if board["p_set"] >= 0.35:
            continue

        # Sort categories by current global count ascending (rarest first),
        # then by number of candidates on this board ascending (shortest first)
        category_lists = sorted(
            [(k, v) for k, v in board.items()
             if k not in meta_keys and isinstance(v, list) and len(v) > 0
             and category_counts[k] < per_category_cap],  # skip saturated categories
            key=lambda x: (category_counts[x[0]], len(x[1]))
        )

        if not category_lists:
            continue

        found = False
        for cat_key, candidates in category_lists:
            for cards_selected in candidates:
                key = tuple(sorted(cards_selected))
                if key not in unique_cards_sets:
                    unique_cards_sets.add(key)
                    board["valid_set"] = list(cards_selected)
                    board["valid_set_category"] = cat_key
                    category_counts[cat_key] += 1
                    found = True
                    break
            if found:
                break

        if not found:
            continue

        examples.append(board)

    print(f"Category counts: {category_counts}")
    print(f"Per-category cap: {per_category_cap}, Total examples: {len(examples)}")
    return examples

def main():
    parser = argparse.ArgumentParser(description='Generate set game examples')
    parser.add_argument('--train_samples', type=int, default=3000, #3000, 7000
                      help='Number of training samples to generate')
    parser.add_argument('--test_samples', type=int, default=501, #501, 504
                      help='Number of test samples to generate')
    parser.add_argument('--difficulty', type=int, default=1,
                      help='Difficulty of task')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument(
        '--output_root',
        type=str,
        default=str(WORKSPACE_ROOT / "data" / "set_game"),
        help='Output root directory. Defaults to <repo>/data/set_game.',
    )
    args = parser.parse_args()

    random.seed(args.seed)

    total_needed = args.train_samples + args.test_samples
    train_ratio = args.train_samples / (args.train_samples + args.test_samples)

    #generate total_needed sets of examples
    if args.difficulty == 1:
        full_deck = [(a, b) for a in ANIMALS for b in BIOME]
    elif args.difficulty == 2:
        full_deck = [(a, b, f) for a in ANIMALS for b in BIOME for f in FOOD]
    else:
        full_deck = [(n, a, b, f) for n in NUMBERS for a in ANIMALS for b in BIOME for f in FOOD]
    
    examples = example_generator(args.difficulty, full_deck, total_needed)
    print("Example generated: ", len(examples))

    # Generate train examples with only train cards
    category_buckets = defaultdict(list)
    for example in examples:
        category_buckets[example["valid_set_category"]].append(example)

    train_data, test_data = [], []
    for cat, cat_examples in category_buckets.items():
        random.shuffle(cat_examples)
        n_train = round(len(cat_examples) * train_ratio)
        train_data += cat_examples[:n_train]
        test_data += cat_examples[n_train:]

    # Shuffle final splits to avoid category clustering
    random.shuffle(train_data)
    random.shuffle(test_data)
    #train_data = train_data[:args.train_samples]
    #test_data = test_data[:args.test_samples]


    # formatting train data file
    train_d = {
            'task': 'set game',
            'difficulty': args.difficulty,
            'dataset_size': len(train_data),
            'data': train_data
        }

    # formatting test data file
    test_d = {
            'task': 'set game',
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
    train_file = train_dir / f"set_game_train_{args.difficulty}.json"
    with open(train_file, 'w') as f:
        json.dump(train_d, f, indent=2)
    print(f"Created train file: {train_file}")
    
    # Save test file
    test_file = test_dir / f"set_game_test_{args.difficulty}.json"
    with open(test_file, 'w') as f:
        json.dump(test_d, f, indent=2)
    print(f"Created test file: {test_file}")

    print(f"Finished generating data.")

    #plot_category_distribution(examples, args.difficulty)
    p_sets = [e["p_set"] for e in examples]
    print(f"p_set stats: min={min(p_sets):.4f}, max={max(p_sets):.4f}, mean={sum(p_sets)/len(p_sets):.4f}")


def plot_category_distribution(examples, difficulty):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    meta_keys = {"valid_set", "p_set", "board"}

    # Collect category keys and counts
    category_counts = {}
    for example in examples:
        for k, v in example.items():
            if k not in meta_keys and isinstance(v, list):
                category_counts[k] = category_counts.get(k, 0) + len(v)

    # Also count which category valid_set was drawn from
    valid_set_source_counts = {}
    for example in examples:
        vs = example.get("valid_set")
        if vs is None:
            continue
        vs_key = tuple(sorted(vs))
        for k, v in example.items():
            if k not in meta_keys and isinstance(v, list):
                for cards in v:
                    if tuple(sorted(cards)) == vs_key:
                        valid_set_source_counts[k] = valid_set_source_counts.get(k, 0) + 1
                        break

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # 1. p_set distribution (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    p_sets = [e["p_set"] for e in examples]
    ax1.hist(p_sets, bins=30, edgecolor="black", color="steelblue")
    ax1.set_xlabel("p_set")
    ax1.set_ylabel("Number of boards")
    ax1.set_title(f"p_set Distribution (difficulty={difficulty})")
    ax1.axvline(sum(p_sets)/len(p_sets), color="red", linestyle="--", label=f"mean={sum(p_sets)/len(p_sets):.3f}")
    ax1.legend()

    # 2. Total valid sets per category across all boards (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    cats = list(category_counts.keys())
    counts = [category_counts[c] for c in cats]
    bars = ax2.bar(cats, counts, edgecolor="black", color="coral")
    ax2.set_xlabel("Category")
    ax2.set_ylabel("Total valid set instances")
    ax2.set_title("Valid Set Instances per Category (all boards)")
    ax2.tick_params(axis="x", rotation=45)
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha="center", va="bottom", fontsize=8)

    # 3. Which category valid_set was drawn from (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    src_cats = list(valid_set_source_counts.keys())
    src_counts = [valid_set_source_counts[c] for c in src_cats]
    bars3 = ax3.bar(src_cats, src_counts, edgecolor="black", color="mediumseagreen")
    ax3.set_xlabel("Category")
    ax3.set_ylabel("Times selected as valid_set")
    ax3.set_title("valid_set Source Category Distribution")
    ax3.tick_params(axis="x", rotation=45)
    for bar, count in zip(bars3, src_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha="center", va="bottom", fontsize=8)

    # 4. Avg sets per category per board (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    n = len(examples)
    avg_counts = [category_counts.get(c, 0) / n for c in cats]
    ax4.bar(cats, avg_counts, edgecolor="black", color="mediumpurple")
    ax4.set_xlabel("Category")
    ax4.set_ylabel("Avg valid set instances per board")
    ax4.set_title("Avg Sets per Category per Board")
    ax4.tick_params(axis="x", rotation=45)

    plt.suptitle(f"Set Game Data Analysis — Difficulty {difficulty}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"analysis_difficulty_{difficulty}.png", dpi=150)
    plt.show()
    print(f"Saved analysis plot to analysis_difficulty_{difficulty}.png")


if __name__ == "__main__":
    main()
