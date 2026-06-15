# Lexical Category Inference Prompt Templates

## Rules

### Shared

#### Difficulty 1
You will be shown a list of four words. Decide whether this list is correct based on the following rules.

RULES:
1. There are exactly four categories: {prompt_labels}.
2. Each category has the meaning shown in the category definitions below.
3. A correct list contains exactly four comma-separated items.
4. In a correct list, the first item is a member of Category 1, the second item is a member of Category 2, and so on.
5. If a category says "at least one", an item may satisfy any listed description. If it says "all", an item must satisfy every listed description.
6. Your answer must be exactly one word: Yes or No.

CATEGORY DEFINITIONS:
{label_block}

List: {candidate_block}
Answer:

### Either

#### Difficulty 2
You will be shown a list of four words. Decide whether this list is correct based on the following rules.

RULES:
1. There are exactly four categories: {prompt_labels}.
2. Each category has the meaning shown in the category definitions below.
3. A correct list contains exactly four comma-separated items.
4. In a correct list, the first item is a member of Category 1, the second item is a member of Category 2, and so on.
5. If a category lists multiple descriptions, an item is a member of that category if it satisfies at least one listed description.
6. Your answer must be exactly one word: Yes or No.

CATEGORY DEFINITIONS:
{label_block}

List: {candidate_block}
Answer:

#### Difficulty 3
You will be shown a list of four words. Decide whether this list is correct based on the following rules.

RULES:
1. There are exactly four categories: {prompt_labels}.
2. Each category has the meaning shown in the category definitions below.
3. A correct list contains exactly four comma-separated items.
4. In a correct list, the first item is a member of Category 1, the second item is a member of Category 2, and so on.
5. If a category lists multiple descriptions, an item is a member of that category if it satisfies at least one listed description.
6. Your answer must be exactly one word: Yes or No.

CATEGORY DEFINITIONS:
{label_block}

List: {candidate_block}
Answer:

### Both

#### Difficulty 2
You will be shown a list of four words. Decide whether this list is correct based on the following rules.

RULES:
1. There are exactly four categories: {prompt_labels}.
2. Each category has the meaning shown in the category definitions below.
3. A correct list contains exactly four comma-separated items.
4. In a correct list, the first item is a member of Category 1, the second item is a member of Category 2, and so on.
5. If a category lists multiple descriptions, an item is a member of that category only if it satisfies all listed descriptions.
6. Your answer must be exactly one word: Yes or No.

CATEGORY DEFINITIONS:
{label_block}

List: {candidate_block}
Answer:

#### Difficulty 3
You will be shown a list of four words. Decide whether this list is correct based on the following rules.

RULES:
1. There are exactly four categories: {prompt_labels}.
2. Each category has the meaning shown in the category definitions below.
3. A correct list contains exactly four comma-separated items.
4. In a correct list, the first item is a member of Category 1, the second item is a member of Category 2, and so on.
5. If a category lists multiple descriptions, an item is a member of that category only if it satisfies all listed descriptions.
6. Your answer must be exactly one word: Yes or No.

CATEGORY DEFINITIONS:
{label_block}

List: {candidate_block}
Answer:
