
from prompts.markdown_loader import render_markdown_prompt

HEADER = "You will be shown 9 cards on the board and have to select 3 cards that form a GAME-SET based on the following rules.\n"

SET_GAME_RULES_BASE = [
    "1. Each card has two attributes: an animal and a biome.\n",
    "2. There are 10 types of animals: armadillo, axolotl, bat, cat, fox, horse, rabbit, sheep, turtle, wolf.\n",
    "3. There are 10 types of biomes: plain, savanna, desert, swamp, forest, jungle, taiga, hill, badland, tundra.\n",
    "4. A GAME-SET is a set of three cards: For each attribute (animal, biome), the three cards must be either ALL the SAME or ALL DIFFERENT. e.g. if 2 of the cards have the same value, and 1 of them has a different value, the set is NOT valid.\n",
]

SET_GAME_RULES_LV2 = [
    "1. Each card has three attributes: an animal, a biome, and a food.\n",
    "2. There are 10 types of animals: armadillo, axolotl, bat, cat, fox, horse, rabbit, sheep, turtle, wolf.\n",
    "3. There are 10 types of biomes: plain, savanna, desert, swamp, forest, jungle, taiga, hill, badland, tundra.\n",
    "4. There are 10 types of food: apple, potato, bread, cake, carrot, mutton, beef, cookie, pie, melon.\n",
    "5. A GAME-SET is a set of three cards: For each attribute (animal, biome, food), the three cards must be either ALL the SAME or ALL DIFFERENT. e.g. if 2 of the cards have the same value, and 1 of them has a different value, the set is NOT valid.\n"
]

SET_GAME_RULES_LV3 = [
    "1. Each card has a number and three attributes: an animal, a biome, and a food.\n",
    "2. Numbers range from 1 to 12.\n",
    "3. There are 10 types of animals: armadillo, axolotl, bat, cat, fox, horse, rabbit, sheep, turtle, wolf.\n",
    "4. There are 10 types of biomes: plain, savanna, desert, swamp, forest, jungle, taiga, hill, badland, tundra.\n",
    "5. There are 10 types of food: apple, potato, bread, cake, carrot, mutton, beef, cookie, pie, melon.\n",
    "6. A GAME-SET is a set of three cards: For each attribute (animal, biome, food), the three cards must be either ALL the SAME or ALL DIFFERENT. e.g. if 2 of the cards have the same value, and 1 of them has a different value, the set is NOT valid.",
    "7. But only for the number, 2 of the cards should have the same number, and 1 of them should have a different number in order for the set to be valid.\n"
]

FORMAT = "Based on the rules, tell me which three cards here constitute a GAME-SET. Type the cards in the following format: \\boxed{First card: CARD1\nSecond card: CARD2\nThird card: CARD3}"

#FORMATTING
def format_card(card, difficulty):
    """Format card based on difficulty level"""
    if difficulty == 1:
        animal, biome = card
        return f"({animal}|{biome})"
    elif difficulty == 2:
        animal, biome, food = card
        return f"({animal}|{biome}|{food})"
    elif difficulty == 3:
        number, animal, biome, food = card
        return f"({number}|{animal}|{biome}|{food})"

def format_board(board, difficulty):
    """Format a 12-card board as text"""
    board_formatted = ""
    for i in range(len(board)):
        board_formatted += format_card(board[i], difficulty)
        if (i+1)%3 == 0:
            board_formatted += "\n"
        
    return board_formatted

#PROMPTS
def get_rule_based_prompt(difficulty, test_input):
    """
    difficulty: 1,2,3
    num_examples: 20
    mode: rule, example, combined
    """

    board = test_input
    return render_markdown_prompt(
        "set_game.md",
        ("Rules", f"Difficulty {difficulty}"),
        board=format_board(board, difficulty).rstrip(),
    )


def get_example_based_prompt(difficulty, test_input, examples):
    prompt = ""

    board = test_input
    for i in range(len(examples)):
        example_cards = examples[i]["valid_set"]
        example_board = examples[i]["board"]
        prompt += format_board(example_board, difficulty)

        c1, c2, c3 = example_cards
        prompt += "→\\boxed{"
        prompt += format_card(c1, difficulty)
        prompt += format_card(c2, difficulty)
        prompt += format_card(c3, difficulty)
        prompt += "}; "

    prompt += format_board(board, difficulty) +"→"
    return prompt

def get_combined_prompt(difficulty, test_input, examples):
    """
    difficulty: 1,2,3
    num_examples: 20
    mode: rule, example, combined
    """

    board = test_input
    prompt = HEADER +"\n"
    if difficulty == 1:
            prompt += "RULES:\n" + "".join(SET_GAME_RULES_BASE)
    elif difficulty == 2:
        prompt += "RULES:\n" + "".join(SET_GAME_RULES_LV2)
    elif difficulty == 3:
        prompt += "RULES:\n" + "".join(SET_GAME_RULES_LV3)

    prompt += "\nFor example,\n"
    for i in range(len(examples)):
        example_cards = examples[i]["valid_set"]
        example_board = examples[i]["board"]

        prompt += format_board(example_board, difficulty) +"→"
        c1, c2, c3 = example_cards
        prompt += "\\boxed{" +format_card(c1, difficulty)
        prompt += format_card(c2, difficulty)
        prompt += format_card(c3, difficulty)
        prompt += "}; "
        
    prompt += "\n" + "Here is the board:\n\n"
    prompt += format_board(board, difficulty) + "\n\n"
    prompt += FORMAT

    return prompt
