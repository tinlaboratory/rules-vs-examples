# Set Game Prompt Templates

## Rules

### Difficulty 1
You will be shown 9 cards on the board and have to select 3 cards that form a GAME-SET based on the following rules.

RULES:
1. Each card has two attributes: an animal and a biome.
2. There are 10 types of animals: armadillo, axolotl, bat, cat, fox, horse, rabbit, sheep, turtle, wolf.
3. There are 10 types of biomes: plain, savanna, desert, swamp, forest, jungle, taiga, hill, badland, tundra.
4. A GAME-SET is a set of three cards: For each attribute (animal, biome), the three cards must be either ALL the SAME or ALL DIFFERENT. e.g. if 2 of the cards have the same value, and 1 of them has a different value, the set is NOT valid.

Here is the board:

{board}

Based on the rules, tell me which three cards here constitute a GAME-SET. There may be multiple valid sets, but you must pick exactly one valid set and output the cards in the following format:
 \boxed{{First card: CARD1
Second card: CARD2
Third card: CARD3}}

### Difficulty 2
You will be shown 9 cards on the board and have to select 3 cards that form a GAME-SET based on the following rules.

RULES:
1. Each card has three attributes: an animal, a biome, and a food.
2. There are 10 types of animals: armadillo, axolotl, bat, cat, fox, horse, rabbit, sheep, turtle, wolf.
3. There are 10 types of biomes: plain, savanna, desert, swamp, forest, jungle, taiga, hill, badland, tundra.
4. There are 10 types of food: apple, potato, bread, cake, carrot, mutton, beef, cookie, pie, melon.
5. A GAME-SET is a set of three cards: For each attribute (animal, biome, food), the three cards must be either ALL the SAME or ALL DIFFERENT. e.g. if 2 of the cards have the same value, and 1 of them has a different value, the set is NOT valid.

Here is the board:

{board}

Based on the rules, tell me which three cards here constitute a GAME-SET. There may be multiple valid sets, but you must pick exactly one valid set and output the cards in the following format:
 \boxed{{First card: CARD1
Second card: CARD2
Third card: CARD3}}

### Difficulty 3
You will be shown 9 cards on the board and have to select 3 cards that form a GAME-SET based on the following rules.

RULES:
1. Each card has a number and three attributes: an animal, a biome, and a food.
2. Numbers range from 1 to 12.
3. There are 10 types of animals: armadillo, axolotl, bat, cat, fox, horse, rabbit, sheep, turtle, wolf.
4. There are 10 types of biomes: plain, savanna, desert, swamp, forest, jungle, taiga, hill, badland, tundra.
5. There are 10 types of food: apple, potato, bread, cake, carrot, mutton, beef, cookie, pie, melon.
6. A GAME-SET is a set of three cards: For each attribute (animal, biome, food), the three cards must be either ALL the SAME or ALL DIFFERENT. e.g. if 2 of the cards have the same value, and 1 of them has a different value, the set is NOT valid.
7. But only for the number, 2 of the cards should have the same number, and 1 of them should have a different number in order for the set to be valid.

Here is the board:

{board}

Based on the rules, tell me which three cards here constitute a GAME-SET. There may be multiple valid sets, but you must pick exactly one valid set and output the cards in the following format:
 \boxed{{First card: CARD1
Second card: CARD2
Third card: CARD3}}
