# Noun Class Agreement Prompt Templates

## Rules

### Difficulty 1
You will be given one sentence in a synthetic language. Decide whether the sentence is grammatical.

RULES:
1. There are 2 noun classes: A and B.
2. Each noun phrase has the form DET NOUN.
3. The class A determiner is ka, and the class B determiner is ti.
4. The class A nouns are tac, sular, fep, wug. The class B nouns are bim, noko, glarn, zesh.
5. Sentences follow the template DET NOUN VERB DET NOUN.
6. Verbs are plain roots like dax, miv, lorp and never change.
7. A sentence is grammatical if and only if, in each noun phrase, the determiner matches the noun's class.
8. Your answer must be exactly one word: Yes or No.

Sentence: {sentence}
Answer:

### Difficulty 2
You will be given one sentence in a synthetic language. Decide whether the sentence is grammatical.

RULES:
1. There are 4 noun classes: A, B, C, D.
2. A noun phrase has the form DET ADJ-SUFF NOUN.
3. The class A determiner is ka, the class B determiner is ti, the class C determiner is su, and the class D determiner is vo.
4. Adjectives have the form STEM-SUFFIX. The suffix must match the noun's class.
5. The class A adjective suffix is -en, the class B adjective suffix is -os, the class C adjective suffix is -im, and the class D adjective suffix is -at.
6. Example adjective tokens are glim-en, prun-os, fen-im, zay-at.
7. The class A nouns are tav, fap. The class B nouns are bim, glarn. The class C nouns are sular, wug. The class D nouns are noko, zesh.
8. Sentences follow the template DET ADJ-SUFF NOUN VERB DET ADJ-SUFF NOUN.
9. Verbs are plain roots like dax, miv, lorp and never change.
10. A sentence is grammatical if and only if, in each noun phrase, both the determiner and the adjective suffix match the noun's class.
11. Your answer must be exactly one word: Yes or No.

Sentence: {sentence}
Answer:

### Difficulty 3
You will be given one sentence in a synthetic language. Decide whether the sentence is grammatical.

RULES:
1. There are 6 noun classes: A, B, C, D, E, F.
2. A noun phrase has the form DET ADJ-SUFF NOUN.
3. The class A determiner is ka, the class B determiner is ti, the class C determiner is su, the class D determiner is vo, the class E determiner is ne, and the class F determiner is la.
4. Adjectives have the form STEM-SUFFIX. The suffix must match the noun's class.
5. The class A adjective suffix is -en, the class B adjective suffix is -os, the class C adjective suffix is -im, the class D adjective suffix is -at, the class E adjective suffix is -uk, and the class F adjective suffix is -esh.
6. The verb has the form PREFIX-ROOT-SUFFIX.
7. The prefix must match the class of the subject noun, and the suffix must match the class of the object noun.
8. The class A subject prefix is ge-, the class B subject prefix is du-, the class C subject prefix is ri-, the class D subject prefix is zo-, the class E subject prefix is pa-, and the class F subject prefix is li-.
9. The class A object suffix is -an, the class B object suffix is -eb, the class C object suffix is -ig, the class D object suffix is -ot, the class E object suffix is -ul, and the class F object suffix is -er.
10. Example verb tokens are ge-dax-eb, pa-miv-ig, du-lorp-er.
11. The class A nouns are tav, fap. The class B nouns are bim, glarn. The class C nouns are sular, wug. The class D nouns are noko, zesh. The class E nouns are tac, fep. The class F nouns are lurn, prax.
12. Sentences follow the template DET ADJ-SUFF NOUN PREFIX-ROOT-SUFFIX DET ADJ-SUFF NOUN.
13. Verb roots are plain roots like dax, miv, lorp. Only the verb's prefix and suffix change.
14. A sentence is grammatical if and only if the subject noun phrase's determiner and adjective suffix match the subject noun's class, the object noun phrase's determiner and adjective suffix match the object noun's class, the verb prefix matches the subject noun's class, and the verb suffix matches the object noun's class.
15. Your answer must be exactly one word: Yes or No.

Sentence: {sentence}
Answer:
