Q1_SENTENCES = [
    "I like cats and dogs",
    "She enjoys reading books and playing video games",
    "They play football every weekend"
]

Q1_VOCAB = {
    "I": 4,
    "She": 5,
    "They": 6,
    "like": 7,
    "enjoys": 8,
    "play": 9,
    "cats": 10,
    "dogs": 11,
    "reading": 12,
    "books": 13,
    "football": 14,
    "every": 15,
    "weekend": 16,
    "and": 17,
    "playing": 18,
    "video": 19,
    "games": 20
}

Q1_DECODE_TRANSITIONS = {
    (): [("ok", 0.4), ("yes", 0.5), ("</s>", 0.1)],
    ("ok",): [("ok", 0.7), ("yes", 0.2), ("</s>", 0.1)],
    ("yes",): [("ok", 0.3), ("yes", 0.4), ("</s>", 0.3)],
    ("ok", "ok"): [("</s>", 1.0)],
    ("ok", "yes"): [("</s>", 1.0)],
    ("yes", "ok"): [("</s>", 1.0)],
    ("yes", "yes"): [("</s>", 1.0)]
}

Q1_BLEU_CANDIDATE = "A small cat is sitting on the wooden table."

Q1_BLEU_REFERENCES = [
    "A little cat is sitting on the leather chair.",
    "A small cat is sitting on a wooden chair."
]
