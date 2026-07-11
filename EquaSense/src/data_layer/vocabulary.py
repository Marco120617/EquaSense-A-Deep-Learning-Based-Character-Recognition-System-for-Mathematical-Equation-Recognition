# Vocabulary mapping dictionary matching predicted token IDs to targets
TOKEN_VOCABULARY = {
    0: "<PAD>",
    1: "<START>",
    2: "<END>",
    3: "<UNK>",
    4: "=",
    5: "+",
    6: "-",
    7: "\\frac",
    8: "\\sqrt",
    9: "\\int",
    10: "^",
    11: "_",
    12: "\\alpha",
    13: "\\beta",
    14: "\\sum",
    15: "\\bar"
}

# Reverse lookup dictionary useful for training data tokenization pipelines
REVERSE_VOCABULARY = {v: k for k, v in TOKEN_VOCABULARY.items()}

def token_ids_to_latex(token_ids):
    """Converts a sequence of generated index IDs into a single clean LaTeX string."""
    latex_tokens = []
    for token_id in token_ids:
        token = TOKEN_VOCABULARY.get(token_id, "<UNK>")
        if token == "<END>":
            break
        if token not in ["<PAD>", "<START>", "<UNK>"]:
            latex_tokens.append(token)
    return " ".join(latex_tokens)