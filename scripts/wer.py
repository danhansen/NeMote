"""Stable word-error scoring for retained native-runtime accuracy experiments."""
import string


def normalize_words(text):
    table = str.maketrans({character: " " for character in string.punctuation})
    return text.lower().replace("’", "'").translate(table).split()


def wer_stats(reference, hypothesis):
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    previous = list(range(len(hyp) + 1))
    for index, word in enumerate(ref, 1):
        current = [index]
        for column, predicted in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (word != predicted)))
        previous = current
    edits = previous[-1]
    return edits, len(ref), edits / max(len(ref), 1)
