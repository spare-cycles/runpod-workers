"""The WER metric, which is the number every model and bias decision will be made on.

Worth testing precisely because it is easy to get subtly wrong in a way that still produces
plausible percentages — and a subtly wrong WER picks the wrong model without ever looking broken.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import edit_distance, normalise, wer


def test_identical_text_scores_zero():
    assert wer("bonjour ça va", "bonjour ça va") == 0.0


def test_punctuation_and_case_are_not_errors():
    assert wer("Bonjour, ça va ?", "bonjour ça va") == 0.0


def test_accents_are_kept_because_they_change_the_word():
    """`ou` and `où` are different words. Folding them makes a model that gets them wrong look right."""
    assert normalise("où") != normalise("ou")
    assert wer("où es-tu", "ou es-tu") > 0


def test_the_two_apostrophes_are_unified():
    """A model emitting U+2019 must not score an error against a reference typed with U+0027."""
    assert wer("l'audio est clair", "l’audio est clair") == 0.0


def test_one_substitution_in_four_words_is_twenty_five_percent():
    assert wer("un deux trois quatre", "un deux trois cinq") == 0.25


def test_deletions_and_insertions_both_count():
    assert wer("un deux trois", "un deux") == 1 / 3
    assert wer("un deux", "un deux trois") == 0.5


def test_an_empty_hypothesis_against_real_speech_is_total_loss():
    assert wer("un deux trois", "") == 1.0


def test_an_empty_reference_is_not_a_division_by_zero():
    assert wer("", "") == 0.0
    assert wer("", "quelque chose") == 1.0


def test_edit_distance_is_symmetric():
    a, b = ["un", "deux", "trois"], ["un", "trois", "quatre"]
    assert edit_distance(a, b) == edit_distance(b, a)
