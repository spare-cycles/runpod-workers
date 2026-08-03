"""The guard is what makes `chat` bias mode safe to try, so these are the tests that matter most.

The failure it exists for is not an error: it is a *fluent summary* written into a searchable
transcript column as if it were speech. Nothing downstream can tell the difference, which is why
the check has to be here and why both halves of it are asserted rather than assumed.
"""

from worker.guards import min_chars_for, reject_reason

# 12 s of French speech is roughly 150-180 characters. The floor at the default 3.0 chars/s is 36 —
# deliberately far below, because this catches summaries, not bad transcripts.
LONG = "Salut, je voulais te dire que je passe demain vers dix heures pour récupérer les clés, à plus."


def test_a_real_transcript_passes():
    assert reject_reason(LONG, 12.0, 3.0) is None


def test_a_summary_is_rejected_on_length():
    reason = reject_reason("Il passe demain.", 12.0, 3.0)
    assert reason is not None
    assert "summary, not a transcript" in reason


def test_meta_commentary_is_rejected_even_when_it_is_long_enough():
    """Length alone would let this through: it is longer than the floor and reads like prose."""
    answer = "Voici la transcription de l'enregistrement : il passe demain vers dix heures pour les clés."
    assert reject_reason(answer, 12.0, 3.0) is not None
    assert "meta-commentary" in (reject_reason(answer, 12.0, 3.0) or "")


def test_english_meta_commentary_is_rejected_too():
    answer = "Here is the transcription of the recording you provided, with the names spelled as requested."
    assert "meta-commentary" in (reject_reason(answer, 12.0, 3.0) or "")


def test_a_refusal_is_reported_as_meta_not_as_too_short():
    """Order matters: 'too short' would send the reader to the audio instead of to the prompt."""
    reason = reject_reason("Je suis désolé, je ne peux pas faire ça.", 30.0, 3.0)
    assert reason is not None
    assert "meta-commentary" in reason


def test_voici_inside_a_sentence_is_not_meta_commentary():
    """`voici` is ordinary French. Matching it anywhere would reject real speech."""
    answer = "Alors je t'explique, voici la situation : on part demain matin et on rentre dimanche soir."
    assert reject_reason(answer, 8.0, 3.0) is None


def test_an_empty_answer_is_rejected():
    assert reject_reason("   ", 5.0, 3.0) == "the model returned nothing"


def test_a_very_short_recording_keeps_the_absolute_floor():
    """A one-second "oui" is a legitimate voice note; no ratio can tell it from a refusal."""
    assert min_chars_for(1.0, 3.0) == 12
    assert reject_reason("Oui, d'accord.", 1.0, 3.0) is None


def test_the_floor_scales_with_duration():
    assert min_chars_for(60.0, 3.0) == 180
    assert min_chars_for(600.0, 3.0) == 1800
