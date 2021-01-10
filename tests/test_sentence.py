from astred import Null


def test_sentence__len(sent1_4_words):
    assert len(sent1_4_words) == 4


def test_sentence__text(sent1_4_words):
    assert sent1_4_words.text == "A B C D"


def test_sentence__is_attached(sent1_4_words):
    for word in sent1_4_words:
        assert word.doc == sent1_4_words


def test_sentence__no_dummy(sent1_4_words):
    # Null tokens are only added to Sentences in AlignedSentences, not in regular Sentences
    assert all(not isinstance(w, Null) for w in sent1_4_words)
