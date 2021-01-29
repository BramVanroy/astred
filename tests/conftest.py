from pytest import fixture

from astred import AlignedSentences, Sentence, Word


@fixture
def sent1_4_words():
    return Sentence(
        [Word(id=i, text=c) for i, c in enumerate("ABCD", 1)]
    )


@fixture
def sent2_4_words():
    return Sentence(
        [Word(id=i, text=c) for i, c in enumerate("ABCD", 1)]
    )


@fixture
def sent3_6_words():
    return Sentence(
        [Word(id=i, text=c) for i, c in enumerate("ABCDEF", 1)]
    )


@fixture
def sent4_6_words():
    return Sentence(
        [Word(id=i, text=c) for i, c in enumerate("ABCDEF", 1)]
    )


class TestAlignedSents:
    def case_no_cross(self, sent1_4_words, sent2_4_words):
        return AlignedSentences(sent1_4_words, sent2_4_words, "0-0 1-1 2-2 3-3", allow_mwe=False)

    def case_one_cross(self, sent1_4_words, sent2_4_words):
        return AlignedSentences(sent1_4_words, sent2_4_words, "0-0 1-2 2-1 3-3", allow_mwe=False)

    def case_long_distance_cross(self, sent1_4_words, sent2_4_words):
        return AlignedSentences(sent1_4_words, sent2_4_words, "0-3 1-1 2-2 3-0", allow_mwe=False)

    def case_three_words_seq(self, sent1_4_words, sent2_4_words):
        return AlignedSentences(sent1_4_words, sent2_4_words, "0-1 1-2 2-3 3-0", allow_mwe=False)

    def case_two_nulls(self, sent1_4_words, sent2_4_words):
        return AlignedSentences(sent1_4_words, sent2_4_words, "0-1 1-2 2-3", allow_mwe=False)

    def case_two_mwes_allowed(self, sent3_6_words, sent4_6_words):
        return AlignedSentences(
            sent3_6_words,
            sent4_6_words,
            "0-0 0-1 0-2 1-0 1-1 1-2 2-0 2-1 2-2 3-3 3-4 3-5 4-3 4-4 4-5 5-3 5-4 5-5",
            allow_mwe=True
        )

    def case_two_mwes_disallowed(self, sent3_6_words, sent4_6_words):
        return AlignedSentences(
            sent3_6_words,
            sent4_6_words,
            "0-0 0-1 0-2 1-0 1-1 1-2 2-0 2-1 2-2 3-3 3-4 3-5 4-3 4-4 4-5 5-3 5-4 5-5",
            allow_mwe=False
        )

    def case_mwe_and_cross(self, sent3_6_words, sent4_6_words):
        return AlignedSentences(
            sent3_6_words,
            sent4_6_words,
            "0-5 1-1 1-2 1-3 1-4 2-1 2-2 2-3 2-4 3-1 3-2 3-3 3-4 4-1 4-2 4-3 4-4 5-0",
            allow_mwe=True,
        )

    def case_m_to_n(self, sent3_6_words, sent4_6_words):
        return AlignedSentences(
            sent3_6_words, sent4_6_words, "0-0 1-1 1-2 2-1 3-3 3-4 3-5 4-3",
            allow_mwe=False
        )
