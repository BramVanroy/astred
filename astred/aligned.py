from functools import cached_property
from itertools import combinations
from typing import Dict, Generator, List, Optional, Union

import stanza

from . import sentence
from .utils import Alignments, aligns_from_str


class AlignedSentences:
    def __init__(self, src, tgt, alignments: Union[Alignments, str]):
        if isinstance(alignments, str):
            alignments = aligns_from_str(alignments)

        self.src = src
        self.tgt = tgt
        self.alignments = Alignments.from_text_and_aligns(self.src, self.tgt, alignments)

        self.align_words()
        self.set_word_cross()

    @cached_property
    def word_cross(self) -> int:
        src_word_cross = self.src.word_cross
        tgt_word_cross = self.tgt.word_cross

        assert src_word_cross == tgt_word_cross

        return src_word_cross

    def align_words(self):
        # Add properties to words so they know who they are aligned with
        self.src.add_aligned_with(self.tgt, self.alignments)
        self.tgt.add_aligned_with(self.src, self.alignments)

    def _word_align_to_groups(self):
        """ Get all possible combinations of src_idxs and tgt_idxs (min_size=2), and find
            groups between these src/tgt combos so that the following requirements are met for
            that group:
            1. no item in src or tgt can be aligned to anything outside the group
            2. no alignments within the group can cross each other

            After finding all possible combinations, it is likely that not all items have been
            grouped. In that case, check which alignments are not grouped yet, and add them to
            their individual group.

            Variable names:
            - (src|tgt)2aligns_d: {idx: list of alignments (as tuples) with this idx}
            - (src2tgt|tgt2src)list_d: {src_idx|tgt_idx: list of tgts or srcs (resp.) connected
            to this item}
            - (src|tgt)_combs: all possible combinations of src or tgt idxs (with a min_size=2)
             and excluding combos
                               that are aligned to -1 (= null alignments; they break groups)
            - (src|tgt)_idxs: indices of src or tgt, not including null alignments
            - (src|tgt)_idxs_grouped: sets to keep track which idxs have already been put into
             a group
        """
        src_combs = self._consec_combinations(
            list(self.src2tgtlist_d.keys()), self.src2tgtlist_d
        )

        # need to listify the generator because it is in the internal loop
        # the generator would exhaust after the first outer loop, but we need to re-use
        # it for all outer loops, so build a list
        tgt_combs = list(
            self._consec_combinations(list(self.tgt2srclist_d.keys()), self.tgt2srclist_d)
        )

        src_idxs_grouped = set()
        tgt_idxs_grouped = set()
        groups = []

        # Try grouping a src_comb with a tgt_comb
        for src_comb in src_combs:
            # If any item in this combination has already been grouped in another
            # combo group, continue
            if any(src in src_idxs_grouped for src in src_comb):
                continue
            n_src_items = len(src_comb)
            # only calculate internal_cross once for src_comb
            has_internal_cross = self._has_internal_cross(src_comb)

            for tgt_comb in tgt_combs:
                # If any item in this combination has already been grouped in another
                # combo group, continue
                if any(tgt in tgt_idxs_grouped for tgt in tgt_comb):
                    continue

                n_tgt_items = len(tgt_comb)
                is_mwe = False
                if n_src_items > 1 and n_tgt_items > 1:
                    is_mwe = self.group_mwe and self._is_in_mwe(src_comb, tgt_comb)

                # If the src_combo+tgt_combo is a MWE or (it has no external aligns and
                # no internal crosses): go on
                if is_mwe or (
                    not self._has_external_aligns(src_comb, tgt_comb) and not has_internal_cross
                ):
                    # Keep track of src+tgt idxs that are already grouped
                    src_idxs_grouped.update(src_comb)
                    tgt_idxs_grouped.update(tgt_comb)
                    # Get all alignments of this group and add them as group
                    alignments_of_group = sorted(
                        [i for src in src_comb for i in self.src2aligns_d[src]]
                    )

                    groups.append(alignments_of_group)
                    # Break because we have found a suitable group
                    break

        groups = self._add_unsolved_idxs(src_idxs_grouped, tgt_idxs_grouped, groups)

        return sorted(groups)

    @staticmethod
    def _consec_combinations(
        idxs: List[int], dir2dirlist_d: Optional[Dict] = None, min_size: int = 1
    ) -> Generator:
        """ Get all consequtive combinations of idxs of all possible lengths.
            When getting consecutive combinations in cross, we want to split on -1 (null),
            but when getting consec groups in SAC, we already have groups without -1, so
            no need to check (dir2dirlist_d will be None).
            Returns largest possible groups first and decreases in length """
        idxs.sort()
        n_idxs = len(idxs)
        for i in range(n_idxs, min_size - 1, -1):
            for j in range(n_idxs - i + 1):
                s = set(idxs[j : j + i])
                # Do not make combinations with -1 (null), because -1 should always break groups
                if dir2dirlist_d is not None and (
                    -1 in s or any(-1 in dir2dirlist_d[i] for i in s)
                ):
                    continue
                yield s

    # #################
    # SETTING METRICS #
    ###################
    def set_word_cross(self):
        for pairs in combinations(self.alignments, 2):
            # Do not align to "unaligned" indices
            if any(pair.src == -1 or pair.tgt == -1 for pair in pairs):
                continue

            pair1, pair2 = pairs

            # If: has cross
            if pair2.tgt < pair1.tgt:
                print("HAS CROSS")
                # Find the relative movement that each src_token has to do
                # We attribute the cross value to the word that moves furthest
                # If the distance is equal, the first pair gets the cross value
                pair1_rel_mvmt = abs(pair1.src - pair1.tgt)
                pair2_rel_mvmt = abs(pair2.src - pair2.tgt)

                if pair2_rel_mvmt > pair1_rel_mvmt:
                    self.src[pair2.src].add_word_cross(pair2.tgt)
                    self.tgt[pair2.tgt].add_word_cross(pair2.src)
                else:
                    self.src[pair1.src].add_word_cross(pair1.tgt)
                    self.tgt[pair1.tgt].add_word_cross(pair1.src)

    @classmethod
    def from_stanza(
        cls,
        src_sentence: stanza.models.common.doc.Sentence,
        tgt_sentence: stanza.models.common.doc.Sentence,
        alignments: Union[Alignments, str],
    ):
        src = sentence.Sentence.from_stanza(src_sentence, side="src")
        tgt = sentence.Sentence.from_stanza(tgt_sentence, side="tgt")

        return cls(src, tgt, alignments)

    @classmethod
    def from_text(
        cls,
        src_sentence: str,
        tgt_sentence: str,
        alignments: Union[Alignments, str],
        src_lang: str = "en",
        tgt_lang: str = "nl",
    ):
        src = sentence.Sentence.from_text(src_sentence, lang=src_lang, side="src")
        tgt = sentence.Sentence.from_text(tgt_sentence, lang=tgt_lang, side="src")

        return cls(src, tgt, alignments)
