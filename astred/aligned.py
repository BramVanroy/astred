from __future__ import annotations

import operator
from dataclasses import dataclass, field
from functools import cached_property
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple, Union

from .aligner import Aligner
from .enum import EditOperation, Side, SpanType
from .pairs import IdxPair
from .sentence import Sentence
from .span import NullSpan, Span, SpanPair
from .tree import AstredConfig, Tree
from .utils import pair_combs, rebase_to_idxs, unique_list
from .word import WordPair, spanpair_to_wordpairs


@dataclass(eq=False)
class AlignedSentences:
    """'AlignedSentences' is the main entry point for using this library. The focus lies on syntactic measures between
    a source and target sentence. 'AlignedSentences' takes as input at least a source and target :class:`Sentence`,
    and word alignments for that sentence pair.
    """

    src: Sentence
    tgt: Sentence
    word_aligns: Union[List[Union[IdxPair, Tuple[int, int]]], str] = field(default=None)
    aligner: Optional[Aligner] = field(default=None, repr=False)
    allow_mwe: bool = field(default=True)

    aligned_words: List[WordPair] = field(default_factory=list, init=False, repr=False)
    word_cross: int = field(default=0, init=False)

    aligned_seq_spans: List[SpanPair] = field(default_factory=list, init=False, repr=False)
    seq_aligns: List[IdxPair] = field(default_factory=list, init=False, repr=False)
    seq_cross: int = field(default=0, init=False)

    aligned_sacr_spans: List[SpanPair] = field(default_factory=list, init=False, repr=False)
    sacr_aligns: List[IdxPair] = field(default_factory=list, init=False, repr=False)
    sacr_cross: int = field(default=0, init=False)

    ted_config: AstredConfig = field(default=AstredConfig(), repr=False)
    ted: int = field(default=0, init=False)
    ted_ops: List[Tuple[Tree]] = field(default_factory=list, repr=False, init=False)

    def __getitem__(self, idx):
        return self.aligned_words[idx]

    def __iter__(self):
        return iter(self.aligned_words)

    def __len__(self):
        return len(self.aligned_words)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(src={self.src.text}, tgt={self.tgt.text},"
            f" aligns={[(i.src, i.tgt) for i in self.word_aligns]})"
        )

    def __post_init__(self):
        self.init_word_aligns()
        self.attach_self_to_sentences()
        # NULL is added to the front of the sentences here
        self.attach_sentences()

        self.aligned_words = [WordPair(self.src[align.src], self.tgt[align.tgt]) for align in self.word_aligns]
        self.attach_pairs(self.aligned_words)
        self.set_cross(self.aligned_words, "word_cross")

        # SEQUENCES
        self.create_seq_spans()
        self.attach_pairs(self.aligned_seq_spans)
        self.set_cross(self.aligned_seq_spans, "seq_cross")

        if self.src.tree and self.tgt.tree:
            # SACR
            self.create_sacr_spans()
            self.attach_pairs(self.aligned_sacr_spans)
            self.set_cross(self.aligned_sacr_spans, "sacr_cross")

            # TED
            self.set_connected()
            self.set_ted()

    @cached_property
    def giza_word_aligns(self):
        return " ".join([f'{p.src-1}-{p.tgt-1}' for p in self.word_aligns if p.src and p.tgt])

    @property
    def idxs_d(self) -> Dict[str, Set[int]]:
        """Extracts the unique source and target word indices from the word alignments.
        :return: a dictionary containg "src" and "tgt" keys with a set of integer values as keys
        """
        src, tgt = zip(*self.word_aligns)
        return {"src": src, "tgt": tgt}

    @property
    def no_null_sacr_pairs(self):
        """Removes any NULL alignments (-1 to exclude MWE from comparison. Is included in list, though)
        :return:
        """
        return [pair for pair in self.aligned_sacr_spans if not any(p.is_null for p in pair[:-1])]

    @property
    def no_null_seq_pairs(self):
        """Removes any NULL alignments (-1 to exclude MWE from comparison. Is included in list, though)
        :return:
        """
        return [pair for pair in self.aligned_seq_spans if not any(p.is_null for p in pair[:-1])]

    @property
    def no_null_word_pairs(self):
        """Removes any NULL alignments
        :return:
        """
        return [pair for pair in self.aligned_words if not any(p.is_null for p in pair)]

    def num_changes(self, attr="deprel"):
        num_changes = self.src.num_changes(attr)
        assert num_changes == self.tgt.num_changes(attr)
        return num_changes

    @staticmethod
    def attach_pairs(pairs: List[Union[SpanPair, WordPair]]):
        """Attach the "src" and "tgt" items in a list of pairs to each other, effectively adding them to
         their "aligned" attribute. This can be done both for aligned :class:`WordPair` and :class:`SpanPair`.
        :param pairs: a list of :class:`WordPair`s or :class:`SpanPair`s
        """
        for pair in pairs:
            pair.src.add_aligned(pair.tgt)
            pair.tgt.add_aligned(pair.src)

    @staticmethod
    def check_mwe_and_external_align(pairs: List[WordPair], src_ids: Set[int], tgt_ids: Set[int]) -> Tuple[bool, bool]:
        """For a given list of :class:`WordPair`, and a set of its ``src_ids`` and ``tgt_ids``, check whether this
        group is a multi-word expression (MWE) and whether any of the involved words is aligned with words outside of
        this group. A multi-word expression here is defined as a group of more than one source and target words, and
        for which all words in the source group are aligned with all words in the target group, and vice-versa.
        :param pairs: a list of :class:`WordPair`
        :param src_ids: a set containing all the source indices (int) in ``pairs``
        :param tgt_ids:a set containing all the target indices (int) in ``pairs``
        :return: a tuple of booleans indicating: (i) whether this list of pairs is a MWE; (ii) whether any of
        the involved words is aligned to words that are not part of any of the involved :class:`WordPair`s.
        """
        n_src = len(unique_list([p.src for p in pairs]))
        n_tgt = len(unique_list([p.tgt for p in pairs]))

        # MWE must consist of more than one source and target word
        # Later we then check whether each word is aligned with all other words in the group
        is_mwe = n_src > 1 and n_tgt > 1

        has_external_align = False
        for wordpair in pairs:
            aligned_to_src = set([w.id for w in wordpair.src.aligned])
            aligned_to_tgt = set([w.id for w in wordpair.tgt.aligned])

            # Check whetther each source word is attached to all target words
            # If it is set to False once, don't try to change it.
            if is_mwe and (aligned_to_src != tgt_ids or aligned_to_tgt != src_ids):
                is_mwe = False

            # Check whether the aligned indices of all words are a subset of the actual idxs.
            # If it is not a subset (and it contains more idxs than the actual idxs), then that
            # means that that word is aligned with a word outside of this pair.
            if not aligned_to_src.issubset(tgt_ids) or not aligned_to_tgt.issubset(src_ids):
                has_external_align = True

            # Break because these proeprties cannot change anymore.
            if not is_mwe and has_external_align:
                break

        return is_mwe, has_external_align

    def init_word_aligns(self):
        if not self.word_aligns:
            if not self.aligner:
                self.aligner = Aligner()

            self.word_aligns = [IdxPair(*val) for val in self.aligner.align_from_objs(self.src, self.tgt)]
        elif isinstance(self.word_aligns, str):
            try:
                self.word_aligns = [IdxPair(*map(int, align.split("-"))) for align in self.word_aligns.split(" ")]
            except ValueError as exc:
                raise ValueError("The passed alignments could not be parsed successfully. Make sure that they are"
                                 " written in the correct format as pairs of src_idx-tgt_idx") from exc
        elif not isinstance(self.word_aligns, IdxPair):
            self.word_aligns = [IdxPair(*val) for val in self.word_aligns]

        # +1 because 0-index is reserved for NULL
        self.word_aligns = [IdxPair(p.src + 1, p.tgt + 1) for p in self.word_aligns]
        self.add_null_aligns()
        self.word_aligns.sort(key=operator.attrgetter("src", "tgt"))

        # Don't keep aligner here
        self.aligner = None

    @staticmethod
    def has_internal_cross(pairs: List):
        for pair1, pair2 in combinations(pairs, 2):
            if pair2.tgt.id < pair1.tgt.id:
                return True

        return False

    @staticmethod
    def idxs_are_consecutive(idxs: List[int]):
        return sorted(idxs) == list(range(min(idxs), max(idxs)+1))

    def add_null_aligns(self):
        # Fill in 0 idx for words that are not aligned
        # The second list comprehension will already take into account the added idxs of the first one
        # That ensures that the NULL words are not added twice.
        self.word_aligns += [IdxPair(idx, 0) for idx in range(len(self.src)+1) if idx not in self.idxs_d["src"]]
        self.word_aligns += [IdxPair(0, idx) for idx in range(len(self.tgt)+1) if idx not in self.idxs_d["tgt"]]

    def attach_sentences(self):
        # This setter adds NULL at the front of the sentence
        self.tgt.aligned_sentence = self.src
        self.src.side = Side.SRC
        self.src.aligned_sentence = self.tgt
        self.tgt.side = Side.TGT

    def attach_self_to_sentences(self):
        self.src.aligned_sentences = self
        self.tgt.aligned_sentences = self

    def is_valid_sequence(self, pairs, src_ids, tgt_ids):
        # Check if:
        # - src and tgt idxs are consecutive and the group has no external alignments
        # - if there are internal crosses, only allow this group if it's MWE and MWE is allowed
        # - if no internal cross at this stage, it is a valid group
        is_mwe, has_external_align = self.check_mwe_and_external_align(pairs, src_ids, tgt_ids)
        idxs_consec = self.idxs_are_consecutive(src_ids) and self.idxs_are_consecutive(tgt_ids)

        is_valid = False
        if idxs_consec and not has_external_align:
            # If there is an internal cross, this can only be a valid group if it is a MWE
            if self.has_internal_cross(pairs):
                is_valid = self.allow_mwe and is_mwe
            else:
                # When we got this far, it must be a valid group:
                # - src and tgt ids are consecutive
                # - there are no external alignments
                # - there are no internal crosses
                is_valid = True

        return is_valid, is_mwe

    def create_sacr_spans(self):
        def is_valid_sacr_pair(pair):
            _is_valid = pair.src.is_valid_subtree and pair.tgt.is_valid_subtree or (self.allow_mwe and spanpair.is_mwe)
            _is_valid = _is_valid or (pair.src.is_null and pair.tgt.is_null)
            return _is_valid

        src_word_groups = []
        tgt_word_groups = []
        sacr_spans: List[Tuple[int, int, bool]] = []
        found: Dict[str, Set[int]] = {"src": set(), "tgt": set()}

        def add_found(spair, s_ids, t_ids):
            found["src"].update(s_ids)
            found["tgt"].update(t_ids)
            s_words, t_words = map(list, spair[:-1])  # Exclude mwe
            src_word_groups.append(s_words)
            tgt_word_groups.append(t_words)
            sacr_spans.append((min(s_ids), min(t_ids), spair.is_mwe))

        # This should probably be written more DRY-y
        for spanpair in self.aligned_seq_spans:
            src_ids = set([w.id for w in spanpair.src])
            tgt_ids = set([w.id for w in spanpair.tgt])

            # Does this span pair contain just one source and one target word?
            is_singles = len(spanpair.src) == 1 and len(spanpair.tgt) == 1

            # If any of the src or tgt ids have already been found as a good match, continue
            # because a word can only ever belong to one group
            # single pairs should always be accepted but are dealt with separately in "create_spans"
            # Always continue if this pair is a singles
            if not is_singles and (not src_ids.isdisjoint(found["src"]) or not tgt_ids.isdisjoint(found["tgt"])):
                continue

            if is_singles or is_valid_sacr_pair(spanpair):
                add_found(spanpair, src_ids, tgt_ids)
            else:
                wpairs = spanpair_to_wordpairs(spanpair)
                for pairs in pair_combs(wpairs, min_length=2):
                    src_ids, tgt_ids = map(set, zip(*[(p.src.id, p.tgt.id) for p in pairs]))
                    tmp_is_singles = len(src_ids) == 1 and len(tgt_ids) == 1

                    if not is_singles and (not src_ids.isdisjoint(found["src"]) or not tgt_ids.isdisjoint(found["tgt"])):
                        continue

                    # First check if this new group is a valid sequence group
                    is_valid_seq, is_mwe = self.is_valid_sequence(pairs, src_ids, tgt_ids)
                    if not is_valid_seq:
                        continue

                    src_words, tgt_words = map(list, zip(*pairs))
                    tmp_src = Span(id=1, words=unique_list(src_words), span_type=SpanType.SACR, attach=False, is_mwe=is_mwe)
                    tmp_tgt = Span(id=1, words=unique_list(tgt_words), span_type=SpanType.SACR, attach=False, is_mwe=is_mwe)
                    tmp_spanpair = SpanPair(tmp_src, tmp_tgt, is_mwe)

                    if tmp_is_singles or is_valid_sacr_pair(tmp_spanpair):
                        add_found(tmp_spanpair, src_ids, tgt_ids)

        self.create_spans(sacr_spans, src_word_groups, tgt_word_groups, found, span_type=SpanType.SACR)

    def create_seq_spans(self):
        src_word_groups = []
        tgt_word_groups = []
        seq_spans = []
        found = {"src": set(), "tgt": set()}

        # pair_combs never returns groups that contain any NULL item
        for pairs in pair_combs(self.aligned_words, min_length=2):
            src_ids, tgt_ids = map(set, zip(*[(p.src.id, p.tgt.id) for p in pairs]))

            # If any of the src or tgt ids have already been found as a good match, continue
            # because a word can only ever belong to one group
            # single pairs should always be accepted
            if not src_ids.isdisjoint(found["src"]) or not tgt_ids.isdisjoint(found["tgt"]):
                continue

            is_valid, is_mwe = self.is_valid_sequence(pairs, src_ids, tgt_ids)
            if is_valid:
                found["src"].update(src_ids)
                found["tgt"].update(tgt_ids)
                src_words, tgt_words = map(list, zip(*pairs))
                src_word_groups.append(src_words)
                tgt_word_groups.append(tgt_words)
                seq_spans.append((min(src_ids), min(tgt_ids), is_mwe))

        self.create_spans(seq_spans, src_word_groups, tgt_word_groups, found, span_type=SpanType.SEQ)

    def create_spans(self, spans, src_word_groups, tgt_word_groups, found, span_type: SpanType):
        # Deal with single pairs separately because unlike other spans, they can be connected with
        # multiple other spans. This includes NULL
        # `pair_combs` starts with the largest groups, so if the current `pairs` only consists
        # of one pair, then that must be a valid pair because it did not belong in other groups
        # This also takes care of pairs with NULL because they are always just one pair
        # (see self.pair_combs).
        # Single pairs with the same src or tgt can appear multiple times (so don't add to "found"):
        # when an item is aligned with multiple items and they do not belong to a larger group together,
        # then those seperate alignments will be separate groups.
        for p in self.aligned_words:
            if (p.src.id in found["src"] and p.tgt.id in found["tgt"]) and not (p.src.is_null or p.tgt.is_null):
                continue

            src_word_groups.append([p.src])
            tgt_word_groups.append([p.tgt])
            spans.append((p.src.id, p.tgt.id, False))

        spans = sorted(set(spans), key=operator.itemgetter(0, 1))
        src_idxs, tgt_idxs, mwes = zip(*spans)
        spans = list(zip(rebase_to_idxs(src_idxs), rebase_to_idxs(tgt_idxs), mwes))

        # Convert src/tgt words in groups of words so that they appear in the same order as in the original sentence
        # So the first item will always be a Null word
        src_word_groups = sorted(unique_list(src_word_groups), key=lambda l: min([w.id for w in l]))
        tgt_word_groups = sorted(unique_list(tgt_word_groups), key=lambda l: min([w.id for w in l]))

        # Convert the groups into actual spans. First items are the NULL spans
        # This means that just like Null words, Null spans have id=0
        src_spans = [
            NullSpan(null_word=words[0], span_type=span_type)
            if words[0].is_null
            else Span(id=idx, words=words, span_type=span_type, doc=self.src)
            for idx, words in enumerate(src_word_groups)
        ]

        tgt_spans = [
            NullSpan(null_word=words[0], span_type=span_type)
            if words[0].is_null
            else Span(id=idx, words=words, span_type=span_type, doc=self.tgt)
            for idx, words in enumerate(tgt_word_groups)
        ]

        # Attach spans to original sentences
        setattr(self.src, f"{span_type}_spans", src_spans)
        setattr(self.tgt, f"{span_type}_spans", tgt_spans)

        # Set MWE
        for src_idx, tgt_idx, mwe in spans:
            src_spans[src_idx].is_mwe = mwe
            tgt_spans[tgt_idx].is_mwe = mwe

        # Create span alignment pairs
        setattr(
            self,
            f"aligned_{span_type}_spans",
            [SpanPair(src_spans[src_idx], tgt_spans[tgt_idx], mwe) for src_idx, tgt_idx, mwe in spans],
        )
        setattr(
            self, f"{span_type}_aligns", [IdxPair(src_idx, tgt_idx) for src_idx, tgt_idx, _ in spans],
        )

    def set_connected(self, attr="deprel"):
        def get_all_connected(start):
            done = set()

            def recursive_connected(item):
                item_repr = f"{item.doc.side}-{item.id}"
                if item_repr in done:
                    return []

                done.add(item_repr)
                connects = []
                for i in item.aligned:
                    i_connects = recursive_connected(i)
                    if i_connects:
                        connects.extend(i_connects)
                return item.aligned + connects

            return sorted(unique_list(recursive_connected(start)), key=operator.attrgetter("id"))

        def get_connected_repr(group):
            src_words = [_w for _w in group if _w.side == Side.SRC]
            return "|".join(
                [
                    f"{src.id}.{getattr(src, attr)}:{','.join([str(tgt.id) + '.' + getattr(tgt, attr) for tgt in src.aligned if not tgt.is_null])}"
                    for src in src_words
                    if not src.is_null
                ]
            )

        connected_set = set()
        # For every source and target word, find all connected words
        # To be as efficient as possible, we keep track of items that we already found.
        # This makes sense, because an item can only be found once because _all_ connected items
        # are taken into account.
        for word in self.src.words + self.tgt.words:
            word_repr = f"{word.doc.side}-{word.id}"
            if word_repr in connected_set:
                continue
            connected_group = get_all_connected(word)
            connected_repr = get_connected_repr(connected_group)
            # Iterate over all the words that are connected to this word
            for connected_word in connected_group:
                c_repr = f"{connected_word.doc.side}-{connected_word.id}"
                if c_repr in connected_set:
                    continue

                connected_word.connected_repr = connected_repr
                # Set c.connected to all connected words that we found EXCLUDING c itself
                for w in connected_group:
                    w_repr = f"{w.doc.side}-{w.id}"
                    if c_repr != w_repr:
                        connected_word.connected.append(w)
                connected_set.add(c_repr)

    def set_cross(self, aligned, attr: str):
        # Given a set of aligned pairs, set a specific cross specified by `attr`
        for pair1, pair2 in combinations(aligned, 2):
            all_items = [pair1.src, pair1.tgt, pair2.src, pair2.tgt]
            # NULL alignments cannot cause crosses
            if any(item.is_null for item in all_items):
                continue

            if pair2.tgt.id < pair1.tgt.id:
                setattr(self, attr, getattr(self, attr) + 1)
                pair1.src.aligned_cross[pair1.tgt.id] += 1
                pair1.tgt.aligned_cross[pair1.src.id] += 1
                pair2.src.aligned_cross[pair2.tgt.id] += 1
                pair2.tgt.aligned_cross[pair2.src.id] += 1

    def set_ted(self):
        # Also sets edit operation for a tree's node. This edit operation is the edit operation that is necessary
        # to change this node in its aligned node, e.g. by matching (~ same connected_repr), renaming (-> other connected_repr),
        # or deleting (-> None). As such, no nodes can have "INSERTION" because we do not have None nodes. That does
        # not mean of course that a tree cannot have insertion operations. It just means that we have no place to put
        # them because we do not have None nodes.

        # TED between an aligned src and tgt sentence are symmetric. However, that is not the same as
        # summing up the astred_cost of each word in the sentence! TED for AlignedSentences counts all operations,
        # including insertions. But a word can never have the "insertion" operation
        # (because insertion is from None -> a Word). Hence, insertion costs will be missing when counting the differences
        # on the word level. DO NOT DO THAT.

        self.ted, self.ted_ops = self.src.tree.get_distance(self.tgt.tree, config=self.ted_config)
        ted_tgt, _ = self.tgt.tree.get_distance(self.src.tree, config=self.ted_config)

        assert self.ted == ted_tgt

        cost = 0
        for src_match, tgt_match in self.ted_ops:
            # node repr as used by the config class to calculate TED
            src_repr = src_match.node.connected_repr if src_match else None
            tgt_repr = tgt_match.node.connected_repr if tgt_match else None

            if src_repr == tgt_repr:
                src_match.astred_op = EditOperation.MATCH
                tgt_match.astred_op = EditOperation.MATCH
            elif src_repr is None:
                tgt_match.astred_op = EditOperation.DELETION
                cost += self.ted_config.costs[EditOperation.DELETION]
            elif tgt_repr is None:
                src_match.astred_op = EditOperation.DELETION
                cost += self.ted_config.costs[EditOperation.DELETION]
            else:
                src_match.astred_op = EditOperation.RENAME
                tgt_match.astred_op = EditOperation.RENAME
                cost += self.ted_config.costs[EditOperation.RENAME]

        assert self.ted == cost
