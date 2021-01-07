from __future__ import annotations

import operator
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Set, Tuple, Union

from .enum import EditOperation, Side, SpanType
from .pairs import IdxPair
from .sentence import Sentence
from .span import NullSpan, Span, SpanPair
from .tree import AstredConfig, Tree
from .utils import pair_combs, rebase_to_idxs, unique_list
from .word import WordPair, spanpair_to_wordpairs


@dataclass
class AlignedSentences:
    src: Sentence
    tgt: Sentence
    word_aligns: Union[List[IdxPair], str]
    allow_mwe: bool = field(default=False)

    aligned_words: List[WordPair] = field(
        default_factory=list, repr=False, compare=False, init=False
    )
    word_cross: int = field(default=0, compare=False, init=False)

    aligned_seq_spans: List[SpanPair] = field(
        default_factory=list, repr=False, compare=False, init=False
    )
    seq_aligns: List[IdxPair] = field(
        default_factory=list, repr=False, compare=False, init=False
    )
    seq_cross: int = field(default=0, compare=False, init=False)

    aligned_sacr_spans: List[SpanPair] = field(
        default_factory=list, repr=False, compare=False, init=False
    )
    sacr_aligns: List[IdxPair] = field(
        default_factory=list, repr=False, compare=False, init=False
    )
    sacr_cross: int = field(default=0, compare=False, init=False)

    ted_config: AstredConfig = field(default=AstredConfig())
    ted: int = field(default=0, compare=False, init=False)
    ted_ops: List[Tuple[Tree]] = field(
        default_factory=list, compare=False, init=False, repr=False
    )

    def __getitem__(self, idx):
        return self.aligned_words[idx]

    def __iter__(self):
        return iter(self.aligned_words)

    def __len__(self):
        return len(self.aligned_words)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(src={self.src}, tgt={self.tgt},"
            f" aligns={[(i.src, i.tgt) for i in self.word_aligns]})"
        )

    def __post_init__(self):
        if isinstance(self.word_aligns, str):
            self.word_aligns = [
                IdxPair(*map(int, align.split("-")))
                for align in self.word_aligns.split(" ")
            ]

        self.attach_sentences()

        # +1 because 0-index is reserved for NULL
        self.word_aligns = [IdxPair(p.src + 1, p.tgt + 1) for p in self.word_aligns]
        self.add_null_aligns()
        self.word_aligns.sort(key=operator.attrgetter("src", "tgt"))

        self.aligned_words = [
            WordPair(self.src[align.src], self.tgt[align.tgt])
            for align in self.word_aligns
        ]

        self.attach_pairs(self.aligned_words)
        self.set_cross(self.aligned_words, "word_cross")

        # SEQUENCES
        self.create_seq_spans()
        # Attach spans to each other
        self.attach_pairs(self.aligned_seq_spans)
        # Calculate seq cross
        self.set_cross(self.aligned_seq_spans, "seq_cross")

        if self.src.tree and self.tgt.tree:
            # SACR
            self.create_sacr_spans()
            # Attach spans to each other
            self.attach_pairs(self.aligned_sacr_spans)
            # Calculate seq cross
            self.set_cross(self.aligned_sacr_spans, "sacr_cross")

            # TED
            self.set_ted()
            self.set_connected()

    @property
    def idxs_d(self) -> Dict[str, Set[int]]:
        src, tgt = zip(*self.word_aligns)
        return {"src": src, "tgt": tgt}

    def add_null_aligns(self):
        # Fill in 0 idx for words that are not aligned
        self.word_aligns += [
            IdxPair(idx, 0)
            for idx in range(len(self.src))
            if idx not in self.idxs_d["src"]
        ]
        self.word_aligns += [
            IdxPair(0, idx)
            for idx in range(len(self.tgt))
            if idx not in self.idxs_d["tgt"]
        ]

    def attach_sentences(self):
        self.tgt.aligned_sentence = self.src
        self.src.side = Side.SRC
        self.src.aligned_sentence = self.tgt
        self.tgt.side = Side.TGT

    @staticmethod
    def attach_pairs(pairs):
        for pair in pairs:
            pair.src.add_aligned(pair.tgt)
            pair.tgt.add_aligned(pair.src)

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

            return sorted(
                unique_list(recursive_connected(start)), key=operator.attrgetter("id")
            )

        def get_connected_repr(group):
            src_words = [_w for _w in group if _w.side == Side.SRC]
            return "|".join(
                [
                    f"{getattr(src, attr)}:{','.join([getattr(tgt, attr) for tgt in src.aligned if not tgt.is_null])}"
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

    @staticmethod
    def idxs_are_consecutive(pairs):
        # NULL has already been filtered
        prev = IdxPair(pairs[0].src.id, pairs[0].tgt.id)
        for pair in pairs[1:]:
            if not (
                (pair.src.id == prev.src + 1 and pair.tgt.id == prev.tgt + 1)
                or (pair.src.id == prev.src and pair.tgt.id == prev.tgt + 1)
                or (pair.src.id == prev.src + 1 and pair.tgt.id == prev.tgt)
            ):
                return False
            prev = IdxPair(pair.src.id, pair.tgt.id)
        return True

    @staticmethod
    def check_mwe_and_external_align(pairs, src_ids, tgt_ids):
        n_src = len(unique_list([p.src for p in pairs]))
        n_tgt = len(unique_list([p.tgt for p in pairs]))

        # MWE must consist of more than one source and target word
        # Later we then check whether each word is aligned with all other words in the group
        is_mwe = n_src > 1 and n_tgt > 1
        has_external_align = False
        for pair in pairs:
            aligned_to_src = set([w.id for w in pair.src.aligned])
            aligned_to_tgt = set([w.id for w in pair.tgt.aligned])

            # check that each source word is attached to all target words
            if is_mwe and aligned_to_src != src_ids or aligned_to_tgt != tgt_ids:
                is_mwe = False

            if not aligned_to_src.issubset(tgt_ids) or not aligned_to_tgt.issubset(
                src_ids
            ):
                has_external_align = True

            if not is_mwe and has_external_align:
                break

        return is_mwe, has_external_align

    def create_seq_spans(self):
        src_word_groups = []
        tgt_word_groups = []
        seq_spans = []
        found = {"src": set(), "tgt": set()}

        for pairs in pair_combs(self.aligned_words, min_length=2):
            src_ids, tgt_ids = map(set, zip(*[(p.src.id, p.tgt.id) for p in pairs]))

            # If any of the src or tgt ids have already been found as a good match, continue
            # because a word can only ever belong to one group
            # single pairs should always be accepted
            if not src_ids.isdisjoint(found["src"]) or not tgt_ids.isdisjoint(
                found["tgt"]
            ):
                continue

            # Check if:
            # Is the group an MWE (all src aligned with all tgt)?
            # Is the group a regular consecutive group with no external alignments?
            is_mwe, has_external_align = self.check_mwe_and_external_align(
                pairs, src_ids, tgt_ids
            )
            is_valid = self.allow_mwe and is_mwe
            is_valid = is_valid or (
                self.idxs_are_consecutive(pairs) and not has_external_align
            )

            if is_valid:
                found["src"].update(src_ids)
                found["tgt"].update(tgt_ids)
                src_words, tgt_words = map(list, zip(*pairs))
                src_word_groups.append(src_words)
                tgt_word_groups.append(tgt_words)
                seq_spans.append((min(src_ids), min(tgt_ids), is_mwe))

        self.create_spans(
            seq_spans, src_word_groups, tgt_word_groups, found, span_type=SpanType.SEQ
        )

    def create_sacr_spans(self):
        def is_valid_sacr_pair(pair):
            is_valid = (
                pair.src.is_valid_subtree
                and pair.tgt.is_valid_subtree
                or (self.allow_mwe and spanpair.mwe)
            )
            is_valid = is_valid or (pair.src.is_null and pair.tgt.is_null)

            return is_valid

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
            sacr_spans.append((min(s_ids), min(t_ids), spair.mwe))

        for spanpair in self.aligned_seq_spans:
            src_ids = set([w.id for w in spanpair.src])
            tgt_ids = set([w.id for w in spanpair.tgt])

            # If any of the src or tgt ids have already been found as a good match, continue
            # because a word can only ever belong to one group
            # single pairs should always be accepted
            if not src_ids.isdisjoint(found["src"]) or not tgt_ids.isdisjoint(
                found["tgt"]
            ):
                continue

            if is_valid_sacr_pair(spanpair):
                add_found(spanpair, src_ids, tgt_ids)
            else:
                wpairs = spanpair_to_wordpairs(spanpair)
                for pairs in pair_combs(wpairs, min_length=2):
                    src_ids, tgt_ids = map(
                        set, zip(*[(p.src.id, p.tgt.id) for p in pairs])
                    )
                    src_words, tgt_words = map(list, zip(*pairs))

                    temp_src = Span(
                        id=1,
                        words=unique_list(src_words),
                        span_type=SpanType.SACR,
                        attach=False,
                    )
                    temp_tgt = Span(
                        id=1,
                        words=unique_list(tgt_words),
                        span_type=SpanType.SACR,
                        attach=False,
                    )

                    if temp_src.is_valid_subtree and temp_tgt.is_valid_subtree:
                        add_found(SpanPair(temp_src, temp_tgt, False), src_ids, tgt_ids)

        self.create_spans(
            sacr_spans, src_word_groups, tgt_word_groups, found, span_type=SpanType.SACR
        )

    def create_spans(
        self, spans, src_word_groups, tgt_word_groups, found, span_type: SpanType
    ):
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
            if p.src.id in found["src"] or p.tgt.id in found["tgt"]:
                continue

            src_word_groups.append([p.src])
            tgt_word_groups.append([p.tgt])
            spans.append((p.src.id, p.tgt.id, False))

        src_idxs, tgt_idxs, mwes = zip(*spans)
        spans = sorted(
            zip(rebase_to_idxs(src_idxs), rebase_to_idxs(tgt_idxs), mwes),
            key=operator.itemgetter(0, 1),
        )

        # Convert src/tgt words in groups of words so that they appear in the same order as in the original sentence
        # So the first item will always be a Null word
        src_word_groups = sorted(
            unique_list(src_word_groups), key=lambda l: min([w.id for w in l])
        )
        tgt_word_groups = sorted(
            unique_list(tgt_word_groups), key=lambda l: min([w.id for w in l])
        )

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

        # Create span alignment pairs
        setattr(
            self,
            f"aligned_{span_type}_spans",
            [
                SpanPair(src_spans[src_idx], tgt_spans[tgt_idx], mwe)
                for src_idx, tgt_idx, mwe in spans
            ],
        )
        setattr(
            self,
            f"{span_type}_aligns",
            [IdxPair(src_idx, tgt_idx) for src_idx, tgt_idx, _ in spans],
        )

    def set_ted(self):
        # Also sets edit operation for a tree's node. This edit operation is the edit operation that is necessary
        # to change this node in its aligned node, e.g. by matching (~ same node_repr), renaming (-> other node_repr),
        # or deleting (-> None). As such, no nodes can have "INSERTION" because we do not have None nodes. That does
        # not mean of course that a tree cannot have insertion operations. It just means that we have no place to put
        # them because we do not have None nodes.

        self.ted, self.ted_ops = self.src.tree.get_distance(
            self.tgt.tree, config=self.ted_config
        )

        for src_match, tgt_match in self.ted_ops:
            # node repr as used by the config class to calculate TED
            src_repr = self.ted_config.node_repr(src_match)
            tgt_repr = self.ted_config.node_repr(tgt_match)

            if src_repr == tgt_repr:
                src_match.astred_op = EditOperation.MATCH
                tgt_match.astred_op = EditOperation.MATCH
            elif src_repr is None:
                tgt_match.astred_op = EditOperation.DELETION
            elif tgt_repr is None:
                src_match.astred_op = EditOperation.DELETION
            else:
                src_match.astred_op = EditOperation.RENAME
                tgt_match.astred_op = EditOperation.RENAME
