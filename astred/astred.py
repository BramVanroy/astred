from typing import Tuple, Optional, List, Dict, Union

from .sacr import SACr
from .tree import GenericTree
from .utils import get_distance


class ASTrED(SACr):
    """ASTrED (Aligned Syntactic Tree Edit Distance) provides an interface
    to calculate tree edit distance, and more importantly astred itself.
    ASTrED is an extension to tree edit distance between the dependency trees
    of the source and target text: rather than comparing the trees as-is, we
    merge the labels between the source and target trees based on the word
    alignments.

    Parameters
    ----------
    alignments : str
        The word alignments, in GIZA format
    src_segment : str
        The tokenized source string
    tgt_segment : str
        The tokenized target string
    kwargs
        Additional keyword arguments that will be passed to the super class `SACr`

    Attributes
    ----------
    astred
    ted
    label_changes
    default_label_changes
    avg_token_label_changes
    overlap_label_changes

    """

    def __init__(self, alignments: str, src_segment: str, tgt_segment: str, **kwargs):
        super().__init__(alignments, src_segment, tgt_segment, **kwargs)

        self._ted: Optional[Tuple[int, List]] = None
        self._astred: Optional[Tuple[int, List]] = None

        self._merged_src_tree: Optional[GenericTree] = None
        self._merged_tgt_tree: Optional[GenericTree] = None

        self._src_labels_map: Optional[Dict[int, str]] = None
        self._tgt_labels_map: Optional[Dict[int, str]] = None
        self._merged_src_labels_map: Optional[Dict[int, str]] = None
        self._merged_tgt_labels_map: Optional[Dict[int, str]] = None

        self._label_changes: Optional[Dict[str, Union[float, Dict]]] = None
        self._default_label_changes: Optional[float] = None
        self._avg_token_label_changes: Optional[Dict[str, float]] = None
        self._overlap_label_changes: Optional[Dict[str, float]] = None

    @property
    def astred(self):
        if self._astred is None:
            self._astred = get_distance(self.merged_src_tree, self.merged_tgt_tree)
        return self._astred

    @property
    def avg_token_label_changes(self):
        if self._avg_token_label_changes is None:
            # method='average': token_avg label changes
            # cf directional_dist()
            # this is NOT necessarily bidirectionaly the same, so take average!
            # e.g. A-A B-B B-C C-D
            self._avg_token_label_changes = {
                "src": self._directional_n_changes(
                    self.src2tgtlist_d,
                    self.src_labels_map,
                    self.tgt_labels_map,
                    method="token_avg",
                ),
                "tgt": self._directional_n_changes(
                    self.tgt2srclist_d,
                    self.tgt_labels_map,
                    self.src_labels_map,
                    method="token_avg",
                ),
            }
            self._avg_token_label_changes["avg"] = (
                self._avg_token_label_changes["src"] + self._avg_token_label_changes["tgt"]
            ) / 2

        return self._avg_token_label_changes

    @property
    def default_label_changes(self):
        # default label changes
        # this is symmetrically the same, so we only do it for src-to-tgt
        if self._default_label_changes is None:
            self._default_label_changes = self._directional_n_changes(
                self.src2tgtlist_d, self.src_labels_map, self.tgt_labels_map
            )
        return self._default_label_changes

    @property
    def label_changes(self):
        if self._label_changes is None:
            self._label_changes = {
                "default": self.default_label_changes,
                "avg_token": self.avg_token_label_changes,
                "overlap": self.overlap_label_changes,
            }
        return self._label_changes

    @property
    def merged_src_tree(self):
        if self._merged_src_tree is None:
            self._merged_src_tree = self._get_merged_tree('src')
        return self._merged_src_tree

    @property
    def merged_src_labels_map(self):
        if self._merged_src_labels_map is None:
            self._merged_src_labels_map, self._merged_tgt_labels_map = self._merge_maps()
        return self._merged_src_labels_map

    @property
    def merged_tgt_labels_map(self):
        if self._merged_tgt_labels_map is None:
            self._merged_src_labels_map, self._merged_tgt_labels_map = self._merge_maps()
        return self._merged_tgt_labels_map

    @property
    def merged_tgt_tree(self):
        if self._merged_tgt_tree is None:
            self._merged_tgt_tree = self._get_merged_tree('tgt')
        return self._merged_tgt_tree

    @property
    def overlap_label_changes(self):
        # method='overlap': only count when alignments don't have overlap at all
        # when there is an overlap between src-tgt: 0, if no overlap: 1
        # cf directional_dist()
        # this is NOT necessarily bidirectionaly the same, so take average!
        # e.g. A-A A-B B-B
        if self._overlap_label_changes is None:
            self._overlap_label_changes = {
                "src": self._directional_n_changes(
                    self.src2tgtlist_d,
                    self.src_labels_map,
                    self.tgt_labels_map,
                    method="overlap",
                ),
                "tgt": self._directional_n_changes(
                    self.tgt2srclist_d,
                    self.tgt_labels_map,
                    self.src_labels_map,
                    method="overlap",
                ),
            }
            self._overlap_label_changes["avg"] = (
                self._overlap_label_changes["src"] + self._overlap_label_changes["tgt"]
            ) / 2
        return self._overlap_label_changes

    @property
    def ted(self):
        # ted is a tuple containing the distance and all operations
        if self._ted is None:
            self._ted = get_distance(self.src_tree, self.tgt_tree)
        return self._ted

    @property
    def src_labels_map(self):
        if self._src_labels_map is None:
            self._src_labels_map = {
                idx: f"src_{tree.label()}"
                for idx, tree in self.src_tree.word_order_idx_mapping.items()
            }
        return self._src_labels_map

    @property
    def tgt_labels_map(self):
        if self._tgt_labels_map is None:
            self._tgt_labels_map = {
                idx: f"tgt_{tree.label()}"
                for idx, tree in self.tgt_tree.word_order_idx_mapping.items()
            }
        return self._tgt_labels_map

    def _merge_maps(self):
        """Merge two dictionary maps based on their alignments.
           The resulting trees contain the information of both trees.
        Returns
        -------

        """

        mod_src2tgtlist_d = self._change_map(self.src2tgtlist_d, "src", "tgt")
        mod_tgt2srclist_d = self._change_map(self.tgt2srclist_d, "tgt", "src")
        merged_maps = {**mod_src2tgtlist_d, **mod_tgt2srclist_d}

        def _recurse(key, _done, _group):
            # if merged_maps[key] is empty list
            if key in done or not merged_maps[key]:
                return
            else:
                done.add(key)
                group.append(key)
                for _i in merged_maps[key]:
                    _recurse(_i, _done, _group)

        done = set()
        connected_groups = []
        # recursively group all connected items together
        # this is important because one src item might be align with two target
        # items of which one is also connected to another src item
        # this means that complex connected chains can occur
        for i in merged_maps.keys():
            group = []
            _recurse(i, done, group)
            if group:
                connected_groups.append(group)

        # ensure that all keys are traversed
        assert set(merged_maps.keys()) == done

        mod_src_labels_map = {}
        mod_tgt_labels_map = {}

        # convert the connected groups to labels
        # set the src_idx/tgt_idx names map so that each index now has an updated label
        for group in connected_groups:
            # For each group, get the source indices
            # Then for each source index, get its target indices
            # Convert all indices to labels
            # Output is dict of src_idx: [tgt_idxs]
            group_mapping = {
                item: mod_src2tgtlist_d[item] for item in group if item.startswith("src_")
            }
            group_mapping = {
                int(src[4:]): [int(tgt[4:]) for tgt in v] for src, v in group_mapping.items()
            }
            group_mapping = {
                self.src_labels_map[src_idx]: [
                    self.tgt_labels_map[tgt_idx] for tgt_idx in v if tgt_idx != -1
                ]
                for src_idx, v in group_mapping.items()
                if src_idx != -1
            }

            # Serialize the mapping to get a string representation
            serialized = self._serialize_group(group_mapping)

            # For all items in the group, set the serialized group
            for item in group:
                is_src = item.startswith("src_")
                idx = int(item[4:])

                if is_src:
                    mod_src_labels_map[idx] = serialized
                else:
                    mod_tgt_labels_map[idx] = serialized

        merged_src_labels_map = {**self.src_labels_map, **mod_src_labels_map}
        merged_tgt_labels_map = {**self.tgt_labels_map, **mod_tgt_labels_map}

        return merged_src_labels_map, merged_tgt_labels_map

    def _get_merged_tree(self, side):
        tree_copy = getattr(self, f"{side}_tree")
        for idx, name in getattr(self, f"merged_{side}_labels_map").items():
            # Skip null alignments (-1)
            if idx != -1:
                tree_copy.word_order_idx_mapping[idx].set_label(name)

        return tree_copy

    @staticmethod
    def _change_map(dmap, key_text, val_text):
        """Change the key and values for a map by adding
            a 'key_text' and 'tgt_text' (e.g. 'src' or 'tgt')
            to be able to distinguish the two.

        Parameters
        ----------
        dmap the initial mapping
        key_text key text to use (e.g. 'src', 'tgt')
        val_text al text to use (e.g. 'src', 'tgt')

        Returns
        -------
        The modified dict with changed labels
        """
        m = {}
        for k, v in dmap.items():
            m[f"{key_text}_{k}"] = [f"{val_text}_{i}" for i in v]

        return m

    @staticmethod
    def _directional_n_changes(align_map, label_map1, label_map2, method="default"):
        """

        Parameters
        ----------
        align_map
            an alignment map of the tokens of one side and the items on the other side that they are
        aligned with, e.g. {0: [0], 1: [1, 2]}
        label_map1
             map of token IDs to their corresponding label, e.g. {0: 'src_det-1', 1: 'src_amod-1'}
        label_map2
            map of token IDs to their corresponding label, e.g. {0: 'src_det-1', 1: 'src_amod-1'}
        method
            method to calculate the number of label changes. Must be one of 'default', 'token_avg',
        'overlap'. Default calculates the differences in label changes in a one-on-one manner. Token_avg checks
        how many items on average have changed label during alignment. Overlap only counts those changes where
        all aligned items have different labels than the original.

        Returns
        -------
        the number of changes according to `method`

        """
        n_changes = 0
        for idx, dir_idxs in align_map.items():
            if idx == -1:
                continue
            main_label = label_map1[idx][4:].split("-")[0]
            dir_labels = [label_map2[i][4:].split("-")[0] for i in dir_idxs if i != -1]

            idx_changes = 0
            if method == "default" or method == "token_avg":
                idx_changes = sum([1 for dir_label in dir_labels if dir_label != main_label])

                if method == "token_avg":
                    idx_changes /= len(dir_idxs)
            elif method == "overlap":
                # if there is any overlap, i.e. a label is is present in its alignment,
                # then count as 0, otherwise 1 (no overlap, completely different labels) (cast bool to int)
                idx_changes = int(not any(dir_label == main_label for dir_label in dir_labels))
            else:
                raise ValueError("'method' must be one of 'default', 'token_avg', 'overlap'")
            n_changes += idx_changes

        # return float for consistency
        return float(n_changes)

    @staticmethod
    def _serialize_group(group):
        """ Serialize a dictionary of src_idx->tgt_idxs
        :param group: input dictionary
        :return: string representation of 'group'
        """
        return "|".join([f"{k}:{','.join(v)}" for k, v in group.items()])
