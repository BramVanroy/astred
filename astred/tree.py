from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Union

from apted import APTED
from apted import Config as AptedConfig
from nltk.draw.tree import draw_trees
from nltk.tree import ParentedTree as NltkTree

from .enum import EditOperation


class AstredConfig(AptedConfig):
    def __init__(self, attrs="connected_repr", attrs_sep="-"):
        self.attrs = [attrs] if isinstance(attrs, str) else attrs
        self.attrs_sep = attrs_sep

    def node_repr(self, tree):
        return (
            self.attrs_sep.join([str(getattr(tree.node, attr)) for attr in self.attrs])
            if tree
            else None
        )

    def rename(self, node1: Tree, node2: Tree):
        return int(self.node_repr(node1) != self.node_repr(node2))

    def children(self, node):
        """Returns children of node"""
        return getattr(node, "children", [])


@dataclass
class TreeBase(ABC):
    node: Any
    children: List[TreeBase] = field(default_factory=list)
    level: int = field(default=0)
    parent: Tree = field(default=None, repr=False, init=False)
    root: Tree = field(default=None, repr=False, init=False)
    doc: Any = field(default=None, repr=False)
    astred_op: EditOperation = field(default=None, init=False)

    def __post_init__(self):
        if any(not isinstance(child, self.__class__) for child in self.children):
            raise ValueError(
                "A tree's children must have the same class as its parent."
            )
        self.attach_self_to_children()
        if self.node.is_root:
            self.attach_self_to_subtrees()

    def as_embedded_tuples(self):
        """Create embedded/recursive tuples in the form of (ROOT, [(child1, [subchildren...]), (child2, [subchildren2...]), ...])
        Returns
        -------

        """

        def children(root):
            return root.node, [children(word) for word in root.children]

        return children(self)

    def subtrees(self, include_self: bool = True):
        """Return a flat list of the unique, full subtrees (so no combinations or subparts of subtrees)
        Parameters
        ----------
        include_self

        Returns
        -------

        """

        def _recursive_children(node, _descendants=None):
            if _descendants is None:
                _descendants = []
            else:
                _descendants.append(node)

            for c in node.children:
                _recursive_children(c, _descendants)

            return _descendants

        if include_self:
            return [self] + _recursive_children(self)
        else:
            return _recursive_children(self)

    def attach_self_to_children(self):
        for subtree in self.children:
            subtree.parent = self

    def attach_self_to_subtrees(self):
        for subtree in self.subtrees(include_self=False):
            subtree.root = self

    def to_string(
        self,
        attrs: Union[List[str], str] = "text",
        attrs_sep: str = "-",
        parens: Union[List[str], Tuple[str], str] = "()",
        pretty: bool = False,
        end_on_newline: bool = False,
        node_sep: str = " ",
        indent: str = "\t",
    ):
        if len(parens) != 2:
            raise ValueError(
                "'parens' must contain exactly two characters to use as"
                " the start and end character respectively"
            )
        start_parens, end_parens = parens[0], parens[-1]

        if isinstance(attrs, str):
            attrs = [attrs]

        def build_str(tree, is_last_child=True):
            s = start_parens
            s += (
                tree.node
                if isinstance(tree.node, str)
                else attrs_sep.join([str(getattr(tree.node, attr)) for attr in attrs])
            )
            s += " "
            n_children = len(tree.children)

            for child_idx, child in enumerate(tree.children, 1):
                s += f"\n{indent * child.level}" if pretty else ""
                s += build_str(child, is_last_child=child_idx == n_children)

            s += (
                f"\n{indent * tree.level}"
                if pretty and end_on_newline and tree.children
                else ""
            )
            s += end_parens
            s += node_sep if not is_last_child else ""
            return s

        return build_str(self)

    def get_distance(self, tgt_tree: Tree, config=None):
        """Calculate the distance between self and target tree.
        :return: the tree edit distance for the given trees and the required operations
        """
        config = AstredConfig() if config is None else config
        apted = APTED(self, tgt_tree, config)
        dist = apted.compute_edit_distance()
        opts = apted.compute_edit_mapping()

        return dist, opts

    @classmethod
    def draw_trees(cls, *trees, **to_string_kwargs):
        strings = [tree.to_string(**to_string_kwargs) for tree in trees]
        try:
            nltk_trees = [NltkTree.fromstring(s) for s in strings]
        except Exception as e:
            raise ValueError(
                "When calling 'draw_trees' on a subclass of TreeBase, the implementation of 'to_string'"
                " MUST be compatible with NLTK's Tree.fronstring method."
                " See the stacktrace above for more details."
            ) from e

        draw_trees(*nltk_trees)


@dataclass
class Tree(TreeBase):
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(node={self.node.text}, children={[w.node.text for w in self.children]},"
            f" level={self.level})"
        )

    def __post_init__(self):
        super(Tree, self).__post_init__()
        self.attach_self_to_node()

    def attach_self_to_node(self):
        """We do not want created subtrees (e.g. SpanTrees) to overwrite a word's tree attribute."""
        if not self.node.tree:
            self.node.tree = self

    @classmethod
    def from_sentence(cls, sentence: Any):
        sent_root = [word for word in sentence if word.is_root]
        n_roots = len(sent_root)

        if n_roots != 1:
            raise ValueError(
                f"A sentence must have exactly only root word to create a {cls.__name__}."
                f" Currently {n_roots} are given."
            )
        sent_root = sent_root[0]
        return cls.from_span(sentence, sent_root, sent_root)

    @classmethod
    def from_span(cls, span, span_root, doc=None):
        if span_root not in span:
            raise ValueError("'span_root' must be an element of 'span'")

        def get_children(head_idx):
            return [word for word in span if word.head == head_idx]

        def parse(root, level=-1):
            children = get_children(int(root.id))
            level += 1
            child_trees = [parse(n, level=level) for n in children] if children else []
            return cls(root, children=child_trees, level=level, doc=doc)

        return parse(span_root)
