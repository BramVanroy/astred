from collections import Counter, defaultdict

from nltk import ParentedTree

from .utils import load_nlp


class GenericTree(ParentedTree):
    NLPS = {}
    """ GenericTree that extends the NLTK Tree by providing some convenience methods
    to transform a spaCy Span into a tree structure.
    """

    def __init__(
        self,
        node,
        is_root=False,
        text=None,
        word_order_idx=0,
        add_word_order_tree=False,
        level=0,
        children=None,
    ):
        """ Build GenericTree
        :param node: this tree's label
        :param is_root: is this the root tree (main tree). If so, the indices will be set automatically.
        See self.set_indices()
        :param word_order_idx: the index of the word/label in the main span
        """
        super().__init__(node, children)
        self.text = text
        self.text_tree = None
        if text:
            self.text_tree = GenericTree(
                text,
                is_root=False,
                word_order_idx=word_order_idx,
                children=[c.text_tree for c in children],
            )

        self.word_order_idx = word_order_idx
        self.word_idx_tree = None
        self.add_word_order_tree = add_word_order_tree
        if add_word_order_tree:
            self.word_idx_tree = GenericTree(
                word_order_idx,
                is_root=False,
                word_order_idx=word_order_idx,
                children=[c.word_idx_tree for c in children],
            )

        self.level = level
        self.is_root = is_root
        if is_root:
            self._set_indices()

    @property
    def children(self):
        """ Returns the direct children trees of this tree's root
        :return: a list of immediate child trees
        """
        return [c for c in self if isinstance(c, GenericTree)]

    @property
    def descendants(self):
        """ Returns all descendant trees of this tree in a depth-first, left-to-right fashion.
            Only returns the actual trees and not subsets or combinations.
        :return: a list of all descendant subtrees
        """

        def _recursive_children(node, _descendants=None):
            if _descendants is None:
                _descendants = []
            else:
                _descendants.append(node)

            for c in node.children:
                _recursive_children(c, _descendants)

            return _descendants

        return _recursive_children(self)

    def grouped_per_level(self, idxs):
        """ Groups given indices per level. """
        levels_d = defaultdict(list)
        for i in idxs:
            t = self.word_order_idx_mapping[i]
            levels_d[t.level].append(i)

        return dict(levels_d)

    def grouped_per_parent(self, idxs, reverse=False):
        """ Groups given indices per parent node. """
        parents_d = {} if reverse else defaultdict(list)

        for i in idxs:
            t = self.word_order_idx_mapping[i]
            if reverse:
                parents_d[i] = t.parent().word_order_idx
            else:
                parents_d[t.parent().word_order_idx].append(i)

        return dict(parents_d)

    @property
    def word_order_idx_mapping(self):
        """ Get a mapping of self and its descendants, going from
            word_order_idx to GenericTree
        :return: sorted dictionary of integers (word_order_idx) to GenericTree
        """
        d = {d.word_order_idx: d for d in self.descendants}
        d[self.word_order_idx] = self
        return dict(sorted(d.items()))

    def _get_children(self):
        """ Convenience method to be used in cls.distance()
        :return: a list of immediate subtrees
        """
        return self.children

    def add_word_idx_to_label(self):
        """ Add word idx to the label to make it visually more clear which word
            is where in the tree. Only do this when drawing the tree, NOT during processing
            or calculating the label changes, because that might give unexpected resutls.
            Should not be called for each subtree, but only on the main tree!
        :return:
        """
        for idx, tree in self.word_order_idx_mapping.items():
            label = tree.label()
            tree.set_label(f"{idx}<{label}")

    def _set_indices(self):
        """ For the whole main tree, set an index on all labels to distinguish ones that have
        the same label. So if two nodes are 'aux', convert them into 'aux-1' and 'aux-2'
        Should not be called for each subtree, but only on the main tree!
        :return:
        """
        label_counter = Counter()
        for idx, tree in self.word_order_idx_mapping.items():
            label = tree.label()
            label_counter[label] += 1
            tree.set_label(f"{label}-{label_counter[label]}")

    def to_string(self, parens="()", pretty=False):
        """ Convert a tree to a string representation (using NLTK's tree methods).

        :param parens: which parenthesis to use to visually represent nodes
        :param pretty: whether to pretty print (structured) or on one line
        :return:
        """
        if pretty:
            form = self.pformat(parens=parens)
        else:
            form = self._pformat_flat("", parens, False).replace(" ", "")

        return form

    @classmethod
    def from_stanza(cls, sentence, label="deprel"):
        """ Converts a parsed Stanza sentence into a GenericTree
        :param sentence: stanza sentence
        :param label: which Stanza Word property to use as labels
        :return: GenericTree based on spaCy span
        """

        def get_children(head_idx):
            return [word for word in sentence.words if word.head == head_idx]

        def parse(root, is_root=False, level=-1):
            children = get_children(int(root.id))
            # Only get the main label, not any subtypes
            # See https://universaldependencies.org/ext-dep-index.html
            root_label = getattr(root, label).split(":")[0]
            level += 1
            child_trees = [parse(n, level=level) for n in children] if children else []
            return cls(
                root_label,
                is_root=is_root,
                text=root.text,
                word_order_idx=int(root.id) - 1,
                add_word_order_tree=True,
                level=level,
                children=child_trees,
            )

        return parse(get_children(0)[0], is_root=True)

    @classmethod
    def from_string(cls, text, lang_or_model="en", **kwargs):
        """ Convert a string into a GenericTree

        :param text: text to process
        :param lang_or_model: stanfordnlp language or model
        :param tokenize_pretokenized: whether or not the text is pretokenized and presegmented
        :param nlp: an existing NLP instance.
        :param use_gpu:
        :return: a GenericTree, representing the given 'text'
        """

        if lang_or_model not in cls.NLPS:
            cls.NLPS[lang_or_model] = load_nlp(lang_or_model, **kwargs)

        doc = cls.NLPS[lang_or_model](text)
        main_sent = doc.sentences[0]
        return cls.from_stanza(main_sent)

    @classmethod
    def convert(cls, val, is_root=True, level=0):
        """ Used by super class when copying the tree """
        if isinstance(val, GenericTree):
            children = [cls.convert(child) for child in val.children]
            return cls(
                val._label,
                is_root=val.is_root,
                text=val.text,
                word_order_idx=val.word_order_idx,
                add_word_order_tree=val.add_word_order_tree,
                level=val.level,
                children=children,
            )
        else:
            return val

    @classmethod
    def init_nlp(cls, lang, nlp):
        if lang not in cls.NLPS:
            cls.NLPS[lang] = nlp
