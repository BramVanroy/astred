from copy import deepcopy

from astred import GenericTree
from astred.utils import draw_trees

def main():
    tree = GenericTree.from_string("I like my grandma 's cookies !")
    print(tree)
    tree_copy = deepcopy(tree)
    draw_trees(tree.text_tree, tree_copy)

if __name__ == '__main__':
    main()