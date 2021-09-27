import logging

from .aligned import AlignedSentences
from .aligner import Aligner
from .sentence import Sentence
from .span import NullSpan, Span
from .tree import Tree
from .word import Null, Word


def set_logger():
    # Do not needlessly expose variables
    logger = logging.getLogger("astred")
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmt="%(asctime)s - [%(levelname)s]: %(message)s", datefmt="%d-%b %H:%M:%S"))
    logger.addHandler(sh)


set_logger()

__version__ = "0.9.6"
