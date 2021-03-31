Syntactic equivalence metrics
=============================

Documentation, and more tests to be added.


Example notebooks
-----------------

A couple example notebooks exist, each with a different grade of automation for the initialisation of the aligned object. 
Once an aligned object has been created, the functionality is identical.

- `High automation`_: *automate all the things*. Tokenisation, parsing, and word alignment is done automatically.
- `Normal automation`_: the typical scenario where you have tokenised and aligned text that is not parsed yet
- `No automation`_: full-manual mode, where you provide all the required information, including dependency labels and heads

.. _High automation: examples/full-auto.ipynb
.. _Normal automation: examples/automatic-parsing.ipynb
.. _No automation: examples/full-manual.ipynb


Installation
------------

Requires Python 3.7 or higher. To keep the overhead low, a default parser is NOT installed. Currently both `spaCy`_ and
`stanza`_ are supported and you can choose which one to use. Stanza is recommended for bilingual research (because it
is ensured that all of its models use Universal Dependencies), but spaCy can be used as well. The latter is especially
used for monolingual comparisons, or if you are not interested in the linguistic comparisons and only require word
reordering metrics.

A pre-release is available on PyPi. You can install it with pip as follows.

.. code-block:: bash

    # Install with stanza (recommended)
    pip install astred[stanza]
    # ... or install with spacy
    pip install astred[spacy]
    # ... or install with both and decide later
    pip install astred[parsers]

If you want to use spaCy, you have to make sure that you `install`_ the required models manually, which cannot be
automated.

.. _spaCy: https://spacy.io/
.. _stanza: https://github.com/stanfordnlp/stanza
.. _install: https://spacy.io/usage/models

Automatic Word Alignment
------------------------

Automatic word alignment is supported by using a modified version of `Awesome Align`_ under the hood. This is a neural
word aligner that uses transfer learning with multilingual models to do word alignment. It does require
some manual installation work. Specifically, you need to install the :code:`astred_compat` branch from `this fork`_.
If you are using pip, you can run the following command:

.. code-block:: bash

    pip install git+https://github.com/BramVanroy/awesome-align.git@astred_compat

Awesome Align requires PyTorch, like :code:`stanza` above.

If it is installed, you can initialize :code:`AlignedSentences` without providing word alignments. Those will be added
automatically behind the scenes. See `this example notebook`_ for more.

.. code-block:: bash

	sent_en = Sentence.from_text("I like eating cookies", "en")
	sent_nl = Sentence.from_text("Ik eet graag koekjes", "nl")

	# Word alignments do not need to be added on init:
	aligned = AlignedSentences(sent_en, sent_nl)

Keep in mind however that automatic alignment will never have the same quality as manual alignments. Use with caution!
I highly suggest reading `the paper`_ of Awesome Align to see whether it is a good pick for you.

.. _Awesome Align: https://github.com/neulab/awesome-align
.. _this fork: https://github.com/BramVanroy/awesome-align/tree/astred_compat
.. _this example notebook: examples/full-auto.ipynb
.. _the paper: https://arxiv.org/abs/2101.08231

License
-------
Licensed under Apache License Version 2.0. See the LICENSE file attached to this repository.

Citation
--------
Please cite our `papers`_ if you use this library.

.. _papers: CITATION
