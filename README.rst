Syntactic equivalence metrics
=============================

Examples, documentation, and tests to be added. You can already use :code:`examples/add_features_tprdb.py`, though. 
Use :code:`python examples/add_features_tprdb.py -h` to get started.

Installation
------------

Requires Python 3.7 or higher.

This library relies on `stanza`_ to parse text into dependencies, which in turn depends on PyTorch. make sure that you
have a valid `PyTorch installation`_ prior to installing this library.

When PyTorch is installed, and you have cloned this library, you can run :code:`pip install .` which will autmatically install
the required dependencies.

.. code-block:: bash

    git clone https://github.com/BramVanroy/astred.git
    cd astred
    pip install .


.. _stanza: https://github.com/stanfordnlp/stanza
.. _PyTorch installation: https://pytorch.org/get-started/locally/


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
automatically behind the scenes.

.. code-block:: bash

	sent_en = Sentence.from_text("I like eating cookies", "en")
	sent_nl = Sentence.from_text("Ik eet graag koekjes", "nl")

	# Word alignments do not need to be added on init:
	aligned = AlignedSentences(sent_en, sent_nl)

Keep in mind however that automatic alignment will never have the same quality as manual alignments. Use with caution!
I highly suggest reading `the paper`_ of Awesome Align to see whether it is a good pick for you.

.. _Awesome Align: https://github.com/neulab/awesome-align
.. _this fork: https://github.com/BramVanroy/awesome-align/tree/astred_compat
.. _the paper: https://arxiv.org/abs/2101.08231

License
-------
Licensed under Apache License Version 2.0. See the LICENSE file attached to this repository.
