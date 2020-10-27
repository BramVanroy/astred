Syntactic equivalence metrics
=============================

**NOTE: THIS IS A BETA VERSION** 

A new version is under construction that includes word-based metrics in addition to the currently implemented segment-based metrics.

Installation
------------

Requires Python 3.6 or higher.

This library relies on `stanza`_ to parse text into dependencies, which in turn depends on PyTorch. make sure that you
have a valid `PyTorch installation`_ prior to installing this library.

When PyTorch is installed, and you have cloned this library, you can run :code:`pip install` which will autmatically install
the required dependencies.

.. code-block:: bash

    git clone https://github.com/BramVanroy/astred.git
    cd astred
    pip install .


.. _stanza: https://github.com/stanfordnlp/stanza
.. _PyTorch installation: https://pytorch.org/get-started/locally/

Before trying to run any code, make sure that you have downloaded the correct stanza models. This can be done by running
the following commands in your Python interpreter:

.. code-block:: python

    import stanza
    # use the correct language code here
    # see https://stanfordnlp.github.io/stanza/models.html#available-ud-models
    stanza.download('en')


Examples
--------
You can find some examples in the examples folder.

:code:`monolingual.py`

Example showing how to use the GenericTree class and utilities to use tree edit distance on a monolingual task.

.. code-block:: bash

    # running from the root of this library
    python examples/monolingual.py "I like cookies" "I hate cookies" -t


License
-------
Licensed under Apache License Version 2.0. See the LICENSE file attached to this repository.
