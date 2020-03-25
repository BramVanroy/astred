Syntactic equivalence metrics
=============================

**NOTE: THIS IS A BETA VERSION**

Installation
------------

Requires Python 3.6 or higher.

This library relies on stanza to parse text into dependencies, which in turn depends on PyTorch. make sure that you
have a valid `PyTorch installation`_ prior to installing this library.

When PyTorch is installed, you can run which will autmatically install the required dependencies.

.. code-block:: bash

    pip install astred


.. _PyTorch installation: https://pytorch.org/get-started/locally/

Examples
--------
You can find some examples in the examples folder.

:code:`monolingual.py`

Example showing how to use the GenericTree class and utilities to use tree edit distance on a monolingual task.

.. code-block:: bash

    python examples/monolingual.py "I like cookies" "I hate cookies" -t


License
-------
Licensed under Apache License Version 2.0. See the LICENSE file attached to this repository.
