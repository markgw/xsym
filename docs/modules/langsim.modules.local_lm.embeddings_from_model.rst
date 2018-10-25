Embeddings from model
~~~~~~~~~~~~~~~~~~~~~

.. py:module:: langsim.modules.local_lm.embeddings_from_model

+------------+------------------------------------------------+
| Path       | langsim.modules.local_lm.embeddings_from_model |
+------------+------------------------------------------------+
| Executable | yes                                            |
+------------+------------------------------------------------+

Simple module to extract the trained embeddings from a model stored by the
training process, which can then be used in a generic way and output to
generic formats.


Inputs
======

+-------------+------------------------------------------------------------------------------------------------------------------------+
| Name        | Type(s)                                                                                                                |
+=============+========================================================================================================================+
| model       | :class:`NeuralSixgramKerasModel <langsim.modules.local_lm.neural_sixgram2.info.NeuralSixgramKerasModel>`               |
+-------------+------------------------------------------------------------------------------------------------------------------------+
| vocabs      | :class:`list <pimlico.datatypes.base.MultipleInputs>` of :class:`Dictionary <pimlico.datatypes.dictionary.Dictionary>` |
+-------------+------------------------------------------------------------------------------------------------------------------------+
| frequencies | :class:`list <pimlico.datatypes.base.MultipleInputs>` of :class:`NumpyArray <pimlico.datatypes.arrays.NumpyArray>`     |
+-------------+------------------------------------------------------------------------------------------------------------------------+

Outputs
=======

+------------+---------------------------------------------------+
| Name       | Type(s)                                           |
+============+===================================================+
| embeddings | :class:`~pimlico.datatypes.embeddings.Embeddings` |
+------------+---------------------------------------------------+

Options
=======

+------------+------------------------------------------------------------------+---------------------------------+
| Name       | Description                                                      | Type                            |
+============+==================================================================+=================================+
| lang_names | (required) Comma-separated list of language IDs to use in output | comma-separated list of strings |
+------------+------------------------------------------------------------------+---------------------------------+

