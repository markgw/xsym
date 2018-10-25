Learned embedding analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:module:: langsim.modules.local_lm.embed_anal

+------------+-------------------------------------+
| Path       | langsim.modules.local_lm.embed_anal |
+------------+-------------------------------------+
| Executable | yes                                 |
+------------+-------------------------------------+

Various analyses thrown together for including things in a paper.

To simplify things, we assume for now that there are exactly two languages (vocabs, corpora).
We could generalize this later, but for now it makes the code much easier and we only do this
for the paper.


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

+----------+--------------------------------------------+
| Name     | Type(s)                                    |
+==========+============================================+
| analysis | :func:`~pimlico.datatypes.files.NamedFile` |
+----------+--------------------------------------------+
| pairs    | :func:`~pimlico.datatypes.files.NamedFile` |
+----------+--------------------------------------------+

Options
=======

+----------------+-----------------------------------------------------------------------------------------------------------------------------------+---------------------------------+
| Name           | Description                                                                                                                       | Type                            |
+================+===================================================================================================================================+=================================+
| oov            | If given, look for this special token in each vocabulary which represents OOVs. These are not filtered out, even if they are rare | string                          |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------+---------------------------------+
| lang_names     | (required) Comma-separated list of language IDs to use in output                                                                  | comma-separated list of strings |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------+---------------------------------+
| min_token_prop | Minimum frequency, as a proportion of tokens, that a character in the vocabulary must have to be shown in the charts              | float                           |
+----------------+-----------------------------------------------------------------------------------------------------------------------------------+---------------------------------+

