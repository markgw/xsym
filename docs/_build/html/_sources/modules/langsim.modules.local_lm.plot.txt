Plots of neural sixgram models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:module:: langsim.modules.local_lm.plot

+------------+-------------------------------+
| Path       | langsim.modules.local_lm.plot |
+------------+-------------------------------+
| Executable | yes                           |
+------------+-------------------------------+

Produces various plots to help with analysing the results of training a
neural_sixgram model.

Note that this used to be designed to support other model types, but I'm now cleaning
up and only supporting neural_sixgram2.


Inputs
======

+-------------+------------------------------------------------------------------------------------------------------------------------+
| Name        | Type(s)                                                                                                                |
+=============+========================================================================================================================+
| model       | :class:`KerasModelBuilderClass <pimlico.datatypes.keras.KerasModelBuilderClass>`                                       |
+-------------+------------------------------------------------------------------------------------------------------------------------+
| vocabs      | :class:`list <pimlico.datatypes.base.MultipleInputs>` of :class:`Dictionary <pimlico.datatypes.dictionary.Dictionary>` |
+-------------+------------------------------------------------------------------------------------------------------------------------+
| corpora     | :class:`list <pimlico.datatypes.base.MultipleInputs>` of TarredCorpus<IntegerListsDocumentType>                        |
+-------------+------------------------------------------------------------------------------------------------------------------------+
| frequencies | :class:`list <pimlico.datatypes.base.MultipleInputs>` of :class:`NumpyArray <pimlico.datatypes.arrays.NumpyArray>`     |
+-------------+------------------------------------------------------------------------------------------------------------------------+

Outputs
=======

+--------+--------------------------------------------------+
| Name   | Type(s)                                          |
+========+==================================================+
| output | :class:`~pimlico.datatypes.base.PimlicoDatatype` |
+--------+--------------------------------------------------+

Options
=======

+----------------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------+
| Name           | Description                                                                                                                        | Type                                      |
+================+====================================================================================================================================+===========================================+
| distance       | Distance metric to use                                                                                                             | 'eucl', 'dot', 'cos', 'man' or 'sig_kern' |
+----------------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------+
| num_pairs      | Number of most frequent character pairs to show on the chart (passed through the composition function to get their representation) | int                                       |
+----------------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------+
| min_token_prop | Minimum frequency, as a proportion of tokens, that a character in the vocabulary must have to be shown in the charts               | float                                     |
+----------------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------+
| lang_names     | (required) Comma-separated list of language IDs to use in output                                                                   | comma-separated list of strings           |
+----------------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------+

