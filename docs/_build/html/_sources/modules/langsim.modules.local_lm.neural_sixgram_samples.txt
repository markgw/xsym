Neural sixgram samples prep
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:module:: langsim.modules.local_lm.neural_sixgram_samples

+------------+-------------------------------------------------+
| Path       | langsim.modules.local_lm.neural_sixgram_samples |
+------------+-------------------------------------------------+
| Executable | yes                                             |
+------------+-------------------------------------------------+

Prepare positive samples for neural sixgram training data.

Instead of doing random shuffling, etc, on the fly while training, which
takes quite a lot of time, we do it once before and just iterate over
the result at training time.

The output is then used by :mod:`~langsim.modules.local_lm.neural_sixgram2` to
train the Xsym model.


Inputs
======

+-------------+------------------------------------------------------------------------------------------------------------------------+
| Name        | Type(s)                                                                                                                |
+=============+========================================================================================================================+
| vocabs      | :class:`list <pimlico.datatypes.base.MultipleInputs>` of :class:`Dictionary <pimlico.datatypes.dictionary.Dictionary>` |
+-------------+------------------------------------------------------------------------------------------------------------------------+
| corpora     | :class:`list <pimlico.datatypes.base.MultipleInputs>` of TarredCorpus<IntegerListsDocumentType>                        |
+-------------+------------------------------------------------------------------------------------------------------------------------+
| frequencies | :class:`list <pimlico.datatypes.base.MultipleInputs>` of :class:`NumpyArray <pimlico.datatypes.arrays.NumpyArray>`     |
+-------------+------------------------------------------------------------------------------------------------------------------------+

Outputs
=======

+---------+----------------------------------------------------------------------+
| Name    | Type(s)                                                              |
+=========+======================================================================+
| samples | :class:`~langsim.datatypes.neural_sixgram.NeuralSixgramTrainingData` |
+---------+----------------------------------------------------------------------+

Options
=======

+-----------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+
| Name            | Description                                                                                                                                                                                                                                                                                     | Type   |
+=================+=================================================================================================================================================================================================================================================================================================+========+
| cross_sentences | By default, the sliding window passed over the corpus stops at the end of a sentence (or whatever sequence division is in the input data) and starts again at the start of the next. Instead, join all sequences within a document into one long sequence and pass the sliding window over that | bool   |
+-----------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+
| oov             | If given, use this special token in each vocabulary to represent OOVs. Otherwise, they are represented by an index added at the end of each vocabulary's indices                                                                                                                                | string |
+-----------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+
| shuffle_window  | We simulate shuffling the data by reading samples into a buffer and taking them randomly from there. This is the size of that buffer. A higher number shuffles more, but makes data preparation slower                                                                                          | int    |
+-----------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+
| corpus_offset   | To avoid training on parallel data, in the case where the input corpora happen to be parallel, jump forward in the second corpus by this number of utterances, putting the skipping utterances at the end instead. Default: 10k utterances                                                      | int    |
+-----------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+--------+

