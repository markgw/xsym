Corrupt text
~~~~~~~~~~~~

.. py:module:: langsim.modules.fake_language.corrupt

+------------+---------------------------------------+
| Path       | langsim.modules.fake_language.corrupt |
+------------+---------------------------------------+
| Executable | yes                                   |
+------------+---------------------------------------+

Introduce random noise into a corpus.

The input corpus is expected to be character-level encoded integer indexed text.
(You could also run it on word-level encoded data, but the results might be odd.)

Produces a new corpus with a new character vocabulary, which might not be identical
to the input vocabulary, depending on the options. E.g. some characters might be
removed or added.

If a token called 'OOV' is found in the vocabulary, it will never be subject to
a mapping or mapped to.

Types of noise, with corresponding parameters:

 - Random character substitutions: randomly sample a given proportion of characters
   and choose a character at random from the unigram distribution of the input
   corpus to replace each with

    * ``char_subst_prop``: proportion of characters (tokens) to sample for substitution.
      Use 0 to disable this corruption

 - Systematic character mapping: perform a systematic substitution throughout
   the corpus of a particular character A (randomly chosen from input vocab) for
   another B (randomly chosen from output vocab). This means that the resulting Bs
   are indistinguishable from those that were Bs in the input. A is removed from
   the output vocab, since it is never used now. When multiple mappings are chosen,
   it is not checked that they have different Bs.

   A number of characters is chosen
   using frequencies so that the expected proportion of tokens affected is at least
   the given parameter. Since the resulting expected proportion of tokens may be
   higher due to the sampling of characters, the actual expected proportion is
   output among the corruption parameters as ``actual_char_subst_prop``.

    * ``char_map_prop``: proportion of characters (types) in input vocab to apply a
      mapping to. Use 0 to disable this corruption

 - Split characters: choose a set of characters. For each A invent a new character B
   and map half of its occurrences to B, leaving half as they were. Each of these
   results in adding a brand new unicode character to the output vocab

   As with ``char_map_prop``, a number of characters is chosen
   using frequencies so that the expected proportion of tokens affected is at least
   the given parameter. Since the resulting expected proportion of tokens may be
   higher due to the sampling of characters, the actual expected proportion is
   output among the corruption parameters as ``actual_char_split_prop``.

    * ``char_split_prop``: proportion of characters (types) to apply this splitting to


Inputs
======

+-------------+---------------------------------------------------------------+
| Name        | Type(s)                                                       |
+=============+===============================================================+
| corpus      | TarredCorpus<IntegerListsDocumentType>                        |
+-------------+---------------------------------------------------------------+
| vocab       | :class:`Dictionary <pimlico.datatypes.dictionary.Dictionary>` |
+-------------+---------------------------------------------------------------+
| frequencies | :class:`NumpyArray <pimlico.datatypes.arrays.NumpyArray>`     |
+-------------+---------------------------------------------------------------+

Outputs
=======

+-------------------+----------------------------------------------------------------------+
| Name              | Type(s)                                                              |
+===================+======================================================================+
| corpus            | :class:`~pimlico.datatypes.tar.IntegerListsDocumentTypeTarredCorpus` |
+-------------------+----------------------------------------------------------------------+
| vocab             | :class:`~pimlico.datatypes.dictionary.Dictionary`                    |
+-------------------+----------------------------------------------------------------------+
| mappings          | :func:`~pimlico.datatypes.files.NamedFile`                           |
+-------------------+----------------------------------------------------------------------+
| close_pairs       | :func:`~pimlico.datatypes.files.NamedFile`                           |
+-------------------+----------------------------------------------------------------------+
| corruption_params | :func:`~pimlico.datatypes.files.NamedFile`                           |
+-------------------+----------------------------------------------------------------------+

Options
=======

+-----------------+------------------------------------------------------------------------------------------------------------+-------+
| Name            | Description                                                                                                | Type  |
+=================+============================================================================================================+=======+
| char_map_prop   | Proportion of character types in input vocab to apply a random mapping to another character to. Default: 0 | float |
+-----------------+------------------------------------------------------------------------------------------------------------+-------+
| char_split_prop | Proportion of character types in input vocab to apply splitting to. Default: 0                             | float |
+-----------------+------------------------------------------------------------------------------------------------------------+-------+
| char_subst_prop | Proportion of characters to sample for random substitution. Default: 0                                     | float |
+-----------------+------------------------------------------------------------------------------------------------------------+-------+

