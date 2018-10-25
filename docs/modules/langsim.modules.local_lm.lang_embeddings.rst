Language-specific embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:module:: langsim.modules.local_lm.lang_embeddings

+------------+------------------------------------------+
| Path       | langsim.modules.local_lm.lang_embeddings |
+------------+------------------------------------------+
| Executable | yes                                      |
+------------+------------------------------------------+

Separate out the embeddings belonging to the two languages, identified by
prefixes on the words.

It's assumed that all embeddings for language "X" have words of the form
"X:word".

This only works currently for cases where there are exactly two languages.


Inputs
======

+------------+---------------------------------------------------------------+
| Name       | Type(s)                                                       |
+============+===============================================================+
| embeddings | :class:`Embeddings <pimlico.datatypes.embeddings.Embeddings>` |
+------------+---------------------------------------------------------------+

Outputs
=======

+------------------+---------------------------------------------------+
| Name             | Type(s)                                           |
+==================+===================================================+
| lang1_embeddings | :class:`~pimlico.datatypes.embeddings.Embeddings` |
+------------------+---------------------------------------------------+
| lang2_embeddings | :class:`~pimlico.datatypes.embeddings.Embeddings` |
+------------------+---------------------------------------------------+

Options
=======

+-------+------------------------------------------------------------------------------------------------------------+--------+
| Name  | Description                                                                                                | Type   |
+=======+============================================================================================================+========+
| lang1 | Prefixes for language 1. If not given, language 1 is taken to be whichever appears first in the vocabulary | string |
+-------+------------------------------------------------------------------------------------------------------------+--------+

