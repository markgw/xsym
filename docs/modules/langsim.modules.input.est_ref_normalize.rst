Est Ref normalization
~~~~~~~~~~~~~~~~~~~~~

.. py:module:: langsim.modules.input.est_ref_normalize

+------------+-----------------------------------------+
| Path       | langsim.modules.input.est_ref_normalize |
+------------+-----------------------------------------+
| Executable | yes                                     |
+------------+-----------------------------------------+

Special normalization routine for Estonian Reference Corpus.

Splits up sentences into separate lines. This is easy to do, since the
corpus puts a double space between sentences. There are also double
spaces in other places, so we only split on double spaces after punctuation.
Other double spaces are removed.

We also lower-case the whole corpus.


Inputs
======

+--------+--------------------------------+
| Name   | Type(s)                        |
+========+================================+
| corpus | TarredCorpus<TextDocumentType> |
+--------+--------------------------------+

Outputs
=======

+--------+-----------------------------------------------------------------+
| Name   | Type(s)                                                         |
+========+=================================================================+
| corpus | :class:`~pimlico.datatypes.tar.RawTextDocumentTypeTarredCorpus` |
+--------+-----------------------------------------------------------------+

Options
=======

+-------+-------------------------------------------------------------------------------------------+------+
| Name  | Description                                                                               | Type |
+=======+===========================================================================================+======+
| forum | Set to T for processing the forum data, which is slightly different to the newspaper data | bool |
+-------+-------------------------------------------------------------------------------------------+------+

