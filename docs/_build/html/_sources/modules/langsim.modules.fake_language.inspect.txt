Inspect corrupted text
~~~~~~~~~~~~~~~~~~~~~~

.. py:module:: langsim.modules.fake_language.inspect

+------------+---------------------------------------+
| Path       | langsim.modules.fake_language.inspect |
+------------+---------------------------------------+
| Executable | yes                                   |
+------------+---------------------------------------+

Display corrupted and uncorrupted texts alongside one another

For observing the output of the corruption process, which otherwise is just a load of
integer-encoded data.


Inputs
======

+---------+---------------------------------------------------------------+
| Name    | Type(s)                                                       |
+=========+===============================================================+
| corpus1 | TarredCorpus<IntegerListsDocumentType>                        |
+---------+---------------------------------------------------------------+
| vocab1  | :class:`Dictionary <pimlico.datatypes.dictionary.Dictionary>` |
+---------+---------------------------------------------------------------+
| corpus2 | TarredCorpus<IntegerListsDocumentType>                        |
+---------+---------------------------------------------------------------+
| vocab2  | :class:`Dictionary <pimlico.datatypes.dictionary.Dictionary>` |
+---------+---------------------------------------------------------------+

Outputs
=======

+---------+-----------------------------------------------------------------+
| Name    | Type(s)                                                         |
+=========+=================================================================+
| inspect | :class:`~pimlico.datatypes.tar.RawTextDocumentTypeTarredCorpus` |
+---------+-----------------------------------------------------------------+

