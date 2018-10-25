Validation criterion correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:module:: langsim.modules.local_lm.val_crit_correlation

+------------+-----------------------------------------------+
| Path       | langsim.modules.local_lm.val_crit_correlation |
+------------+-----------------------------------------------+
| Executable | yes                                           |
+------------+-----------------------------------------------+

Compute correlation between the validation criterion and the retrieval
of known correspondences. See the paper for more details.


Inputs
======

+--------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Name   | Type(s)                                                                                                                                   |
+========+===========================================================================================================================================+
| models | :class:`list <pimlico.datatypes.base.MultipleInputs>` of :class:`KerasModelBuilderClass <pimlico.datatypes.keras.KerasModelBuilderClass>` |
+--------+-------------------------------------------------------------------------------------------------------------------------------------------+

Outputs
=======

+---------------+--------------------------------------------+
| Name          | Type(s)                                    |
+===============+============================================+
| metrics       | :func:`~pimlico.datatypes.files.NamedFile` |
+---------------+--------------------------------------------+
| final_metrics | :func:`~pimlico.datatypes.files.NamedFile` |
+---------------+--------------------------------------------+
| correlations  | :func:`~pimlico.datatypes.files.NamedFile` |
+---------------+--------------------------------------------+

