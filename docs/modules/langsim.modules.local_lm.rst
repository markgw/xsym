Symbol embedding methods
~~~~~~~~~~~~~~~~~~~~~~~~


.. py:module:: langsim.modules.local_lm


Neural network-based symbol (phoneme/character) representation learning techniques
that work by applying the distributional hypothesis cross-lingually and simultaneously learning
representations for both languages.

Some ways of doing this work better than others. The best method appears to be
:mod:`~langsim.modules.local_lm.neural_sixgram2`, which is now the only one implemented
here. It takes into account a relatively broad context of the symbols, and seems to be
fairly robust across language pairs.



.. toctree::
   :maxdepth: 2
   :titlesonly:

   langsim.modules.local_lm.corruption_results
   langsim.modules.local_lm.embed_anal
   langsim.modules.local_lm.embeddings_from_model
   langsim.modules.local_lm.lang_embeddings
   langsim.modules.local_lm.neural_sixgram
   langsim.modules.local_lm.neural_sixgram2
   langsim.modules.local_lm.neural_sixgram_samples
   langsim.modules.local_lm.plot
   langsim.modules.local_lm.store_tsv
   langsim.modules.local_lm.val_crit_correlation
