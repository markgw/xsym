"""Symbol embedding methods

Neural network-based symbol (phoneme/character) representation learning techniques
that work by applying the distributional hypothesis cross-lingually and simultaneously learning
representations for both languages.

Some ways of doing this work better than others. The best method appears to be
:mod:`~langsim.modules.local_lm.neural_sixgram2`, which is now the only one implemented
here. It takes into account a relatively broad context of the symbols, and seems to be
fairly robust across language pairs.

"""