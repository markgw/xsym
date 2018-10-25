"""
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

"""
from pimlico.core.modules.map import DocumentMapModuleInfo

from pimlico.datatypes.arrays import NumpyArray
from pimlico.datatypes.dictionary import Dictionary
from pimlico.datatypes.files import NamedFile
from pimlico.datatypes.ints import IntegerListsDocumentType, IntegerListsDocumentCorpusWriter
from pimlico.datatypes.tar import TarredCorpusType, tarred_corpus_with_data_point_type


class ModuleInfo(DocumentMapModuleInfo):
    module_type_name = "corrupt"
    module_readable_name = "Corrupt text"
    module_inputs = [
        ("corpus", TarredCorpusType(IntegerListsDocumentType)),
        ("vocab", Dictionary),
        ("frequencies", NumpyArray),
    ]
    module_outputs = [
        ("corpus", tarred_corpus_with_data_point_type(IntegerListsDocumentType)),
        ("vocab", Dictionary),
        ("mappings", NamedFile("mappings.json")),
        ("close_pairs", NamedFile("close_pairs.json")),
        ("corruption_params", NamedFile("corruption_params.json")),
    ]
    module_options = {
        "char_subst_prop": {
            "help": "Proportion of characters to sample for random substitution. Default: 0",
            "type": float,
            "default": 0.,
        },
        "char_map_prop": {
            "help": "Proportion of character types in input vocab to apply a random mapping to "
                    "another character to. Default: 0",
            "type": float,
            "default": 0.,
        },
        "char_split_prop": {
            "help": "Proportion of character types in input vocab to apply splitting to. "
                    "Default: 0",
            "type": float,
            "default": 0.,
        },
    }

    def get_writer(self, output_name, output_dir, append=False):
        return IntegerListsDocumentCorpusWriter(output_dir, append=append)
