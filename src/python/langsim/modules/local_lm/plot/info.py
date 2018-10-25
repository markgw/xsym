from pimlico.core.dependencies.python import keras_tensorflow_dependency, sklearn_dependency
from pimlico.core.modules.base import BaseModuleInfo
from pimlico.core.modules.options import choose_from_list, comma_separated_strings
from pimlico.datatypes import NumpyArray
from pimlico.datatypes.base import PimlicoDatatype, MultipleInputs
from pimlico.datatypes.dictionary import Dictionary
from pimlico.datatypes.ints import IntegerListsDocumentType
from pimlico.datatypes.keras import KerasModelBuilderClass
from pimlico.datatypes.tar import TarredCorpusType
from pimlico.modules.visualization import matplotlib_dependency


class ModuleInfo(BaseModuleInfo):
    """
    Produces various plots to help with analysing the results of training a
    neural_sixgram model.

    Note that this used to be designed to support other model types, but I'm now cleaning
    up and only supporting neural_sixgram2.

    """
    module_type_name = "neural_ngram_plots"
    module_readable_name = "Plots of neural sixgram models"
    module_inputs = [
        ("model", KerasModelBuilderClass),
        ("vocabs", MultipleInputs(Dictionary)),
        ("corpora", MultipleInputs(TarredCorpusType(IntegerListsDocumentType))),
        ("frequencies", MultipleInputs(NumpyArray)),
    ]
    module_outputs = [("output", PimlicoDatatype)]
    module_options = {
        "distance": {
            "type": choose_from_list(["eucl", "dot", "cos", "man", "sig_kern"]),
            "help": "Distance metric to use",
            "default": "eucl",
        },
        "min_token_prop": {
            "type": float,
            "help": "Minimum frequency, as a proportion of tokens, that a character in the vocabulary must have "
                    "to be shown in the charts",
            "default": 0.01,
        },
        "num_pairs": {
            "type": int,
            "help": "Number of most frequent character pairs to show on the chart (passed through the composition "
                    "function to get their representation)",
            "default": 50,
        },
        "lang_names": {
            "type": comma_separated_strings,
            "help": "Comma-separated list of language IDs to use in output",
            "required": True,
        },
    }

    def get_software_dependencies(self):
        return super(ModuleInfo, self).get_software_dependencies() + [
            keras_tensorflow_dependency,
            matplotlib_dependency,
            sklearn_dependency,
        ]
