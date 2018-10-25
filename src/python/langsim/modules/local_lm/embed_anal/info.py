from langsim.modules.local_lm.neural_sixgram2.info import NeuralSixgramKerasModel
from pimlico.core.dependencies.python import keras_tensorflow_dependency, sklearn_dependency
from pimlico.core.modules.base import BaseModuleInfo
from pimlico.core.modules.options import comma_separated_strings
from pimlico.datatypes import NumpyArray
from pimlico.datatypes.base import MultipleInputs
from pimlico.datatypes.dictionary import Dictionary
from pimlico.datatypes.files import NamedFile


class ModuleInfo(BaseModuleInfo):
    """
    Various analyses thrown together for including things in a paper.

    To simplify things, we assume for now that there are exactly two languages (vocabs, corpora).
    We could generalize this later, but for now it makes the code much easier and we only do this
    for the paper.

    """
    module_type_name = "embedding_analysis"
    module_readable_name = "Learned embedding analysis"
    module_inputs = [
        ("model", NeuralSixgramKerasModel),
        ("vocabs", MultipleInputs(Dictionary)),
        ("frequencies", MultipleInputs(NumpyArray))
    ]
    module_outputs = [
        ("analysis", NamedFile("analysis.txt")),
        ("pairs", NamedFile("pairs.txt")),
    ]
    module_options = {
        "lang_names": {
            "type": comma_separated_strings,
            "help": "Comma-separated list of language IDs to use in output",
            "required": True,
        },
        "min_token_prop": {
            "type": float,
            "help": "Minimum frequency, as a proportion of tokens, that a character in the vocabulary must have "
                    "to be shown in the charts",
            "default": 0.01,
        },
        "oov": {
            "help": "If given, look for this special token in each vocabulary which represents OOVs. These are "
                    "not filtered out, even if they are rare",
        },
    }

    def get_software_dependencies(self):
        return super(ModuleInfo, self).get_software_dependencies() + [
            keras_tensorflow_dependency,
            sklearn_dependency,
        ]
