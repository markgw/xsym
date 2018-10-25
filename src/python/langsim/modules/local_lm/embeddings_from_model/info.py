from langsim.modules.local_lm.neural_sixgram2.info import NeuralSixgramKerasModel
from pimlico.core.modules.base import BaseModuleInfo
from pimlico.core.modules.options import comma_separated_strings
from pimlico.datatypes import MultipleInputs, Dictionary, NumpyArray
from pimlico.datatypes.embeddings import Embeddings


class ModuleInfo(BaseModuleInfo):
    """
    Simple module to extract the trained embeddings from a model stored by the
    training process, which can then be used in a generic way and output to
    generic formats.

    """
    module_type_name = "embeddings_from_model"
    module_inputs = [
        ("model", NeuralSixgramKerasModel),
        ("vocabs", MultipleInputs(Dictionary)),
        ("frequencies", MultipleInputs(NumpyArray)),
    ]
    module_outputs = [("embeddings", Embeddings)]
    module_options = {
        "lang_names": {
            "type": comma_separated_strings,
            "help": "Comma-separated list of language IDs to use in output",
            "required": True,
        },
    }
