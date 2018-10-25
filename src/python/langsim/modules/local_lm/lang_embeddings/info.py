from pimlico.core.modules.base import BaseModuleInfo
from pimlico.datatypes.embeddings import Embeddings


class ModuleInfo(BaseModuleInfo):
    """
    Separate out the embeddings belonging to the two languages, identified by
    prefixes on the words.

    It's assumed that all embeddings for language "X" have words of the form
    "X:word".

    This only works currently for cases where there are exactly two languages.

    """
    module_type_name = "lang_embeddings"
    module_readable_name = "Language-specific embeddings"
    module_inputs = [("embeddings", Embeddings)]
    module_outputs = [
        ("lang1_embeddings", Embeddings),
        ("lang2_embeddings", Embeddings),
    ]
    module_options = {
        "lang1": {
            "help": "Prefixes for language 1. If not given, language 1 is taken to be "
                    "whichever appears first in the vocabulary",
        },
    }

