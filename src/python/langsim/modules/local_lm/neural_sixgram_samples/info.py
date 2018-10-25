"""
Prepare positive samples for neural sixgram training data.

Instead of doing random shuffling, etc, on the fly while training, which
takes quite a lot of time, we do it once before and just iterate over
the result at training time.

The output is then used by :mod:`~langsim.modules.local_lm.neural_sixgram2` to
train the Xsym model.

"""
from langsim.datatypes.neural_sixgram import NeuralSixgramTrainingData

from pimlico.core.modules.base import BaseModuleInfo
from pimlico.core.modules.options import str_to_bool
from pimlico.datatypes import Dictionary, NumpyArray
from pimlico.datatypes.base import MultipleInputs
from pimlico.datatypes.ints import IntegerListsDocumentType
from pimlico.datatypes.tar import TarredCorpusType


class ModuleInfo(BaseModuleInfo):
    module_type_name = "neural_sixgram_samples_prep"
    module_inputs = [
        ("vocabs", MultipleInputs(Dictionary)),
        ("corpora", MultipleInputs(TarredCorpusType(IntegerListsDocumentType))),
        ("frequencies", MultipleInputs(NumpyArray)),
    ]
    module_outputs = [("samples", NeuralSixgramTrainingData)]
    module_options = {
        "cross_sentences": {
            "help": "By default, the sliding window passed over the corpus stops at the end of a sentence (or "
                    "whatever sequence division is in the input data) and starts again at the start of the next. "
                    "Instead, join all sequences within a document into one long sequence and pass the sliding "
                    "window over that",
            "type": str_to_bool,
        },
        "corpus_offset": {
            "help": "To avoid training on parallel data, in the case where the input corpora happen to be parallel, "
                    "jump forward in the second corpus by this number of utterances, putting the skipping utterances "
                    "at the end instead. Default: 10k utterances",
            "type": int,
            "default": 10000,
        },
        "oov": {
            "help": "If given, use this special token in each vocabulary to represent OOVs. Otherwise, they are "
                    "represented by an index added at the end of each vocabulary's indices",
        },
        "shuffle_window": {
            "help": "We simulate shuffling the data by reading samples into a buffer and taking them randomly from "
                    "there. This is the size of that buffer. A higher number shuffles more, but makes data "
                    "preparation slower",
            "type": int,
            "default": 1000,
        },
    }
