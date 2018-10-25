"""
A special kind of six-gram model that combines 1-3 characters on the left with 1-3 characters on the right
to learn unigram, bigram and trigram representations.

This is one of the most successful representation learning methods among those here. It's also very robust
across language pairs and different sizes of dataset. It's therefore the model that I've opted to use in
subsequent work that uses the learned representations.

"""
from pimlico.core.dependencies.python import PythonPackageOnPip, theano_dependency, sklearn_dependency
from pimlico.core.modules.base import BaseModuleInfo
from pimlico.core.modules.options import comma_separated_list, str_to_bool
from pimlico.datatypes.arrays import NumpyArray
from pimlico.datatypes.base import MultipleInputs
from pimlico.datatypes.dictionary import Dictionary
from pimlico.datatypes.files import FileInput
from pimlico.datatypes.ints import IntegerListsDocumentType
from pimlico.datatypes.keras import KerasModelBuilderClass
from pimlico.datatypes.tar import TarredCorpusType
from pimlico.modules.visualization import matplotlib_dependency


def context_weights(text):
    """
    Option type for context weight schedule parameters

    """
    weight_list = [x.strip() for x in text.split(",")]

    if len(weight_list) < 6:
        raise ValueError("context weight list must have at least 6 values, specifying the initial weights")
    if (len(weight_list) - 6) % 7 != 0:
        raise ValueError("after initial 6 weights, context weight list must be made up of a multiple of 7 weights, "
                         "each giving an epoch number / sample number and 6 weights")

    epoch_weights = [(0., [float(x) for x in weight_list[:6]])] + \
                    [(float(x[0]), [float(w) for w in x[1:]]) for x in zip(*([iter(weight_list[6:])]*7))]
    return epoch_weights


class ModuleInfo(BaseModuleInfo):
    module_type_name = "neural_sixgram_trainer"
    module_readable_name = "Neural sixgram (Xsym) trainer, v1"
    module_inputs = [
        ("vocabs", MultipleInputs(Dictionary)),
        ("corpora", MultipleInputs(TarredCorpusType(IntegerListsDocumentType))),
        ("frequencies", MultipleInputs(NumpyArray))
    ]
    module_optional_inputs = [
        ("mapped_pairs", FileInput()),
    ]
    module_outputs = [("model", KerasModelBuilderClass)]
    module_options = {
        "embedding_size": {
            "help": "Number of dimensions in the hidden representation. Default: 200",
            "type": int,
            "default": 100,
        },
        "epochs": {
            "help": "Max number of training epochs. Default: 5",
            "type": int,
            "default": 5,
        },
        "store_all": {
            "help": "Store updated representations from every epoch, even if the validation loss goes up. The default "
                    "behaviour is to only store the parameters with best validation loss, but for these purposes "
                    "we probably want to set this to T most of the time. (Defaults to F for backwards compatibility)",
            "type": str_to_bool,
            "default": False,
        },
        "batch": {
            "help": "Training batch size. Default: 100",
            "type": int,
            "default": 100,
        },
        "cross_sentences": {
            "help": "By default, the sliding window passed over the corpus stops at the end of a sentence (or "
                    "whatever sequence division is in the input data) and starts again at the start of the next. "
                    "Instead, join all sequences within a document into one long sequence and pass the sliding "
                    "window over that",
            "type": str_to_bool,
        },
        "dropout": {
            "help": "Dropout to apply to embeddings during training. Default: 0.3",
            "type": float,
            "default": 0.3,
        },
        "l2_reg": {
            "help": "L2 regularization to apply to all layers' weights. Default: 0.",
            "type": float,
            "default": 0.,
        },
        "composition_dropout": {
            "help": "Dropout to apply to composed representation during training. Default: same as dropout",
            "type": float,
        },
        "validation": {
            "help": "Number of samples to hold out as a validation set for training. Simply taken "
                    "from the start of the corpus. Rounded to the nearest number of batches",
            "type": int,
            "default": 1000,
        },
        "composition2_layers": {
            "help": "Number and size of layers to use to combine pairs of characters, given as a list of integers. "
                    "The final layer must be the same size as the embeddings, so is not included in this list",
            "type": comma_separated_list(int),
            "default": [100],
        },
        "composition3_layers": {
            "help": "Number and size of layers to use to combine triples of characters, given as a list of integers. "
                    "The final layer must be the same size as the embeddings, so is not included in this list",
            "type": comma_separated_list(int),
            "default": [100],
        },
        "predictor_layers": {
            "help": "Number and size of layers to use to take a pair of vectors and say whether they belong beside "
                    "each other. Given as a list of integers. Doesn't include the final projection to a single "
                    "score",
            "type": comma_separated_list(int),
            "default": [100, 50],
        },
        "embedding_activation": {
            "help": "Activation function to apply to the learned embeddings before they're used, and also to every "
                    "projection into the embedding space (the final layers of compositions). By default, 'linear' is "
                    "used, i.e. normal embeddings with no activation and a linear layer at the end of the composition "
                    "functions. Choose any Keras named activation",
            "default": "linear",
        },
        "limit_training": {
            "help": "Limit training to this many batches. Default: no limit",
            "type": int,
        },
        "corpus_offset": {
            "help": "To avoid training on parallel data, in the case where the input corpora happen to be parallel, "
                    "jump forward in the second corpus by this number of utterances, putting the skipping utterances "
                    "at the end instead. Default: 10k utterances",
            "type": int,
            "default": 10000,
        },
        "plot_freq": {
            "help": "Output plots to the output directory while training is in progress. This slows down training if "
                    "it's done very often. Specify how many batches to wait between each plot. Fewer means you get "
                    "a finer grained picture of the training process, more means training goes faster. 0 (default) "
                    "turns off plotting",
            "type": int,
            "default": 0,
        },
        "sim_freq": {
            "help": "How often (in batches) to compute the similarity of overlapping phonemes between the languages. "
                    "-1 (default) means never, 0 means once at the start of each epoch",
            "type": int,
            "default": -1,
        },
        "context_weights": {
            "help": "Coefficients that specify the relative frequencies with which each of the different lengths "
                    "of contexts (1, 2 and 3) will be used in training examples. For each sample, a pair context "
                    "lengths is selected at random. Six coefficients specify the weights given to (1,1), (1,2), (1,3), "
                    "(2,2), (2,3) and (3,3). The opposite orderings have the same probability. "
                    "By default, they are uniformly sampled ('1,1,1,1,1,1'), but you may adjust their "
                    "relative frequencies to put more weight on some lengths than others. "
                    "The first 6 values are the starting weights. After that, you may specify sets of 7 values: "
                    "num_epochs, weight1, weight2, .... The weights at any point will transition smoothly (linearly) "
                    "from the previous 6-tuple to the next, arriving at the epoch number given (i.e. 1=start "
                    "of epoch 1 / end of first epoch). You may use float epoch numbers, e.g. 0.5",
            "type": context_weights,
            "default": [(0., 1., 1., 1., 1., 1., 1.)],
        },
        "oov": {
            "help": "If given, use this special token in each vocabulary to represent OOVs. Otherwise, they are "
                    "represented by an index added at the end of each vocabulary's indices",
        },
        "unit_norm": {
            "help": "If true, enforce a unit norm constraint on the learned embeddings. Default: false",
            "type": str_to_bool,
        },
        "word_internal": {
            "help": "Only train model on word-internal sequences. Word boundaries will be included, but no "
                    "sequences spanning over word boundaries",
            "type": str_to_bool,
        },
        "word_boundary": {
            "help": "If using word_internal, use this character (which must be in the vocabulary) to split "
                    "words. Default: space",
            "type": unicode,
            "default": u" ",
        }
    }

    def get_software_dependencies(self):
        return super(ModuleInfo, self).get_software_dependencies() + [
            # Require >= 1.0.6 (but can't yet specify that in dep -- future Pimlico feature)
            PythonPackageOnPip("keras", dependencies=[theano_dependency]),
            PythonPackageOnPip("h5py"),
            matplotlib_dependency,
            sklearn_dependency,
        ]
