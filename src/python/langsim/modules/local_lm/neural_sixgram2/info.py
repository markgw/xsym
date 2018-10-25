"""
A special kind of six-gram model that combines 1-3 characters on the left with 1-3 characters on the right
to learn unigram, bigram and trigram representations.

This is one of the most successful representation learning methods among those here. It's also very robust
across language pairs and different sizes of dataset. It's therefore the model that I've opted to use in
subsequent work that uses the learned representations.

This is a new version of the code for the model training. It will include random restarts and
early stopping using the new validation criterion. I've moved to a new version so that I can get rid
of old things from experiments with different types of models and clean up the code. The old version
was used to measure the validity of the validation criterion. From now on, I'm using the
validation criterion in earnest.

I'm now changing all default parameters to those use in the submitted paper and removing some parameters
for features that no longer need to be parameterized.

.. note::

   A note on using GPUs

   We use Keras to train. If you're using the tensorflow backend (which is what is assumed
   by this module's dependencies) and you want to use GPUs, you'll need to install the GPU
   version of Tensorflow, not just "tensorflow", which will be installed during dependency
   resolution. Try this (changing the virtualenv directory name if you're not using the
   default)::

      ./pimlico/lib/virtualenv/default/bin/pip install --upgrade tensorflow-gpu

.. note::

   *Changed 12.09.18*: this module takes prepared positive sample data as input instead of
   doing the preparation (random shuffling, etc) during training. I found a bug that meant
   that we weren't training on the full datasets, so training actually takes much longer
   than it seemed. It's therefore important not to waste time redoing data processing on
   each training epoch.

   Some pipelines that were written before this change will no longer work, but they're
   quite simple to fix. Add an extra data preparation module before the training module,
   taking the inputs and parameters from the training module as appropriate (and removing
   some of them from there).

"""
from langsim.datatypes.neural_sixgram import NeuralSixgramTrainingData
from pimlico.core.dependencies.python import sklearn_dependency, \
    keras_tensorflow_dependency
from pimlico.core.modules.base import BaseModuleInfo
from pimlico.core.modules.options import comma_separated_list, str_to_bool
from pimlico.datatypes.base import MultipleInputs
from pimlico.datatypes.dictionary import Dictionary
from pimlico.datatypes.files import FileInput
from pimlico.datatypes.keras import KerasModelBuilderClass
from pimlico.modules.visualization import matplotlib_dependency


class NeuralSixgramKerasModel(KerasModelBuilderClass):
    # Overridden just to add a dependency (and to be sure we're using the right model type)
    def get_software_dependencies(self):
        return super(NeuralSixgramKerasModel, self).get_software_dependencies() + [
            sklearn_dependency, keras_tensorflow_dependency
        ]


class ModuleInfo(BaseModuleInfo):
    module_type_name = "neural_sixgram_trainer"
    module_readable_name = "Neural sixgram (Xsym) trainer, v2"
    module_inputs = [
        ("vocabs", MultipleInputs(Dictionary)),
        ("samples", NeuralSixgramTrainingData),
    ]
    module_optional_inputs = [
        ("mapped_pairs", FileInput()),
    ]
    module_outputs = [("model", NeuralSixgramKerasModel)]
    module_options = {
        "embedding_size": {
            "help": "Number of dimensions in the hidden representation. Default: 30",
            "type": int,
            "default": 30,
        },
        "epochs": {
            "help": "Max number of training epochs. Default: 10",
            "type": int,
            "default": 10,
        },
        "split_epochs": {
            "help": "Normal behaviour is to iterate over the full dataset once in each epoch, generating random "
                    "negative samples to accompany it. Early stopping is done using the validation metric over the "
                    "learned representations after each epoch. With larger datasets, this may mean waiting too long "
                    "before we start measuring the validation metric. If split_epochs > 1, one epoch involves "
                    "1/split_epochs of the data. The following epoch continues iterating over the dataset, so "
                    "all the data gets used, but the early stopping checks are performed split_epochs times in each "
                    "iteration over the dataset",
            "type": int,
            "default": 1,
        },
        "batch": {
            "help": "Training batch size in training samples (pos-neg pairs). Default: 1000",
            "type": int,
            "default": 1000,
        },
        "dropout": {
            "help": "Dropout to apply to embeddings during training. Default: 0.1",
            "type": float,
            "default": 0.1,
        },
        "composition_dropout": {
            "help": "Dropout to apply to composed representation during training. Default: 0.01",
            "type": float,
            "default": 0.01,
        },
        "validation": {
            "help": "Number of samples to hold out as a validation set for training. Simply taken "
                    "from the start of the corpus. Rounded to the nearest number of batches",
            "type": int,
            "default": 1000,
        },
        "composition2_layers": {
            "help": "Number and size of layers to use to combine pairs of characters, given as a list of integers. "
                    "The final layer must be the same size as the embeddings, so is not included in this list. "
                    "Default: nothing, i.e. linear transformation",
            "type": comma_separated_list(int),
            "default": [],
        },
        "composition3_layers": {
            "help": "Number and size of layers to use to combine triples of characters, given as a list of integers. "
                    "The final layer must be the same size as the embeddings, so is not included in this list. "
                    "Default: nothing, i.e. linear transformation",
            "type": comma_separated_list(int),
            "default": [],
        },
        "predictor_layers": {
            "help": "Number and size of layers to use to take a pair of vectors and say whether they belong beside "
                    "each other. Given as a list of integers. Doesn't include the final projection to a single "
                    "score. Default: 30 (single hidden layer)",
            "type": comma_separated_list(int),
            "default": [30],
        },
        "limit_training": {
            "help": "Limit training to this many batches. Default: no limit",
            "type": int,
        },
        "plot_freq": {
            "help": "Output plots to the output directory while training is in progress. This slows down training if "
                    "it's done very often. Specify how many batches to wait between each plot. Fewer means you get "
                    "a finer grained picture of the training process, more means training goes faster. -1 "
                    "turns off plotting. 0 (default) means once at the start/end of each epoch",
            "type": int,
            "default": 0,
        },
        "sim_freq": {
            "help": "How often (in batches) to compute the similarity of overlapping phonemes between the languages. "
                    "-1 (default) means never, 0 means once at the start of each epoch. If input mapped_pairs is "
                    "given, the similarity is computed between these pairs; otherwise we use any identical pairs that "
                    "exist between the vocabularies",
            "type": int,
            "default": -1,
        },
        "unit_norm": {
            "help": "If true, enforce a unit norm constraint on the learned embeddings. Default: true",
            "type": str_to_bool,
            "default": True,
        },
        "restarts": {
            "help": "How many random restarts to perform. Each time, the model is randomly re-initialized from "
                    "scratch. All models are saved and the one with the best value of the validation criterion "
                    "is stored as the output. Default: 1, just train once",
            "type": int,
            "default": 1,
        },
        "patience": {
            "help": "Early stopping patience. "
                    "Number of epochs with no improvement after which training will be stopped. Default: 2",
            "type": int,
            "default": 2,
        },
    }

    def get_software_dependencies(self):
        return super(ModuleInfo, self).get_software_dependencies() + [
            # Require >= 1.0.6 (but can't yet specify that in dep -- future Pimlico feature)
            keras_tensorflow_dependency,
            matplotlib_dependency,
            sklearn_dependency,
        ]
