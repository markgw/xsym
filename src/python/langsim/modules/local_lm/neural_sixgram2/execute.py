import gc
import json
import matplotlib
import os
import shutil
import warnings

from pimlico import cfg
from pimlico.core.modules.execute import ModuleExecutionError

matplotlib.rc('font', family="Gentium", size=10)
matplotlib.use("svg")
import pylab as plt

try:
    # Just importing this does the necessary to select a suitable GPU, where multiple are available
    import setGPU
except ImportError:
    warnings.warn("setGPU is not installed. If you have trouble with GPU allocation, trying installing it "
                  "in running virtualenv")

from sklearn.manifold.mds import MDS
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

import numpy as np
import numpy
import numpy.random as np_random
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ProgbarLogger
from keras.backend import backend

from langsim.modules.local_lm.neural_sixgram2.model import NeuralSixgramModel, ValidationCriterionCalculator, \
    MappedPairsRankCalculator
from pimlico.core.modules.base import BaseModuleExecutor
from pimlico.datatypes.keras import KerasModelBuilderClassWriter
from pimlico.utils.core import infinite_cycle


class ModuleExecutor(BaseModuleExecutor):
    def execute(self):
        # Load the inputs
        # Char vocabularies
        vocabs = [v.get_data() for v in self.info.get_input("vocabs", always_list=True)]
        # Training data, already prepared
        samples = self.info.get_input("samples")
        self.log.info("Training samples: {}".format(len(samples)))

        # We might have been supplied with a set of pairs that we expect to be mapped closely
        mapped_pairs_input = self.info.get_input("mapped_pairs")
        if mapped_pairs_input is not None:
            mapped_pairs = json.loads(mapped_pairs_input.read_file())
        else:
            mapped_pairs = None

        # Architecture parameters
        embedding_size = self.info.options["embedding_size"]
        composition2_layer_sizes = self.info.options["composition2_layers"]
        composition3_layer_sizes = self.info.options["composition3_layers"]
        predictor_layer_sizes = self.info.options["predictor_layers"]

        # Training parameters
        batch = self.info.options["batch"]
        epochs = self.info.options["epochs"]
        split_epochs = self.info.options["split_epochs"]
        dropout = self.info.options["dropout"]
        unit_norm_constraint = self.info.options["unit_norm"]
        composition_dropout = self.info.options["composition_dropout"]
        sim_freq = self.info.options["sim_freq"]
        early_stopping_patience = self.info.options["patience"]
        random_restarts = self.info.options["restarts"] or 1

        # Training dataset parameters
        validation_size = self.info.options["validation"]
        validation_batches = int(max(round(float(validation_size) / batch), 1))
        limit_training = self.info.options["limit_training"]
        # Allow 0 to mean don't limit at all
        limit_training = limit_training if limit_training is not None and limit_training > 0 else None

        # Visualization parameters
        plot_freq = self.info.options["plot_freq"]

        # Need to ensure the output dir exists before the writer takes care of it, so that we can output some
        # preliminary info there
        output_dir = self.info.get_absolute_output_dir("model")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Create a subdirectory to store the random restarts, one of which gets restored at the end
        restarts_dir = os.path.join(output_dir, "restarts")
        if os.path.exists(restarts_dir):
            shutil.rmtree(restarts_dir)
        os.makedirs(restarts_dir)

        data_dir = os.path.join(output_dir, "data")

        # If we're using Tensorflow, check what device(s) we use and output
        # E.g. if we're expecting to be using a GPU, it's useful to know if we're not
        if backend() == "tensorflow":
            self.log.info("Keras is using Tensorflow backend: showing device placement for some simple ops")
            show_tensorflow_devices()
        else:
            self.log.info("Keras is using '{}' backend".format(backend()))

        vocab_sizes = [len(vocab) for vocab in vocabs]
        total_vocab_size = sum(vocab_sizes)
        self.log.info("Vocab sizes: %s" % ", ".join(str(v) for v in vocab_sizes))

        if sim_freq > -1:
            overlap_index_shifts = reduce(lambda c, x: c + [c[-1] + x], vocab_sizes, [0])[:-1]
            if mapped_pairs is not None:
                # We've been given a specific set of pairs that should be mapped to one another
                self.log.info("Using given pair mapping (%d pairs) to see how well the training is going" %
                              len(mapped_pairs))
                # Check the mapping only includes valid chars before going on
                for lhs, rhs in mapped_pairs:
                    if lhs not in vocabs[0].token2id:
                        raise ModuleExecutionError("specified pair mapping includes a char on the LHS that's not "
                                                   "in vocab 0: {}".format(lhs.encode("ascii", "replace")))
                    if rhs not in vocabs[1].token2id:
                        raise ModuleExecutionError("specified pair mapping includes a char on the RHS that's not "
                                                   "in vocab 1: {}".format(rhs.encode("ascii", "replace")))
                # For this we assume there are only two vocabs and the pairs are between them
                # If you want to extend this to more in future you'll need to allow some way to specify the vocab
                overlapping_chars = [
                    (vocabs[0].token2id[lhs], vocabs[1].token2id[rhs] + overlap_index_shifts[1])
                    for (lhs, rhs) in mapped_pairs
                ]
            else:
                # No pair correspondences given, but we're going to be outputting this metric anyway
                # Look for overlap between the vocabs and use those as correspondences
                overlapping_chars = [
                    (v0_id, other_vocab.token2id[v0_char] + overlap_index_shifts[other_vocab_num])
                    for (v0_char, v0_id) in vocabs[0].token2id.items()
                    for other_vocab_num, other_vocab in enumerate(vocabs[1:], start=1)
                    if v0_char in other_vocab.token2id
                ]
                self.log.info("Found %d overlapping chars between vocabularies" % len(overlapping_chars))
        else:
            overlapping_chars = None

        # Restart the whole training process, including random initialization, a number of times
        restart_scores = []
        for restart_num in range(random_restarts):
            if random_restarts > 1:
                self.log.info("RANDOM RESTART {}/{}".format(restart_num+1, random_restarts))
                self.info.add_execution_history_record("Random restart {}/{}".format(restart_num+1, random_restarts))

            model = NeuralSixgramModel(build_params=dict(
                embedding_size=embedding_size,
                composition2_layer_sizes=composition2_layer_sizes,
                composition3_layer_sizes=composition3_layer_sizes,
                dropout=dropout,
                composition_dropout=composition_dropout,
                vocab_size=total_vocab_size,
                predictor_layer_sizes=predictor_layer_sizes,
                unit_norm_constraint=unit_norm_constraint,
            ))
            if restart_num == 0:
                # Same each time
                self.log.info(
                    "Built model architecture with embedding size=%d, embedding dropout=%.2f, "
                    "composition dropout=%.2f, vocab size=%d, composition2 layers=%s, composition3 layers=%s, "
                    "predictor layers=%s%s" % (
                        embedding_size, dropout, composition_dropout, total_vocab_size,
                        "->".join(str(s) for s in model.composition2_layer_sizes),
                        "->".join(str(s) for s in model.composition3_layer_sizes),
                        "->".join(str(s) for s in model.predictor_layer_sizes),
                        ", unit norm constraint" if unit_norm_constraint else "",
                    )
                )
            model.compile()

            with KerasModelBuilderClassWriter(output_dir,
                                              model.params,
                                              "langsim.modules.local_lm.neural_sixgram2.model.NeuralSixgramModel") as writer:
                training_iter = BatchedIterator(samples, vocab_sizes, batch,
                                                skip_batches=validation_batches, limit_batches=limit_training)
                validation_iter = BatchedIterator(samples, vocab_sizes, batch, limit_batches=validation_batches)

                self.log.info("Beginning training for max %d epochs, batch size=%d, %d training batches" %
                              (epochs*split_epochs, batch, training_iter.batches/split_epochs))
                if restart_num == 0:
                    # Store in the module's history how many samples we trained on, since it might not be full corpus
                    self.info.add_execution_history_record("Training on %d samples (%d batches)" % (
                        training_iter.samples, training_iter.batches
                    ))
                    if split_epochs > 1:
                        self.log.info("Splitting dataset into %d epochs per iteration" % split_epochs)
                        self.info.add_execution_history_record("Splitting data into %d epochs, each with %d batches" % (
                            epochs*split_epochs, training_iter.batches/split_epochs
                        ))

                lang0_indices = range(0, vocab_sizes[0])
                lang1_indices = range(vocab_sizes[0], vocab_sizes[0]+vocab_sizes[1]),

                module_history_cb = ModuleTrainingHistory(self.info)
                checkpoint = ModelCheckpoint(
                    writer.weights_filename, monitor="nn_dist", mode="min",
                    save_weights_only=True, save_best_only=True,
                )

                callbacks = [
                    # Computes 'nn_dist' metric
                    NearestNeighbourDistanceCallback(lang0_indices, lang1_indices),
                    # Only save the model with the best nn_dist
                    checkpoint,
                    # Use early stopping based on nn_dist
                    EarlyStopping(monitor="nn_dist", mode="min", patience=early_stopping_patience, verbose=1),
                    module_history_cb,
                    LogPrinter(["nn_dist"], self.log),
                ]
                if not cfg.NON_INTERACTIVE_MODE:
                    # Put progbar at the start of callbacks, so the others can output things between epochs
                    callbacks.insert(0, ProgbarLogger(count_mode='steps'))

                if plot_freq >= 0:
                    # Output plots now and again
                    plot_vocab = sum(
                        [
                            [u"%s:%s" % (vocab_num, char) for i, char in sorted(vocab.id2token.items())]
                            for vocab_num, vocab in enumerate(vocabs)
                        ], []
                    )
                    plot_dir = os.path.join(data_dir, "training_plots")
                    if os.path.exists(plot_dir):
                        shutil.rmtree(plot_dir)
                    os.makedirs(plot_dir)
                    if restart_num == 0:
                        self.log.info("Outputting plots %s to %s" %
                                      ("every %s batches" % plot_freq if plot_freq > 0 else "between every epoch",
                                       plot_dir))

                    callbacks.append(PlotMDS(
                        plot_dir, model, plot_vocab, frequency=plot_freq, lines=overlapping_chars,
                        model_name=self.info.module_name
                    ))

                if sim_freq > -1:
                    metrics_dir = os.path.join(data_dir, "metrics")
                    if os.path.exists(metrics_dir):
                        shutil.rmtree(metrics_dir)
                    os.makedirs(metrics_dir)

                    callbacks.append(OverlapSimilarity(overlapping_chars, model, vocab_sizes[0], frequency=sim_freq,
                                                       filename=os.path.join(metrics_dir, "overlap_rank.csv")))

                model.model.fit_generator(
                    iter(infinite_cycle(training_iter)),
                    training_iter.batches/split_epochs,
                    epochs*split_epochs,
                    callbacks=callbacks,
                    validation_data=iter(infinite_cycle(validation_iter)),
                    validation_steps=validation_batches,
                    verbose=0 if cfg.NON_INTERACTIVE_MODE else 1,
                )

                # Don't save the weights here, as they might not be the best ones
                # The ModelCheckpoint has already saved them for us
                writer.task_complete("weights")

                # Use the checkpoint callback to find what the best NN dist score was -- the one that got kept
                restart_scores.append(checkpoint.best)

                if random_restarts > 1:
                    # Move the result of this restart to a subdirectory ready to move back if it's the best
                    shutil.move(data_dir, os.path.join(restarts_dir, str(restart_num)))

            # Prompt garbage collection now so that things get cleared up before the next restart
            callbacks = model = training_iter = validation_iter = None
            gc.collect()

        if random_restarts > 1:
            self.log.info("Completed all random restarts")
            best_restart = numpy.argmin(restart_scores)
            self.log.info("Best score came from run {}: {}. Storing that model".format(
                best_restart, restart_scores[best_restart]))
            self.info.add_execution_history_record("Best score came from run {}: {}. Storing that model".format(
                best_restart, restart_scores[best_restart]))
            shutil.copytree(os.path.join(restarts_dir, str(best_restart)), data_dir)


class BatchedIterator(object):
    def __init__(self, samples, vocab_sizes, batch_size, skip_batches=0, limit_batches=None):
        # Number of batches to skip at the beginning
        # Used to hold out a validation set
        self.skip_batches = skip_batches
        # Likewise, for producing the validation generator
        self.limit_batches = limit_batches

        self.batch_size = batch_size
        self.vocab_sizes = vocab_sizes
        self.training_samples = samples

    def __iter__(self):
        # Reuse the same array between sequences to avoid slowing things down with memory allocation
        input_array = np.zeros((self.batch_size, 6), dtype=np.int32)
        neg_input_array = np.zeros((self.batch_size, 6), dtype=np.int32)
        # The output array isn't actually needed, since we assume it's full of 1s, but we have to provide something
        # Unless we're doing negative sampling
        output_array = np.ones((self.batch_size, 1), dtype=np.int32)
        done = 0
        row = 0

        for pos_ngram, neg_ngram in self.training_samples:
            # Accumulate windows until we've built up a whole batch
            input_array[row, :] = pos_ngram
            neg_input_array[row, :] = neg_ngram

            if row == self.batch_size - 1:
                # Don't yield the first batches if we're skipping some
                if done >= self.skip_batches:
                    # Choose randomly (uniformly) what size of context to use for each training example
                    # Equally likely to sample any combination of sizes, but never sample 1-1 (0,0),
                    # as this is too little context
                    # This used to be parameterized, but using fancy schedules for increasing the context size didn't
                    # have any (major) postive impact, so now we keep it fixed
                    choices = numpy.array([[0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]])
                    lr_selection = choices[np_random.choice(8, size=self.batch_size)]

                    context_selector_l = np.zeros((self.batch_size, 3), dtype=np.uint8)
                    context_selector_l[np.arange(self.batch_size), lr_selection[:, 0]] = 1

                    context_selector_r = np.zeros((self.batch_size, 3), dtype=np.uint8)
                    context_selector_r[np.arange(self.batch_size), lr_selection[:, 1]] = 1

                    # Filled up a batch, yield it as numpy arrays
                    yield [input_array[:, i, np.newaxis] for i in range(6)] + \
                          [neg_input_array[:, i, np.newaxis] for i in range(6)] + \
                          [context_selector_l, context_selector_r], output_array
                done += 1
                row = 0

                if self.limit_batches is not None and (done - self.skip_batches) >= self.limit_batches:
                    # Reached the limit, stop now
                    return
            else:
                row += 1

        # Skip the final batch, as it causes a problem for keras to have a batch of a different size

    @property
    def samples(self):
        # Since we round to a number of batches, compute #batches and multiply
        # That also takes account of skipped and limited batches
        return self.batches * self.batch_size

    @property
    def batches(self):
        # First read the total number of samples
        samples = len(self.training_samples)
        # Note we don't return the last partial batch, so round down
        b = samples / self.batch_size
        # Account for batches skipped at the beginning
        if self.skip_batches is not None:
            b -= self.skip_batches
        # Account for a limit put on the number of batches
        if self.limit_batches is not None:
            b = min(b, self.limit_batches)
        return b


class NearestNeighbourDistanceCallback(Callback):
    """
    Keras callback that computes the validation criterion after every epoch
    and stores it in the logs. The value can then be used for early stopping,
    or other things, using the name 'nn_dist'.

    We store the negative of the nn_dist metric, so that a higher value is better.
    This makes it easier to use for early stopping, etc.

    """
    def __init__(self, lang0_indices, lang1_indices):
        super(NearestNeighbourDistanceCallback, self).__init__()
        self.val_crit_calculator = ValidationCriterionCalculator(lang0_indices, lang1_indices)

    def update_nn_dist(self, logs):
        if logs is None:
            raise ValueError("nearest neighbour distance callback needs to update logs to store metric, but "
                             "logs dictionary was not given")
        embeddings = self.model.get_layer(name="single_char_embeddings").get_weights()[0]
        dist = self.val_crit_calculator.compute(embeddings)
        logs["nn_dist"] = dist

    def on_epoch_begin(self, epoch, logs=None):
        # Update at the start of the first epoch, so we get an initial value
        if epoch == 0:
            self.update_nn_dist(logs)

    def on_epoch_end(self, epoch, logs=None):
        # Then update at the end of every epoch
        self.update_nn_dist(logs)


class ModuleTrainingHistory(Callback):
    """
    Store training info to model history between epochs
    """
    def __init__(self, module, early_stopping=None):
        super(ModuleTrainingHistory, self).__init__()
        self.early_stopping = early_stopping
        self.module = module

    def on_epoch_end(self, epoch, logs={}):
        self.module.add_execution_history_record("Completed training epoch %d. %s" %
                                                 (epoch, ", ".join("%s=%s" % (k, v) for (k, v) in logs.iteritems())))

    def on_train_end(self, logs={}):
        # Check whether the early stopping callback has something to say
        if self.early_stopping and self.early_stopping.stopped_epoch > 0:
            self.module.add_execution_history_record("Triggered early stopping on epoch %d" %
                                                     self.early_stopping.stopped_epoch)


class OverlapSimilarity(Callback):
    """
    Handy for debugging.

    Prints the computed value. Can also output to a file, so you can check the values later.

    """
    def __init__(self, overlapping_indices, model_structure, vocab1_size, frequency=200, filename=None):
        super(OverlapSimilarity, self).__init__()
        self.vocab1_size = vocab1_size
        self.model_structure = model_structure
        self.frequency = frequency
        self.filename = filename
        self.output_file = None
        self._current_epoch = 0
        self.calculator = MappedPairsRankCalculator(overlapping_indices)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            self.print_sim(0, epoch, self.compute_sim())
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self.print_sim(0, epoch+1, self.compute_sim())

    def on_batch_end(self, batch, logs=None):
        if self.frequency > 0 and batch % self.frequency == 0:
            sim = self.compute_sim()
            self.output_to_file(self._current_epoch, batch, sim)
            if batch > 0:
                self.print_sim(batch, None, sim)

    def on_train_begin(self, logs=None):
        if self.filename is not None:
            self.output_file = open(self.filename, "w")
            self.output_file.write("epoch,batch,value\n")

    def on_train_end(self, logs=None):
        if self.output_file is not None:
            self.output_file.close()

    def compute_sim(self):
        embeddings = self.model_structure.get_embeddings()
        embeddings1 = embeddings[:self.vocab1_size]
        embeddings2 = embeddings[self.vocab1_size:]
        return self.calculator.compute(embeddings1, embeddings2)

    def output_to_file(self, epoch, batch, value):
        if self.output_file is not None:
            self.output_file.write("{},{},{}\n".format(epoch, batch, value))
            self.output_file.flush()

    def print_sim(self, batch, epoch, value):
        if not batch and epoch is not None:
            if epoch == 0:
                place = "at beginning"
            else:
                place = "after epoch %d" % (epoch-1)
        elif epoch is not None:
            place = "after epoch %d, batch %d" % (epoch, batch)
        else:
            place = "after batch %d" % batch

        print " Mean similarity rank of overlapping chars %s: %.3f%%" % (place, value*100.)


class PlotMDS(Callback):
    """
    Handy for debugging

    """
    def __init__(self, directory, model, names, frequency=200, lines=None, include=None, context_size_schedule=None,
                 model_name=None):
        super(PlotMDS, self).__init__()
        self.model_name = model_name
        self.context_size_schedule = context_size_schedule
        self.include = include
        self.lines = lines
        self.model_structure = model
        self.directory = directory
        self.frequency = frequency
        self.epochs = 0
        self.names = names

        if self.include is not None:
            self.names = [n for (i, n) in enumerate(names) if i in self.include]
            idx_map = dict((orig, new) for (new, orig) in enumerate(self.include))
            if self.lines is not None:
                self.lines = [
                    (idx_map[a], idx_map[b]) for (a, b) in self.lines if a in idx_map and b in idx_map
                ]
        langs = set(n.partition(":")[0] for n in self.names)
        self.colours = dict(zip(langs, ["r", "b", "g", "y", "o"]))

        self.cos_dir = os.path.join(self.directory, "cosine")
        self.eucl_dir = os.path.join(self.directory, "euclidean")

    def on_epoch_begin(self, epoch, logs=None):
        self.epochs = epoch

        if epoch == 0:
            # At the start of the whole process, output the names to a file
            with open(os.path.join(self.directory, "names.txt"), "w") as f:
                f.write(u"\n".join(self.names).encode("utf-8"))
            if not os.path.exists(self.cos_dir):
                os.makedirs(self.cos_dir)
            if not os.path.exists(self.eucl_dir):
                os.makedirs(self.eucl_dir)

            self.plot(0)

    def on_batch_end(self, batch, logs=None):
        if self.frequency > 0 and batch > 0 and batch % self.frequency == 0:
            self.plot(batch)

    def on_epoch_end(self, epoch, logs=None):
        self.plot(0, end=True)

    def plot(self, batch, end=False):
        epochs_complete = self.epochs
        if end:
            epochs_complete += 1
        # Fetch embedding matrix from the model
        # Cast to float64 or else you can get non-symmetric problems
        embed = self.model_structure.get_embeddings().astype(numpy.float64)
        if self.include is not None:
            embed = embed[np.array(self.include), :]
        # Output the embeddings to a file, so we can replot everything later
        numpy.save(os.path.join(self.directory, "embed_%03d_%03d.npy" % (epochs_complete, batch)), embed)

        for metric, metric_name, directory in [
            (cosine_distances, "cos", self.cos_dir),
            (euclidean_distances, "eucl", self.eucl_dir)
        ]:
            distances = metric(embed)
            # Use consistent random init
            mds = MDS(n_components=2, random_state=1234, n_init=1, dissimilarity="precomputed")
            mds.fit(distances)

            # Output a plot
            filename = os.path.join(directory, "plot_%03d_%03d.svg" % (epochs_complete, batch))
            latest_filename = os.path.join(directory, "latest.svg")

            coords = mds.embedding_
            fig, ax = plt.subplots()
            for (x, y), name in zip(coords, self.names):
                ax.annotate(self.point_name(name), (x, y), color=self.point_colour(name))
            plt.xlim((1.1*coords[:, 0].min(), 1.1*coords[:, 0].max()))
            plt.ylim((1.1*coords[:, 1].min(), 1.1*coords[:, 1].max()))

            title = "%sMDS (%s) after %d epochs + %d batches" % (
                "%s " % self.model_name if self.model_name is not None else "", metric_name, epochs_complete, batch
            )
            if self.context_size_schedule is not None:
                context_size_weights = self.context_size_schedule.get_dist()
                title += "\ncontext weights: %s" % (", ".join("%.2f" % x for x in context_size_weights))

            plt.title(title)

            if self.lines:
                for a, b in self.lines:
                    ax.annotate(
                        "",
                        xy=coords[a], xycoords='data',
                        xytext=coords[b], textcoords='data',
                        arrowprops=dict(arrowstyle="-", color="0.85", connectionstyle="arc3,rad=0"),
                    )
            plt.savefig(filename)
            shutil.copy2(filename, latest_filename)
            plt.close()

    @staticmethod
    def point_name(name):
        char = name.partition(":")[2]
        if char == u" ":
            char = u"-"
        return char

    def point_colour(self, name):
        return self.colours[name.partition(":")[0]]


class LogPrinter(Callback):
    """
    Simple Keras callback to print out values of named metrics in the logs
    at the end of each epoch.

    """
    def __init__(self, metric_names, log):
        super(LogPrinter, self).__init__()
        self.log = log
        self.metric_names = metric_names

    def print_logs(self, logs, prefix=""):
        self.log.info("{}{}".format(
            prefix,
            ", ".join("{}={}".format(name, logs.get(name, "not set")) for name in self.metric_names)
        ))

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0 and logs is not None:
            self.print_logs(logs, "Before first epoch: ")

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            self.print_logs(logs, "After epoch {}: ".format(epoch+1))


def plot_dist(dist, vocab, filename, title=None):
    fig, ax = plt.subplots()

    ordered_ids = list(reversed(numpy.argsort(dist)))

    xs = numpy.arange(dist.shape[0])
    ax.bar(xs, dist[ordered_ids], 0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([vocab.id2token[i] for i in ordered_ids])

    if title is not None:
        plt.title(title)

    plt.savefig(filename)
    plt.close()


def show_tensorflow_devices():
    import tensorflow as tf
    # Create a simple graph.
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op to see what devices are used
    print sess.run(c)
