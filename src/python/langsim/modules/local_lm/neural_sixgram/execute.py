import copy
import itertools
import json
import os
import shutil
from itertools import chain, islice

import matplotlib

from pimlico import cfg
from pimlico.core.modules.execute import ModuleExecutionError
from pimlico.utils.probability import limited_shuffle, limited_shuffle_numpy

matplotlib.rc('font', family="Gentium", size=10)
matplotlib.use("svg")
import pylab as plt

from sklearn.manifold.mds import MDS
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

import random
from operator import itemgetter

import numpy as np
import numpy
import numpy.random as np_random
from keras.callbacks import ModelCheckpoint, Callback

from langsim.modules.local_lm.neural_sixgram.model import NeuralSixgramModel
from pimlico.core.modules.base import BaseModuleExecutor
from pimlico.datatypes.base import InvalidDocument
from pimlico.datatypes.keras import KerasModelBuilderClassWriter
from pimlico.utils.core import infinite_cycle, split_seq


class ModuleExecutor(BaseModuleExecutor):
    def execute(self):
        vocabs = [v.get_data() for v in self.info.get_input("vocabs", always_list=True)]
        corpora = self.info.get_input("corpora", always_list=True)
        if len(vocabs) != len(corpora):
            raise ValueError("you must specify one vocabulary per corpus")
        frequency_arrays = self.info.get_input("frequencies", always_list=True)
        if len(frequency_arrays) != len(corpora):
            raise ValueError("you must specify one frequency array per corpus")
        frequency_arrays = [a.array for a in frequency_arrays]

        # We might have been supplied with a set of pairs that we expect to be mapped closely
        mapped_pairs_input = self.info.get_input("mapped_pairs")
        if mapped_pairs_input is not None:
            mapped_pairs = json.loads(mapped_pairs_input.read_file())
        else:
            mapped_pairs = None

        embedding_size = self.info.options["embedding_size"]
        dropout = self.info.options["dropout"]
        l2_reg = self.info.options["l2_reg"]
        composition_dropout = self.info.options["composition_dropout"]
        if composition_dropout is None:
            composition_dropout = dropout
        epochs = self.info.options["epochs"]
        batch = self.info.options["batch"]
        cross_sentences = self.info.options["cross_sentences"]
        word_internal = self.info.options["word_internal"]
        word_boundary = self.info.options["word_boundary"]

        validation_size = self.info.options["validation"]
        composition2_layer_sizes = self.info.options["composition2_layers"]
        composition3_layer_sizes = self.info.options["composition3_layers"]
        predictor_layer_sizes = self.info.options["predictor_layers"]
        embedding_activation = self.info.options["embedding_activation"]
        unit_norm_constraint = self.info.options["unit_norm"]
        limit_training = self.info.options["limit_training"]
        # Allow 0 to mean don't limit at all
        limit_training = limit_training if limit_training is not None and limit_training > 0 else None
        corpus_offset = self.info.options["corpus_offset"]
        plot_freq = self.info.options["plot_freq"]
        sim_freq = self.info.options["sim_freq"]

        context_weights = self.info.options["context_weights"]
        context_weight_schedule = ContextWeightSchedule(context_weights)

        oov_token = self.info.options["oov"]
        store_all_epochs = self.info.options["store_all"]

        if word_internal:
            # Check that the word boundary is in the input vocabulary for every language
            for v_num, v in enumerate(vocabs):
                if word_boundary not in v.token2id:
                    raise ValueError(u"word boundary character ('{}') not found in vocabulary {}".format(
                        word_boundary, v_num
                    ))
            split_sequences = [v.token2id[word_boundary] for v in vocabs]
        else:
            # Setting this means that sequences won't be split at any word boundary
            split_sequences = None

        validation_batches = int(max(round(float(validation_size) / batch), 1))

        output_dir = self.info.get_absolute_output_dir("model")
        # Need to ensure the output dir exists before the writer takes care of it, so that we can output some
        # preliminary info there
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Vocab indices within the model work as follows:
        #  0 -- vocab1_size-1                       = vocab1 indices
        # vocab1_size                               = OOV1
        # vocab1_size+1 -- vocab1_size+vocab2_size  = vocab2 indices
        # vocab1_size+vocab2_size+1                 = OOV2
        # etc
        if oov_token is None:
            # We assume the vocabs don't include this word
            oov_token = "OOV"
            for vocab in vocabs:
                if oov_token in vocab.token2id:
                    raise ModuleExecutionError("tried to add 'OOV' to vocab as a special value, but it already "
                                               "exists")
                vocab.add_term(oov_token)
                # Unfortunately, we don't know how often OOV occurs
                # We make a guess that it's around the median frequency
                vocab.dfs[oov_token] = float(numpy.median(vocab.dfs.values()))

        try:
            oov_indices = [vocab.token2id[oov_token] for vocab in vocabs]
        except KeyError:
            raise KeyError("special OOV token '%s' not found in all vocabs" % oov_token)
        vocab_sizes = [len(vocab) for vocab in vocabs]
        total_vocab_size = sum(vocab_sizes)

        # Prepare a negative sampling distribution for each of the vocabs
        # Do this in the same way as word2vec
        neg_dists = []
        for v, vocab in enumerate(vocabs):
            # Get char freqs from input and compute unigram dist
            neg_dist = numpy.array(frequency_arrays[v], dtype=numpy.float32)
            # Standard whacky word2vec distribution: f^(3/4) / Z
            # Trying with and without this. If it doesn't help particularly, definitely better to avoid it
            #neg_dist **= 0.75
            neg_dist /= neg_dist.sum()
            neg_dists.append(neg_dist)

            # Output a plot of the distribution used for negative sampling to help understand what the training's doing
            neg_dist_plot_fn = os.path.join(output_dir, "neg_dist_%d.svg" % v)
            self.log.info("Plotting negative sampling (unigram) distribution to %s" % neg_dist_plot_fn)
            plot_dist(neg_dist, vocab, neg_dist_plot_fn, title="Negative sampling dist (unigram) for vocab %d" % v)

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
                # For debugging/progress checking, look for overlap between the vocabs and see how close
                # in the embedding space the overlapping characters are
                overlapping_chars = [
                    (v0_id, other_vocab.token2id[v0_char] + overlap_index_shifts[other_vocab_num])
                    for (v0_char, v0_id) in vocabs[0].token2id.items()
                    for other_vocab_num, other_vocab in enumerate(vocabs[1:], start=1)
                    if v0_char in other_vocab.token2id
                ]
                self.log.info("Found %d overlapping chars between vocabularies" % len(overlapping_chars))
        else:
            overlapping_chars = None

        model = NeuralSixgramModel(build_params=dict(
            embedding_size=embedding_size,
            composition2_layer_sizes=composition2_layer_sizes,
            composition3_layer_sizes=composition3_layer_sizes,
            dropout=dropout,
            composition_dropout=composition_dropout,
            vocab_size=total_vocab_size,
            predictor_layer_sizes=predictor_layer_sizes,
            embedding_activation=embedding_activation,
            unit_norm_constraint=unit_norm_constraint,
            l2_reg=l2_reg,
        ))
        self.log.info(
            "Built model architecture with embedding size=%d, embedding dropout=%.2f, composition dropout=%.2f, "
            "vocab size=%d, composition2 layers=%s, composition3 layers=%s, predictor layers=%s%s" % (
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
                                          "langsim.modules.local_lm.neural_sixgram.model.NeuralSixgramModel") as writer:
            training_iter = CorpusIterator(corpora, vocab_sizes, batch,
                                           skip_batches=validation_batches, neg_dists=neg_dists,
                                           limit_batches=limit_training,
                                           corpus_offset=corpus_offset,
                                           context_size_schedule=context_weight_schedule,
                                           cross_sentences=cross_sentences,
                                           split_sequences=split_sequences)
            validation_iter = CorpusIterator(corpora, vocab_sizes, batch,
                                             limit_batches=validation_batches, neg_dists=neg_dists,
                                             corpus_offset=corpus_offset,
                                             cross_sentences=cross_sentences,
                                             split_sequences=split_sequences)

            # Store this count, as it's irritating to have to wait for it every time if we have to restart several times
            sample_count_filename = os.path.join(self.info.get_absolute_output_dir("model"), "batch_count")
            if os.path.exists(sample_count_filename):
                self.log.info("Using stored batch count")
                with open(sample_count_filename, "r") as f:
                    num_batches = int(f.read())
                # Update the generator's count cache
                training_iter._batches = num_batches
            else:
                self.log.info("Counting training samples")
                num_batches = training_iter.batches
                with open(sample_count_filename, "w") as f:
                    f.write(str(num_batches))

            # Some debugging output to check what the samples look like
            self.log.info("A few training examples:")
            debug_weight_schedule = ContextWeightSchedule(context_weights)
            debug_training_iter = iter(
                CorpusIterator(corpora, vocab_sizes, batch,
                               skip_batches=validation_batches, neg_dists=neg_dists,
                               limit_batches=limit_training,
                               corpus_offset=corpus_offset,
                               context_size_schedule=debug_weight_schedule)
            )
            printer = SampleArrayPrinter(vocabs)
            for whole_epoch in range(5):
                for epoch_frac in [0., 0.5]:
                    print "Epoch=%s" % (whole_epoch + epoch_frac)
                    debug_weight_schedule.update_position(epoch_frac)
                    # Skip randomly into the dataset a bit, so we get different samples each time
                    for i in range(random.randint(0, 10)):
                        debug_training_iter.next()
                    print printer.format(debug_training_iter.next()[0], ordered=True, samples=3).encode("utf8")
                debug_weight_schedule.new_epoch()

            self.log.info("Beginning training for max %d epochs, batch size=%d, %d training batches" %
                          (epochs, batch, training_iter.batches))
            self.log.info("Context size sampling schedule: %s" % context_weight_schedule)
            # Store in the module's history how many samples we trained on, since it might not be the full corpus
            self.info.add_execution_history_record("Training on %d samples (%d batches)" % (
                training_iter.samples, training_iter.batches
            ))

            module_history_cb = ModuleTrainingHistory(self.info)

            callbacks = [
                ModelCheckpoint(
                    writer.weights_filename, monitor="val_loss",
                    save_weights_only=True, save_best_only=not store_all_epochs,
                ),
                module_history_cb,
            ]

            if plot_freq > 0:
                # Output plots now and again
                plot_vocab = sum(
                    [
                        [u"%s:%s" % (vocab_num, char) for i, char in sorted(vocab.id2token.items())]
                        for vocab_num, vocab in enumerate(vocabs)
                    ], []
                )
                plot_dir = os.path.join(output_dir, "training_plots")
                if os.path.exists(plot_dir):
                    shutil.rmtree(plot_dir)
                os.makedirs(plot_dir)
                self.log.info("Outputting plots to %s" % plot_dir)

                # Filter out low-frequency chars for the plotting
                ## I haven't got this quite right yet, so better to leave it out for now
                #freq_threshold = sum(sum(vocab.dfs.values()) for vocab in vocabs) * 0.001
                #include = [i for (i, freq) in enumerate([vocab.dfs[i] for vocab in vocabs for i in range(len(vocab))] + [None])
                #           if freq is None or freq >= freq_threshold]
                callbacks.append(PlotMDS(plot_dir, model, plot_vocab, frequency=plot_freq, lines=overlapping_chars,
                                         context_size_schedule=context_weight_schedule, model_name=self.info.module_name))

            if sim_freq > -1:
                metrics_dir = os.path.join(output_dir, "metrics")
                if os.path.exists(metrics_dir):
                    shutil.rmtree(metrics_dir)
                os.makedirs(metrics_dir)

                callbacks.append(OverlapSimilarity(overlapping_chars, model, frequency=sim_freq,
                                                   filename=os.path.join(metrics_dir, "overlap_rank.csv")))
                # Also output the suggested validation criterion, which doesn't require the overlapping chars map
                callbacks.append(NearestNeighbourDistance(
                    range(0, vocab_sizes[0]), range(vocab_sizes[0], vocab_sizes[0]+vocab_sizes[1]),
                    model, frequency=sim_freq,
                    filename=os.path.join(metrics_dir, "nearest_neighbour_sim.csv")
                ))

            model.model.fit_generator(
                iter(infinite_cycle(training_iter)), training_iter.batches,
                epochs,
                callbacks=callbacks,
                validation_data=iter(infinite_cycle(validation_iter)),
                validation_steps=validation_batches,
                verbose=1 if not cfg.NON_INTERACTIVE_MODE else 0,
            )
            # Don't save the weights here, as they might not be the best ones
            # The ModelCheckpoint has already saved them for us
            writer.task_complete("weights")


class CorpusIterator(object):
    def __init__(self, corpora, vocab_sizes, batch_size, skip_batches=0, limit_batches=None, neg_dists=None,
                 corpus_offset=0, context_size_schedule=None, cross_sentences=False, split_sequences=None):
        # Number of batches to skip at the beginning
        # Used to hold out a validation set
        self.skip_batches = skip_batches
        # Likewise, for producing the validation generator
        self.limit_batches = limit_batches
        self.cross_sentences = cross_sentences
        self.split_sequences = split_sequences is not None
        self.split_sequences_on = split_sequences

        self.batch_size = batch_size
        self.vocab_sizes = vocab_sizes
        self.corpora = corpora
        self.corpus_offset = corpus_offset
        self.corpus_offsets = [i*corpus_offset for i in range(len(corpora))]

        # Compute how much each corpus' indices should be shifted up by
        # OOV is includes in the vocab sizes
        self.index_shifts = reduce(lambda c, x: c + [c[-1] + x], self.vocab_sizes, [0])[:-1]

        if neg_dists is None:
            self.neg_dists = [None] * len(corpora)
        else:
            assert len(neg_dists) == len(corpora)
            self.neg_dists = neg_dists
        self.context_size_schedule = context_size_schedule

        self._batches = None

    def iter_sequences(self, corpus_num, rand=False):
        """
        Setting rand causes us to jump randomly into the corpus before starting iteration, and
        then iterate infinitely over the corpus.

        """
        shift = self.index_shifts[corpus_num]
        corpus = self.corpora[corpus_num]

        if rand:
            # Jump to a random point in the corpus to start iteration
            # First, randomly jump into the corpus at the start of an archive
            # (Don't allow skipping the last archive)
            skip_archive = random.choice([None] + corpus.tarballs[:-1])
            if skip_archive is not None:
                skip_archive = (skip_archive, None)
            # Also (after that) skip a random number of docs
            # We don't know how many docs there are in an archive, but just jumping on by up to 500 docs should
            # give us sufficient randomness
            skip_docs = random.randint(0, 500)
            # By doing these using archive_iter(), we avoid time-consuming preprocessing of skipped docs
            # In the rand case, we iterate infinitely, as it's used to get negative samplers
            corpus_it = (
                (doc_name, doc) for (archive, doc_name, doc) in
                itertools.chain(
                    corpus.archive_iter(start_after=skip_archive, skip=skip_docs),
                    infinite_cycle(corpus.archive_iter)
                )
            )
        else:
            # Just iterate over docs from the start
            corpus_it = corpus

        for doc_name, utts in corpus_it:
            if type(utts) is not InvalidDocument:
                if self.cross_sentences:
                    yield [c+shift for utt in utts for c in utt]
                else:
                    for utt in utts:
                        yield [c+shift for c in utt]

    def iter_split_sequences(self, corpus_num, rand=False):
        """
        Sequences are split on given splitting tokens. Used for splitting at word boundaries,
        if requested.

        """
        if self.split_sequences:
            splitter = self.split_sequences_on[corpus_num]
            for seq in self.iter_sequences(corpus_num, rand=rand):
                for word in split_seq(seq, splitter, ignore_empty_final=True):
                    # Skip empty words
                    if len(word):
                        yield [splitter] + word + [splitter]
        else:
            for seq in self.iter_sequences(corpus_num, rand=rand):
                yield seq

    def iter_char_ngrams(self, corpus_num, n, rand=False):
        # Don't worry about windows that overlap with the ends of the utterance
        offset = self.corpus_offsets[corpus_num]
        if offset > 0 and not rand:
            # First skip this number of utterances, then use them at the end
            utt_iter = chain(islice(self.iter_split_sequences(corpus_num), offset),
                             islice(self.iter_split_sequences(corpus_num), 0, offset))
        else:
            # Just start from the beginning (or jump randomly)
            utt_iter = self.iter_split_sequences(corpus_num, rand=rand)

        for utt in utt_iter:
            if len(utt) >= n:
                # Size-n sliding window over the utterance
                buff = [None] + utt[:n-1]
                for new_item in utt[n-1:]:
                    buff.pop(0)
                    buff.append(new_item)
                    yield copy.copy(buff)

    def iter_interleaved_ngrams(self, n):
        # Stop after one corpus is exhausted, so we don't end up over-trained on a longer corpus at the end
        return interleave_shortest([self.iter_char_ngrams(num, n) for num in range(len(self.corpora))])

    def __iter__(self):
        input_batch = []
        neg_input_batch = []
        output_batch = []

        # Reuse the same array between sequences to avoid slowing things down with memory allocation
        input_array = np.zeros((self.batch_size, 6), dtype=np.int32)
        neg_input_array = np.zeros((self.batch_size, 6), dtype=np.int32)
        # The output array isn't actually needed, since we assume it's full of 1s, but we have to provide something
        # Unless we're doing negative sampling
        output_array = np.ones((self.batch_size, 1), dtype=np.int32)
        done = 0

        # Whole trigrams are drawn from elsewhere in the corpus. This works with methods where positive and negative
        # examples are mixed in the objective function. It changes the behaviour of some methods. E.g. if we use
        # negative composed bigrams (~c1*~c2), this method makes more sense, since the two phonemes belong together,
        # but not with the positive on the other side (c3). With other methods, including "original", there
        # shouldn't theoretically be any difference, but this method is slower, so the other may be preferred.
        if False:
            negative_samplers = [
                limited_shuffle(self.iter_char_ngrams(corpus_num, 3, rand=True), 10000)
                for corpus_num in range(len(self.corpora))
            ]
        else:
            negative_samplers = [
                negative_samples(vocab_size, neg_dist, shift=shift, width=3)
                for (vocab_size, neg_dist, shift) in zip(self.vocab_sizes, self.neg_dists, self.index_shifts)
            ]

        if self.context_size_schedule is not None:
            self.context_size_schedule.new_epoch()

        for corpus_num, pos_ngram in limited_shuffle_numpy(self.iter_interleaved_ngrams(6), 100000):
            # Accumulate windows until we've built up a whole batch
            pos_ngram = list(pos_ngram)
            neg_ngram = list(next(negative_samplers[corpus_num]))
            # Make sure we choose a negative that's different from the pos ngram's first half
            while neg_ngram == pos_ngram[:3]:
                neg_ngram = list(next(negative_samplers[corpus_num]))

            input_batch.append(pos_ngram)
            neg_input_batch.append(neg_ngram)

            if len(input_batch) >= self.batch_size:
                # Don't yield the first batches if we're skipping some
                if done >= self.skip_batches:
                    # Fill the array with the positive and examples
                    fill_array(input_array, input_batch[:self.batch_size])
                    # Fill the negative array first with all the positive examples
                    neg_input_array[:] = input_array
                    # Choose randomly for each sample whether to put the negative trigram on the right or left
                    neg_sides = np_random.randint(2, size=self.batch_size)
                    for i, (lst, side) in enumerate(zip(neg_input_batch[:self.batch_size], neg_sides)):
                        if side == 0:
                            neg_input_array[i, :3] = lst
                        else:
                            neg_input_array[i, 3:] = lst

                    # Choose randomly (uniformly) what size of context to use for each training example
                    # Only use one for each example, so we never train two context sizes at once
                    # Compute the sampling weights for different context sizes
                    if self.context_size_schedule is None:
                        context_size_dist = numpy.array([1., 1., 1., 1., 1., 1., 1., 1., 1.])
                        context_size_dist /= context_size_dist.sum()
                    else:
                        self.context_size_schedule.update_position(float(done - self.skip_batches) / self.batches)
                        context_size_dist = self.context_size_schedule.get_dist()
                    choices = numpy.array([[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]])
                    lr_selection = choices[np_random.choice(9, size=self.batch_size, p=context_size_dist)]

                    context_selector_l = np.zeros((self.batch_size, 3), dtype=np.uint8)
                    context_selector_l[np.arange(self.batch_size), lr_selection[:, 0]] = 1

                    context_selector_r = np.zeros((self.batch_size, 3), dtype=np.uint8)
                    context_selector_r[np.arange(self.batch_size), lr_selection[:, 1]] = 1

                    # Filled up a batch, yield it as numpy arrays
                    yield [input_array[:, i, np.newaxis] for i in range(6)] + \
                          [neg_input_array[:, i, np.newaxis] for i in range(6)] + \
                          [context_selector_l, context_selector_r], output_array
                done += 1
                input_batch = input_batch[self.batch_size:]
                neg_input_batch = neg_input_batch[self.batch_size:]
                output_batch = output_batch[self.batch_size:]

                if self.limit_batches is not None and (done - self.skip_batches) >= self.limit_batches:
                    # Reached the limit, stop now
                    return
        # Skip the final batch, as it causes a problem for keras to have a batch of a different size

    @property
    def samples(self):
        # Since we round to a number of batches, compute #batches and multiply
        # That also takes account of skipped and limited batches
        return self.batches * self.batch_size

    @property
    def batches(self):
        if self._batches is None:
            # First compute the total number of samples
            # Take account of the limit, so we don't need to count the full corpus if we only want a small part
            if self.limit_batches is not None:
                max_samples = (self.skip_batches or 0 + self.limit_batches) * self.batch_size
            else:
                max_samples = None
            samples = sum(1 for __ in islice(self.iter_interleaved_ngrams(6), max_samples))
            # Note we don't return the last partial batch, so round down
            b = samples / self.batch_size
            self._batches = b
        b = self._batches
        # Account for batches skipped at the beginning
        if self.skip_batches is not None:
            b -= self.skip_batches
        # Account for a limit put on the number of batches
        if self.limit_batches is not None:
            b = min(b, self.limit_batches)
        return b


def fill_array(arr, lists):
    for i, lst in enumerate(lists):
        arr[i, :] = lst


def interleave(seqs, finish_all=True):
    """
    Takes a list of sequences and interleaves elements from them, like a flattened zip.

    :param finish_all: keep going until all sequences have been exhausted (default). Once one sequence has been
    exhausted, the others will continue until done. If False, stop as soon as the first sequence is exhausted,
    meaning you get balanced samples from all the sequences, but truncate longer ones
    :param seqs:
    :return:
    """
    iters = [iter(seq) for seq in seqs]
    while len(iters):
        completed = []
        # Take one value from each sequence in turn
        for it_num, it in enumerate(iters):
            try:
                yield it_num, next(it)
            except StopIteration:
                if finish_all:
                    # No more items left in this iter: stop trying to take from it after this round
                    completed.append(it_num)
                else:
                    # Finished one of the sequences: stop now
                    return
        # Remove any iters that have completed from the list
        # Once they're all gone, we'll stop
        iters = [it for (it_num, it) in enumerate(iters) if it_num not in completed]


def interleave_shortest(seqs):
    """
    Takes a list of sequences and interleaves elements from them, like a flattened zip.

    Stop as soon as one sequence is exhausted,
    meaning you get balanced samples from all the sequences, but truncate longer ones.

    """
    iters = [iter(seq) for seq in seqs]
    while True:
        # Take one value from each sequence in turn
        try:
            next_vals = [next(it) for it in iters]
        except StopIteration:
            # If any of the iterables reaches the end, stop, so we have the same number from all
            return
        else:
            # If we've got a value from each, yield them in order, together with their identifier
            for it_num, val in enumerate(next_vals):
                yield it_num, val


def negative_samples(vocab_size, dist, precompute=1000, shift=0, width=4):
    while True:
        # Precompute a large array of samples for efficiency
        # Draw from vocab, including OOV
        # Shift up into shared vocab indices
        samples = np.random.choice(vocab_size, size=(precompute, width), p=dist) + shift
        for i in range(samples.shape[0]):
            yield samples[i]


class SampleArrayPrinter(object):
    def __init__(self, vocabs):
        self.vocabs = vocabs
        self.index_shifts = reduce(lambda c, x: c + [c[-1] + x], (len(v) for v in self.vocabs), [0])[:-1]
        self.vocab = dict(
            (i+self.index_shifts[vocab_num], char)
            for vocab_num, vocab in enumerate(self.vocabs)
            for i, char in vocab.id2token.items()
        )
        self.lang_lookup = dict(
            (i+self.index_shifts[vocab_num], vocab_num)
            for vocab_num, vocab in enumerate(self.vocabs)
            for i, char in vocab.id2token.items()
        )

    def format_char(self, idx):
        return self.vocab.get(int(idx), u"?")

    def format_row(self, arrs, row):
        row_vals = [arrs[i][row] for i in range(14)]

        context_size_l = row_vals[12]
        if context_size_l.sum() > 1:
            raise ValueError("cannot format sample with multiple context sizes")
        context_size_l = numpy.where(context_size_l)[0][0] + 1

        context_size_r = row_vals[13]
        if context_size_r.sum() > 1:
            raise ValueError("cannot format sample with multiple context sizes")
        context_size_r = numpy.where(context_size_r)[0][0] + 1

        # Check all the chars are from the same language
        langs = set(self.lang_lookup[int(row_vals[i])] for i in range(12))
        if len(langs) > 1:
            lang_txt = "multiple!: %s" % langs
        else:
            lang_txt = str(list(langs)[0])

        # Get the right context size for each side
        pos_lhs = u"".join(self.format_char(row_vals[i]) for i in range(3-context_size_l, 3))
        pos_rhs = u"".join(self.format_char(row_vals[i]) for i in range(3, 3+context_size_r))
        neg_lhs = u"".join(self.format_char(row_vals[i]) for i in range(9-context_size_l, 9))
        neg_rhs = u"".join(self.format_char(row_vals[i]) for i in range(9, 9+context_size_r))

        pos_eg = u"%s|%s" % (pos_lhs, pos_rhs)
        neg_eg = u"%s|%s" % (neg_lhs, neg_rhs)

        return u"Lang %s  pos: '%s'  neg: '%s'" % (lang_txt, pos_eg, neg_eg)

    def format(self, arrs, samples=20, ordered=False):
        if ordered:
            rows = range(samples)
        else:
            rows = list(sorted(random.sample(range(arrs[0].shape[0]), samples)))
        return u"\n".join(self.format_row(arrs, row) for row in rows)


class ContextWeightSchedule(object):
    """
    Encapsulates the implementation of a schedule of context-size weights (defining the probability distribution
    from which context sizes are sampled for each training sample) as it potentially changes over the course of
    training.

    """
    def __init__(self, weight_list):
        self.weight_list = [(e, numpy.array(ws)) for (e, ws) in weight_list]
        self.weight_list.sort(key=itemgetter(0))
        self.current_index = 0
        self.current_epoch = 0.
        self.current_whole_epoch = -1.

    def update_position(self, epoch_progress):
        self.current_epoch = self.current_whole_epoch + min(epoch_progress, 1.)

    def new_epoch(self):
        self.current_whole_epoch += 1.

    def get_weights(self):
        if self.current_index is None:
            return self.weight_list[-1][1]
        if len(self.weight_list) > self.current_index+1 and \
                self.current_epoch >= self.weight_list[self.current_index+1][0]:
            # Reached the next checkpoint
            self.current_index += 1
        if self.current_index+1 >= len(self.weight_list):
            # Gone past the end, use the last set of weights
            self.current_index = None
            return self.weight_list[-1][1]
        # Work out where we are between the two current checkpoints
        current_point, current_weights = self.weight_list[self.current_index]
        next_point, next_weights = self.weight_list[self.current_index+1]
        next_scale = (self.current_epoch - current_point) / (next_point - current_point)
        return current_weights*(1.-next_scale) + next_weights*next_scale

    def get_dist(self):
        weights = self.get_weights()
        context_size_pair_dist = weights[[0,1,2,1,3,4,2,4,5]]
        context_size_pair_dist /= context_size_pair_dist.sum()
        return context_size_pair_dist

    def __str__(self):
        return ", ".join("%s:%s,%s,%s,%s,%s,%s" % tuple([epoch] + list(w)) for (epoch, w) in self.weight_list)


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
    def __init__(self, overlapping_indices, model_structure, frequency=200, filename=None):
        super(OverlapSimilarity, self).__init__()
        self.model_structure = model_structure
        self.frequency = frequency
        self.overlapping_indices = overlapping_indices
        self.filename = filename
        self.output_file = None
        self._current_epoch = 0

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
        ranks = [
            float(numpy.where(numpy.argsort(cosine_distances(embeddings[a].reshape(1, -1), embeddings))[0] == b)[0][0]) / embeddings.shape[0]
            for (a, b) in self.overlapping_indices
        ]
        mean_rank = sum(ranks) / len(ranks)
        return mean_rank

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


class NearestNeighbourDistance(Callback):
    """
    Computes the value of a possible unsupervised validation criterion that I'm testing.
    It's the mean cosine distance of each character in lang A to its nearest neighbour in lang B.

    Hopefully this will correlate highly with the mean similarity rank between characters
    that are known to map to each other (in the artificial setup where we know the mapping).

    This was used in the experiments to measure how well this validation criterion correlated
    with actual mapping accuracy, as measured by the OverlapRank callback.

    """
    def __init__(self, lang0_indices, lang1_indices, model_structure, frequency=200, filename=None):
        super(NearestNeighbourDistance, self).__init__()
        self.calculator = ValidationCriterionCalculator(lang0_indices, lang1_indices)
        self.model_structure = model_structure

        self.frequency = frequency
        self.filename = filename
        self.output_file = None
        self._current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_batch_end(self, batch, logs=None):
        if self.frequency > 0 and batch % self.frequency == 0:
            sim = self.compute_sim()
            self.output_to_file(self._current_epoch, batch, sim)

    def on_train_begin(self, logs=None):
        if self.filename is not None:
            self.output_file = open(self.filename, "w")
            self.output_file.write("epoch,batch,value\n")

    def on_train_end(self, logs=None):
        if self.output_file is not None:
            self.output_file.close()

    def compute_sim(self):
        embeddings = self.model_structure.get_embeddings()
        return self.calculator.compute(embeddings)

    def output_to_file(self, epoch, batch, value):
        if self.output_file is not None:
            self.output_file.write("{},{},{}\n".format(epoch, batch, value))
            self.output_file.flush()


class ValidationCriterionCalculator(object):
    """
    Utility to compute the validation criterion given a particular learned set of
    embeddings. This criterion measures the cosine distance of every symbol in language
    A to its nearest neighbour in language B. The result is the average of these
    distances and is a value between 0 and 1. A value closer to 0 means a closer
    matching between the symbols, which is expected to mean a better model.

    There is, of course, no guarantee that a lower validation criterion represents a
    better model, since the nearest neighbours might not be valid mappings that the
    model should learn. However, in our experiments with artificial data where we
    know what the correct mapping between symbols is, we found a very high correlation
    between this metric and a metric that measures the ranking of the correct mappings
    among nearest neighbours. It is therefore a useful proxy for model quality when
    we don't know what the correct mappings are (i.e. in the case of any real
    linguistic data).

    """
    def __init__(self, lang0_indices, lang1_indices):
        self.lang0_indices = lang0_indices
        self.lang1_indices = lang1_indices

    def compute(self, embeddings):
        lang0_embed = embeddings[self.lang0_indices]
        lang1_embed = embeddings[self.lang1_indices]
        # Compute distances between all lang0 chars and all lang1 chars
        dists = cosine_distances(lang0_embed, lang1_embed)
        # Find the min distance from each lang0 char to any lang1 char
        nn_dists = numpy.min(dists, axis=1)
        # The metric is then just the average of these distances
        return nn_dists.mean()


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
        if epoch == 0:
            # At the start of the whole process, output the names to a file
            with open(os.path.join(self.directory, "names.txt"), "w") as f:
                f.write(u"\n".join(self.names).encode("utf-8"))
            if not os.path.exists(self.cos_dir):
                os.makedirs(self.cos_dir)
            if not os.path.exists(self.eucl_dir):
                os.makedirs(self.eucl_dir)

        self.epochs = epoch
        self.plot(0)

    def on_batch_end(self, batch, logs=None):
        if batch > 0 and batch % self.frequency == 0:
            self.plot(batch)

    def plot(self, batch):
        epochs_complete = self.epochs
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
