import copy
from itertools import chain, islice

import numpy as np
import random

from langsim.datatypes.neural_sixgram import NeuralSixgramTrainingDataWriter
from pimlico.core.modules.base import BaseModuleExecutor
from pimlico.core.modules.execute import ModuleExecutionError
from pimlico.datatypes.base import InvalidDocument
from pimlico.utils.probability import limited_shuffle_numpy
from pimlico.utils.progress import get_progress_bar


class ModuleExecutor(BaseModuleExecutor):
    def execute(self):
        # Load the inputs
        # Char vocabularies
        vocabs = [v.get_data() for v in self.info.get_input("vocabs", always_list=True)]
        # Training corpora (same num as vocabs)
        corpora = self.info.get_input("corpora", always_list=True)
        self.log.info("Documents in training corpora: {}".format(", ".join(str(len(c)) for c in corpora)))
        # Frequencies of characters for unigram distribution (same num as vocabs)
        frequency_arrays = self.info.get_input("frequencies", always_list=True)
        if len(frequency_arrays) != len(vocabs):
            raise ValueError("you must specify one frequency array per corpus")
        frequency_arrays = [a.array for a in frequency_arrays]

        # Training parameters
        cross_sentences = self.info.options["cross_sentences"]
        corpus_offset = self.info.options["corpus_offset"]
        oov_token = self.info.options["oov"]
        shuffle_window = self.info.options["shuffle_window"]

        vocab_sizes = [len(vocab) for vocab in vocabs]
        self.log.info("Vocab sizes: %s" % ", ".join(str(v) for v in vocab_sizes))

        # Vocab indices within the model work as follows:
        #  0 -- vocab1_size-1                       = vocab1 indices
        # vocab1_size                               = OOV1
        # vocab1_size+1 -- vocab1_size+vocab2_size  = vocab2 indices
        # vocab1_size+vocab2_size+1                 = OOV2
        # etc
        if oov_token is None:
            # Supposedly, the vocabs don't include a special OOV token, so we add one now
            oov_token = "OOV"
            for vocab in vocabs:
                if oov_token in vocab.token2id:
                    raise ModuleExecutionError("tried to add 'OOV' to vocab as a special value, but it already exists")
                vocab.add_term(oov_token)
                # We don't know how often OOV occurs, but it probably doesn't matter
                # Guess that it's around the median frequency
                vocab.dfs[oov_token] = float(np.median(vocab.dfs.values()))
        if not all(oov_token in vocab.token2id for vocab in vocabs):
            raise KeyError("special OOV token '%s' not found in all vocabs" % oov_token)

        # Prepare a negative sampling distribution for each of the vocabs
        neg_dists = []
        for v, vocab in enumerate(vocabs):
            # Get char freqs from input and compute unigram dist
            neg_dist = np.array(frequency_arrays[v], dtype=np.float32)
            neg_dist /= neg_dist.sum()
            neg_dists.append(neg_dist)

        with NeuralSixgramTrainingDataWriter(self.info.get_absolute_output_dir("samples")) as writer:
            # Wrap the first corpus in a progress bar monitor, so we have some idea of progress
            pbar = get_progress_bar(len(corpora[0]), title="Sampling")
            corpora[0] = pbar(corpora[0])

            training_iter = CorpusIterator(corpora, vocab_sizes,
                                           corpus_offset=corpus_offset,
                                           cross_sentences=cross_sentences,
                                           shuffle_window=shuffle_window,
                                           neg_dists=neg_dists)

            self.log.info("Iterating over training data, sampling negatives and writing samples")
            for pos_sample, neg_sample in training_iter:
                writer.add_sample(pos_sample, neg_sample)
            self.log.info("Finished writing output data to {}".format(writer.data_file_path))


class CorpusIterator(object):
    def __init__(self, corpora, vocab_sizes, corpus_offset=0, cross_sentences=False, shuffle_window=1000, neg_dists=None):
        self.cross_sentences = cross_sentences

        self.shuffle_window = shuffle_window
        self.cross_sentences = cross_sentences

        self.vocab_sizes = vocab_sizes
        self.corpora = corpora
        self.corpus_offset = corpus_offset
        self.corpus_offsets = [i*corpus_offset for i in range(len(corpora))]

        # Compute how much each corpus' indices should be shifted up by
        # OOV is includes in the vocab sizes
        self.index_shifts = reduce(lambda c, x: c + [c[-1] + x], self.vocab_sizes, [0])[:-1]

        if neg_dists is None:
            self.neg_dists = [None] * len(vocab_sizes)
        else:
            assert len(neg_dists) == len(vocab_sizes)
            self.neg_dists = neg_dists

        self._batches = None

    def iter_sequences(self, corpus_num):
        """
        Setting rand causes us to jump randomly into the corpus before starting iteration, and
        then iterate infinitely over the corpus.

        """
        shift = self.index_shifts[corpus_num]
        corpus = self.corpora[corpus_num]

        for doc_name, utts in corpus:
            if type(utts) is not InvalidDocument:
                if self.cross_sentences:
                    yield [c+shift for utt in utts for c in utt]
                else:
                    for utt in utts:
                        yield [c+shift for c in utt]

    def iter_char_ngrams(self, corpus_num, n):
        # Don't worry about windows that overlap with the ends of the utterance
        offset = self.corpus_offsets[corpus_num]
        if offset > 0:
            # First skip this number of utterances, then use them at the end
            utt_iter = chain(islice(self.iter_sequences(corpus_num), offset, None),
                             islice(self.iter_sequences(corpus_num), offset))
        else:
            # Just start from the beginning (or jump randomly)
            utt_iter = self.iter_sequences(corpus_num)

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

    def iter_positive_samples(self):
        return limited_shuffle_numpy(self.iter_interleaved_ngrams(6), self.shuffle_window)

    def __iter__(self):
        # Prepare samplers to draw negative samples from the unigram distributions of each corpus
        negative_samplers = [
            negative_samples(vocab_size, neg_dist, shift=shift, width=3)
            for (vocab_size, neg_dist, shift) in zip(self.vocab_sizes, self.neg_dists, self.index_shifts)
        ]

        for corpus_num, pos_ngram in self.iter_positive_samples():
            # Choose randomly for each sample whether to put the negative trigram on the right or left
            neg_side = bool(random.getrandbits(1))
            pos_half = pos_ngram[3:] if neg_side else pos_ngram[:3]
            # Draw a negative sample to go with this positive
            neg_half = list(next(negative_samplers[corpus_num]))
            # Make sure we choose a negative that's different from the half of the pos ngram that we're replacing
            while neg_half == pos_half:
                neg_half = list(next(negative_samplers[corpus_num]))

            # Put the negative half-n-gram together with the other half to get a negative sample
            if neg_side:
                neg_ngram = pos_ngram[:3] + neg_half
            else:
                neg_ngram = neg_half + pos_ngram[3:]

            yield pos_ngram, neg_ngram


def negative_samples(vocab_size, dist, precompute=1000, shift=0, width=4):
    while True:
        # Precompute a large array of samples for efficiency
        # Draw from vocab, including OOV
        # Shift up into shared vocab indices
        samples = np.random.choice(vocab_size, size=(precompute, width), p=dist) + shift
        for i in range(samples.shape[0]):
            yield samples[i]


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

