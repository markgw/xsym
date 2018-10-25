import copy
from sklearn.metrics.pairwise import cosine_distances

import numpy
from keras import backend as K
from keras.constraints import unit_norm
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.core import Dense, SpatialDropout1D, Lambda, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate, _Merge, Add
from keras.optimizers import Adam
from keras.regularizers import l2


class NeuralSixgramModel(object):
    default_params = {
        "embedding_size": 100,
        "composition2_layer_sizes": [100],
        "composition3_layer_sizes": [100],
        "dropout": 0.3,
        "composition_dropout": 0.1,
        "vocab_size": None,
        "predictor_layer_sizes": [100, 50],
        "unit_norm_constraint": False,
        "l2_reg": 0.,   # No longer used by training module: left here for bw compat with trained models
    }

    def __init__(self, build_params={}):
        for key in build_params:
            if key not in self.default_params:
                raise ValueError("invalid model param '%s' for trigram model" % key)
        self.params = copy.copy(self.default_params)
        self.params.update(build_params)

        self.embedding_size = self.params["embedding_size"]
        self.composition2_layer_sizes = self.params["composition2_layer_sizes"] + [self.embedding_size]
        self.composition3_layer_sizes = self.params["composition3_layer_sizes"] + [self.embedding_size]
        self.unit_norm_constraint = self.params["unit_norm_constraint"]
        self.predictor_layer_sizes = self.params["predictor_layer_sizes"] + [1]
        self.dropout = self.params["dropout"]
        self.l2_reg = self.params["l2_reg"]
        self.composition_dropout = self.params["composition_dropout"]
        self.vocab_size = self.params["vocab_size"]
        if self.vocab_size is None:
            raise ValueError("vocab_size must be specified when initializing trigram model")

        # An input for each character in the trigram
        self.input1 = Input(shape=(1,), dtype='int32', name="char1")
        self.input2 = Input(shape=(1,), dtype='int32', name="char2")
        self.input3 = Input(shape=(1,), dtype='int32', name="char3")
        self.input4 = Input(shape=(1,), dtype='int32', name="char4")
        self.input5 = Input(shape=(1,), dtype='int32', name="char5")
        self.input6 = Input(shape=(1,), dtype='int32', name="char6")
        # We also need inputs for negative (randomly chosen) examples
        self.neg_input1 = Input(shape=(1,), dtype='int32', name="neg_char1")
        self.neg_input2 = Input(shape=(1,), dtype='int32', name="neg_char2")
        self.neg_input3 = Input(shape=(1,), dtype='int32', name="neg_char3")
        self.neg_input4 = Input(shape=(1,), dtype='int32', name="neg_char4")
        self.neg_input5 = Input(shape=(1,), dtype='int32', name="neg_char5")
        self.neg_input6 = Input(shape=(1,), dtype='int32', name="neg_char6")
        # A further input selects, for each example, whether to include factors from each of the ngram sizes in
        # the BPR objective
        self.context_selector_l = Input(shape=(3,), dtype='uint8', name="context_selector_l")
        self.context_selector_r = Input(shape=(3,), dtype='uint8', name="context_selector_r")

        # These embeddings are used to represent each char, including those that then get composed
        constraint = unit_norm(axis=1) if self.unit_norm_constraint else None
        self.char_embeddings = Embedding(
            self.vocab_size, self.embedding_size, name="single_char_embeddings", embeddings_constraint=constraint
        )
        self.dropout_layer = SpatialDropout1D(self.dropout, name="dropout")
        # Keras expects us to use Embedding with sequences, but we've only got one input index per sample
        # We therefore need to squeeze the output to drop the time dimension
        squeeze_time = Lambda(lambda x: K.squeeze(x, 1), output_shape=lambda s: (s[0], s[2]), name="drop_time_dim")

        # Single-char embeddings
        def _apply_embeddings(inp):
            return squeeze_time(self.dropout_layer(self.char_embeddings(inp)))

        self.c1 = _apply_embeddings(self.input1)
        self.c2 = _apply_embeddings(self.input2)
        self.c3 = _apply_embeddings(self.input3)
        self.c4 = _apply_embeddings(self.input4)
        self.c5 = _apply_embeddings(self.input5)
        self.c6 = _apply_embeddings(self.input6)
        self.neg_c1 = _apply_embeddings(self.neg_input1)
        self.neg_c2 = _apply_embeddings(self.neg_input2)
        self.neg_c3 = _apply_embeddings(self.neg_input3)
        self.neg_c4 = _apply_embeddings(self.neg_input4)
        self.neg_c5 = _apply_embeddings(self.neg_input5)
        self.neg_c6 = _apply_embeddings(self.neg_input6)

        # Create a function to compose pairs of chars into a new vector, which is in the same vector space (see later)
        self.composition2_layers = []
        for i, layer_size in enumerate(self.composition2_layer_sizes):
            # The last composition layer projects back into the embedding space, so is a linear layer
            layer_activation = "linear" if i == len(self.composition2_layer_sizes)-1 else "tanh"
            self.composition2_layers.append(Dense(layer_size,
                                                  activation=layer_activation,
                                                  name="compose2_%d" % i,
                                                  kernel_initializer='glorot_normal',
                                                  kernel_regularizer=l2(self.l2_reg)))

        # Apply the composition to the pairs of consecutive characters
        dropout2 = Dropout(self.composition_dropout, name="dropout2")
        def _apply_comp2(input1, input2, name):
            last_output = Concatenate(name="comp2-%s" % name)([input1, input2])
            for layer in self.composition2_layers:
                last_output = layer(last_output)
            last_output = dropout2(last_output)
            return last_output
        # Char2 x char3
        self.c2c3 = _apply_comp2(self.c2, self.c3, "c2xc3")
        # Char4 x char5
        self.c4c5 = _apply_comp2(self.c4, self.c5, "c4xc5")
        # ~char2 x ~char3
        self.neg_c2c3 = _apply_comp2(self.neg_c2, self.neg_c3, "nc2xnc3")
        self.neg_c4c5 = _apply_comp2(self.neg_c4, self.neg_c5, "nc4xnc5")

        # Create a function to compose triples of chars into a new vector, which is in the same vector space (see later)
        self.composition3_layers = []
        for i, layer_size in enumerate(self.composition3_layer_sizes):
            # The last composition layer projects back into the embedding space, so is a linear layer
            layer_activation = "linear" if i == len(self.composition2_layer_sizes)-1 else "tanh"
            self.composition3_layers.append(Dense(layer_size,
                                                  activation=layer_activation,
                                                  name="compose3_%d" % i,
                                                  kernel_initializer='glorot_normal',
                                                  kernel_regularizer=l2(self.l2_reg)))

        # Apply the triple composition to the two triples of consecutive characters
        dropout3 = Dropout(self.composition_dropout, name="dropout3")
        def _apply_comp3(input1, input2, input3, name):
            last_output = Concatenate(name="comp3-%s" % name)([input1, input2, input3])
            for layer in self.composition3_layers:
                last_output = layer(last_output)
            last_output = dropout3(last_output)
            return last_output
        # Char1 x char2 x char3
        self.c1c2c3 = _apply_comp3(self.c1, self.c2, self.c3, "c1xc2xc3")
        # Char4 x char5 x char6
        self.c4c5c6 = _apply_comp3(self.c4, self.c5, self.c6, "c4xc5xc6")
        # ~Char1 x ~char2 x ~char3
        self.neg_c1c2c3 = _apply_comp3(self.neg_c1, self.neg_c2, self.neg_c3, "nc1xnc2xnc3")
        self.neg_c4c5c6 = _apply_comp3(self.neg_c4, self.neg_c5, self.neg_c6, "nc4xnc5xnc6")

        # First we build the single predictor function, which we'll use for all
        self.predictor_layers = []
        for i, layer_size in enumerate(self.predictor_layer_sizes):
            # Linear final layer to produce the score for BPR
            layer_activation = "linear" if layer_size == 1 else "tanh"
            self.predictor_layers.append(Dense(layer_size, activation=layer_activation,
                                               name="predict_%d_%s" % (i, layer_activation),
                                               kernel_regularizer=l2(self.l2_reg)))

        def _pred(input1, input2, name):
            last_output = Concatenate(name="pred-%s" % name)([input1, input2])
            for layer in self.predictor_layers:
                last_output = layer(last_output)
            return last_output

        # Use this to weight the contribution of each prediction function output to the overall score
        def _scale(factor, node):
            scaler = Lambda(lambda x: x*factor, output_shape=lambda x: x)
            return scaler(node)

        def _mask_ctxt_sel_l(size, node):
            # Not sure why we need this extra dimension, but weird things happen otherwise
            sel = Lambda(lambda x: x*K.expand_dims(K.cast(self.context_selector_l[:, size-1], K.floatx()), -1), output_shape=lambda x: x)
            return sel(node)

        def _mask_ctxt_sel_r(size, node):
            sel = Lambda(lambda x: x*K.expand_dims(K.cast(self.context_selector_r[:, size-1], K.floatx()), -1), output_shape=lambda x: x)
            return sel(node)

        # Sum these, though only one will typically be unmasked
        self.pos_lhs = Add()([
            _mask_ctxt_sel_l(1, self.c3),
            _mask_ctxt_sel_l(2, self.c2c3),
            _mask_ctxt_sel_l(3, self.c1c2c3),
        ])
        self.neg_lhs = Add()([
            _mask_ctxt_sel_l(1, self.neg_c3),
            _mask_ctxt_sel_l(2, self.neg_c2c3),
            _mask_ctxt_sel_l(3, self.neg_c1c2c3),
        ])
        self.pos_rhs = Add()([
            _mask_ctxt_sel_r(1, self.c4),
            _mask_ctxt_sel_r(2, self.c4c5),
            _mask_ctxt_sel_r(3, self.c4c5c6),
        ])
        self.neg_rhs = Add()([
            _mask_ctxt_sel_r(1, self.neg_c4),
            _mask_ctxt_sel_r(2, self.neg_c4c5),
            _mask_ctxt_sel_r(3, self.neg_c4c5c6),
        ])

        self.positive_pred = _pred(self.pos_lhs, self.pos_rhs, "P-LxR")
        self.negative_pred = _pred(self.neg_lhs, self.neg_rhs, "P-nLxnR")

        # The model's score for this example is the sum of all the scores from the different combinations
        # minus the scores from the negative combinations
        # This gets fed into a sigmoid to produce the BPR objective
        self.bpr = BprMerge(name="bpr")
        self.score = self.bpr([self.positive_pred, self.negative_pred])

        # This doesn't really make sense, since the BPR score is really only for representation learning,
        #  but it has to do as a stand-in for a LM score if we need one
        self.test_score = K.sigmoid(self.positive_pred)

        self.loss = identity_loss
        self.inputs = [self.input1, self.input2, self.input3, self.input4, self.input5, self.input6,
                       self.neg_input1, self.neg_input2, self.neg_input3, self.neg_input4, self.neg_input5, self.neg_input6,
                       self.context_selector_l, self.context_selector_r]

        self.model = Model(inputs=self.inputs, outputs=self.score)

    def get_weights(self):
        return self.model.get_weights()

    def get_embeddings(self):
        return self.char_embeddings.get_weights()[0]

    def compile(self):
        self.model.compile(loss=identity_loss, optimizer=Adam())

    def get_trigram_score_function(self):
        func = K.function(
            inputs=[self.input1, self.input2, self.input3, K.learning_phase()],
            outputs=self.test_score
        )

        def _fn(*items):
            return func([items[0], items[1], items[2], 0])
        return _fn


def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)


class BprMerge(_Merge):
    """
    Compute the BPR objective given a linear prediction weight for the positive and negative cases.

    """
    def _merge_function(self, inputs):
        assert len(inputs) == 2
        return 1. - K.sigmoid(inputs[0] - inputs[1])


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


class MappedPairsRankCalculator(object):
    """
    Computes a metric that is the average rank by cosine distance of point B from point A,
    among all the points in the same vocabulary as B, given a set of (A, B)
    pairs of points. These pairs should be pairs of characters that, in an ideal
    model, we would expect to see mapped closely to one another.

    This measure can then be used as an evaluation metric for how well the learned
    embeddings recover the expected mappings.

    Each pair (A, B) should be an index within the first set of embeddings and the second
    repectively.

    """
    def __init__(self, overlapping_indices):
        self.overlapping_indices = overlapping_indices

    def compute(self, embeddingsA, embeddingsB):
        ranks_a_b = [
            float(
                numpy.where(
                    numpy.argsort(cosine_distances(embeddingsA[a].reshape(1, -1), embeddingsB))[0] == b
                )[0][0]
            ) / embeddingsB.shape[0]
            for (a, b) in self.overlapping_indices
        ]
        ranks_b_a = [
            float(
                numpy.where(
                    numpy.argsort(cosine_distances(embeddingsB[b].reshape(1, -1), embeddingsA))[0] == a
                )[0][0]
            ) / embeddingsA.shape[0]
            for (a, b) in self.overlapping_indices
        ]
        ranks = ranks_a_b + ranks_b_a
        mean_rank = sum(ranks) / len(ranks)
        return mean_rank
