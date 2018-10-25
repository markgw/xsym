# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import matplotlib
import os
import pickle
import pylab as plt
import shutil
from StringIO import StringIO
from collections import Counter
from copy import copy
from itertools import islice, cycle
from matplotlib.patches import Circle
from operator import itemgetter
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, manhattan_distances, sigmoid_kernel

import numpy
from keras import backend as K

from langsim.modules.local_lm.neural_sixgram2.model import NeuralSixgramModel as NeuralSixgramModel2
from langsim.modules.local_lm.neural_sixgram_samples.execute import CorpusIterator
from pimlico.core.modules.base import BaseModuleExecutor
from pimlico.utils.progress import get_progress_bar


# matplotlib.use("svg")


class ModuleExecutor(BaseModuleExecutor):
    def execute(self):
        output_path = self.info.get_absolute_output_dir("output")
        distance_metric_name = self.info.options["distance"]
        distance = distance_metric(distance_metric_name)
        min_token_prop = self.info.options["min_token_prop"]
        num_pairs = self.info.options["num_pairs"]
        lang_names = self.info.options["lang_names"]
        # We could have multiple langs with the same name, which we do sometimes for particular experiments
        old_lang_names = copy(lang_names)
        for i, ln in enumerate(lang_names):
            if old_lang_names.count(ln) > 1:
                lang_names[i] = "%s%d" % (ln, old_lang_names[:i].count(ln))
        self.log.info("Input languages: %s" % ", ".join(lang_names))

        # How big a sample to use to get bigram statistics
        # Could set this as a parameter, but I don't think I'd ever want to change it
        frequent_pair_sample = 100000

        vocabs = [v.get_data() for v in self.info.get_input("vocabs", always_list=True)]
        vocab_sizes = [len(v) for v in vocabs]
        self.log.info("Vocab sizes: %s" % ", ".join(str(s) for s in vocab_sizes))

        # We're going to need a sample of data from the training set to get bigram statistics
        corpora = self.info.get_input("corpora", always_list=True)
        sample_iterator = CorpusIterator(corpora, vocab_sizes)

        frequency_arrays = self.info.get_input("frequencies", always_list=True)
        if len(frequency_arrays) != len(vocabs):
            raise ValueError("you must specify one frequency array per corpus")
        frequency_arrays = [a.array for a in frequency_arrays]
        # Normalize the frequency arrays
        frequency_arrays = [a.astype(numpy.float32) / a.sum() for a in frequency_arrays]

        single_only = False

        # Used to use the same plotting module for different model types
        model = self.info.get_input("model").load_model()
        if type(model) is not NeuralSixgramModel2:
            raise TypeError("invalid model type for this plotting module: %s. It now only supports NeuralSixgramModel2"
                            % type(model).__name__)

        # This works for all model types
        single_embed = model.get_embeddings()

        # Infer whether the model was trained with a vocab that includes OOV (more recent models) or whether OOV
        # was represented by an extra index
        oov_included = single_embed.shape[0] == sum(vocab_sizes)
        extra_indices = 0 if oov_included else 1

        # Prepare output dir
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

        # Select which names we've going to allow into our vocabulary for visualization
        allowed_names = [
            # Apply proportional frequency cutoff
            [vocab[i] for i in vocab.id2token.keys() if freqs[i] >= min_token_prop]
            for vocab, freqs in zip(vocabs, frequency_arrays)
        ]

        index_shifts = reduce(lambda c, x: c + [c[-1] + x + extra_indices], vocab_sizes, [0])[:-1]
        vocab_lookup = dict(
            (id+shift, (name, lang_num))
            for lang_num, (vocab, shift) in enumerate(zip(vocabs, index_shifts))
            for (id, name) in vocab.id2token.items() + ([(len(vocab), "OOV")] if not oov_included else [])
        )
        all_names = [(name, lang_num) for (id, (name, lang_num)) in sorted(vocab_lookup.items(), key=itemgetter(0))]

        ### Some convenience functions for converting between vocab indices and names to be used in visualization
        def _plain_char_symbol(char):
            return char if char != " " else "\u21A6"

        def _char_symbol(char, lang_num):
            return "%s:%s" % (lang_names[lang_num], _plain_char_symbol(char))

        def _char_ngram_symbol(lang_num, *chars):
            return "%s:%s" % (
                lang_names[lang_num],
                "".join(_plain_char_symbol(c) for c in chars)
            )

        def _indexed_char_ngram_symbol(*ids):
            chars, langs = zip(*(vocab_lookup[i] for i in ids))
            assert all(l == langs[0] for l in langs[1:])
            return _char_ngram_symbol(langs[0], *chars)

        def _index_to_lang_num(idx):
            return vocab_lookup[idx][1]

        # Filter out rows corresponding to disallowed names
        allowed_ids, names = zip(*(
            (id, (name, lang_num)) for (id, (name, lang_num)) in vocab_lookup.items()
            if name in allowed_names[lang_num]
        ))
        names = list(names)
        single_embed = single_embed[allowed_ids, :]

        self.log.info("Filtered out %d chars from vocab: %s" % (len(all_names)-len(names),
                                                                ", ".join(
                                                                    _char_symbol(char, lang_num)
                                                                    for (char, lang_num) in all_names
                                                                    if (char, lang_num) not in names)
                                                                ))

        lang_ids = [lang_num for (char, lang_num) in names]
        names = [_char_symbol(char, lang_num) for (char, lang_num) in names]
        single_names = copy(names)

        # Now we want to add the learned compositions of chars into our vector space
        # We've unfortunately not yet kept a count of frequent ngrams
        # Do a quick count now to find some frequent ones to try out
        self.log.info(
            "Couting most frequent ngrams in a small sample (%d) from the training corpus" % frequent_pair_sample
        )

        if single_only:
            embed = single_embed
        else:
            lhs_function = K.function(
                [model.input2, model.input3, K.learning_phase()],
                [model.c2c3],
            )

            pair_counter = Counter(
                islice((tuple(ngram[:2]) for (corpus_num, ngram) in sample_iterator.iter_interleaved_ngrams(3)), frequent_pair_sample)
            )
            # Just include the most frequent num_pairs char pairs
            top_pairs_ids = [pair for (pair, count) in pair_counter.most_common(num_pairs)]
            top_pairs_names = [_indexed_char_ngram_symbol(a, b) for (a, b) in top_pairs_ids]
            top_pairs_langs = [_index_to_lang_num(a) for (a, b) in top_pairs_ids]
            self.log.info("Most frequent pairs: %s" % ", ".join(top_pairs_names))

            # Also include some trigrams
            triple_counter = Counter(
                islice((tuple(ngram[:3]) for (corpus_num, ngram) in sample_iterator.iter_interleaved_ngrams(3)), frequent_pair_sample)
            )
            # Just include the most frequent num_pairs char pairs
            # Also apply a cutoff to avoid super-low counts
            top_trigrams_ids = [triple for (triple, count) in triple_counter.most_common(num_pairs*5) if count > 100]
            top_trigrams_names = [_indexed_char_ngram_symbol(a, b, c) for (a, b, c) in top_trigrams_ids]
            top_trigrams_langs = [_index_to_lang_num(a) for (a, b, c) in top_trigrams_ids]
            self.log.info("Most frequent trigrams: %s" % ", ".join(top_trigrams_names))

            # Now we need a function for projecting the ngrams into the embedding space
            # We actually have two: those from the LHS of a single char in the LM and those from the RHS
            pair_vectors = lhs_function([
                numpy.array([a for (a, b) in top_pairs_ids])[:, numpy.newaxis],
                numpy.array([b for (a, b) in top_pairs_ids])[:, numpy.newaxis],
                0
            ])[0]
            embed_with_pairs = numpy.vstack((single_embed, pair_vectors))
            names.extend(top_pairs_names)
            lang_ids.extend(top_pairs_langs)
            embed = embed_with_pairs

            if len(top_trigrams_ids):
                # Same for trigrams
                lhs3_function = K.function(
                    [model.input1, model.input2, model.input3, K.learning_phase()],
                    [model.c1c2c3],
                )
                trigram_vectors = lhs3_function([
                    numpy.array([a for (a, b, c) in top_trigrams_ids])[:, numpy.newaxis],
                    numpy.array([b for (a, b, c) in top_trigrams_ids])[:, numpy.newaxis],
                    numpy.array([c for (a, b, c) in top_trigrams_ids])[:, numpy.newaxis],
                    0
                ])[0]

                # We only include here the top num_pairs, but we've also projected more, so we can plot them later
                embed = numpy.vstack((embed_with_pairs, trigram_vectors[:num_pairs]))
                names.extend(top_trigrams_names[:num_pairs])
                lang_ids.extend(top_trigrams_langs[:num_pairs])

        # Use a font the supports all the IPA chars we need
        #matplotlib.rc('font', family="Gentium", size=10)
        matplotlib.rc('font', family="DejaVu Sans", size=10)

        # Prepare functions for doing plotting
        colors = "bgrcmyb"
        if len(colors) < len(lang_names):
            self.log.warn("Not got enough matplotlib colors for %d different languages, reusing some")
        lang_colours = dict(zip(lang_names, cycle(colors)))

        def point_colour(name):
            lang_name = name.partition(":")[0]
            return lang_colours[lang_name]

        def point_name(name):
            # Strip the language marker
            return name.partition(":")[2]


        # Calculate all pairwise distances
        self.log.info("Calculating pairwise distances (%s)" % distance_metric_name)
        dists = distance(embed)

        # We might very well have zero distances, so add a tiny value to everything apart from the diagonal
        if any((x != y) for (x, y) in zip(*numpy.where(dists == 0.))):
            dists[:, :] += 0.00001
            numpy.fill_diagonal(dists, 0.)

        top_sims = numpy.unravel_index(numpy.argsort(dists.flatten()), dists.shape)
        top_non_i_sims = [(x, y) for (x, y) in zip(*top_sims) if x > y]

        self.log.info("Top similarities")
        for x, y in top_non_i_sims[:20]:
            self.log.info("%s -- %s" % (names[x], names[y]))

        # Output the most similar pairs across languages
        cross_lingual_tops_file = os.path.join(output_path, "cross_lingual_tops.txt")
        self.log.info("Outputting top cross-lingual similarities to %s" % cross_lingual_tops_file)
        buff = StringIO()

        print >>buff, "Most similar cross-lingual pairs\n(Showing %s distances)\n" % distance_metric_name
        for x, y in top_non_i_sims:
            if lang_ids[x] != lang_ids[y]:
                # Order consistently to make it easier to read
                pair_names = list(sorted([names[x], names[y]]))
                print >>buff, "%s  %s  %.3f" % (pair_names[0], pair_names[1], dists[x, y])

        with open(cross_lingual_tops_file, "w") as f:
            f.write(buff.getvalue().encode("utf8"))


        if not single_only:
            # Do the same thing for individual phonemes
            self.log.info("Calculating pairwise distances between phonemes (%s)" % distance_metric_name)
            single_dists = distance(single_embed)

            if any((x != y) for (x, y) in zip(*numpy.where(single_dists == 0.))):
                single_dists[:, :] += 0.00001
                numpy.fill_diagonal(single_dists, 0.)

            single_top_sims = numpy.unravel_index(numpy.argsort(single_dists.flatten()), single_dists.shape)
            single_top_non_i_sims = [(x, y) for (x, y) in zip(*single_top_sims) if x > y]

            self.log.info("Top phoneme similarities")
            for x, y in single_top_non_i_sims[:20]:
                self.log.info("%s -- %s" % (single_names[x], single_names[y]))

            # Output the most similar pairs across languages
            cross_lingual_tops_single_file = os.path.join(output_path, "cross_lingual_tops_single.txt")
            self.log.info("Outputting top cross-lingual similarities to %s" % cross_lingual_tops_single_file)
            buff = StringIO()

            print >>buff, "Most similar cross-lingual single-phoneme pairs\n(Showing %s distances)\n" % distance_metric_name
            for x, y in single_top_non_i_sims:
                if lang_ids[x] != lang_ids[y]:
                    # Order consistently to make it easier to read
                    pair_names = list(sorted([single_names[x], single_names[y]]))
                    print >>buff, "%s  %s  %.3f" % (pair_names[0], pair_names[1], dists[x, y])

            with open(cross_lingual_tops_single_file, "w") as f:
                f.write(buff.getvalue().encode("utf8"))

        # Allow the pickling output to be disabled if it fails, so we don't keep trying and failing on every one
        self.pickle_plots = True

        # Define a plotting function, which we'll use many times over below
        def plot(coords, point_names, title, output_path=None, additional_plotting=None, code=False):
            fig, ax = plt.subplots()
            for (x, y), name in zip(coords, point_names):
                ax.annotate(point_name(name), (x, y), color=point_colour(name))
            plt.xlim((1.1*coords[:, 0].min(), 1.1*coords[:, 0].max()))
            plt.ylim((1.1*coords[:, 1].min(), 1.1*coords[:, 1].max()))
            plt.title(title)

            if additional_plotting is not None:
                additional_plotting(fig, ax, coords)

            if output_path is not None:
                plt.savefig(output_path)
                if self.pickle_plots:
                    try:
                        # Also save the plot as a pickle, so we can reload the interactive plot
                        pickle_filename = "%s.pkl" % output_path.rpartition(".")[0]
                        with open(pickle_filename, "w") as f:
                            pickle.dump(ax, f)
                    except Exception, e:
                        self.log.error("Could not pickle chart: %s" % e)
                        self.log.error("Not trying to pickle the rest of the charts")
                        self.log.error("Note that you need matplotlib >=2 for this to work, so perhaps you need to upgrade?")
                        self.pickle_plots = False

                if code:
                    # Also output the code for doing this plotting, so we can rerun it and tweak for papers, etc
                    # Output the coordinates to a file
                    coords_path = "%s.npy" % output_path.rpartition(".")[0]
                    numpy.save(coords_path, coords)
                    # Output point names
                    names_path = "%s_names.txt" % output_path.rpartition(".")[0]
                    with open(names_path, "w") as f:
                        f.write(u"\n".join(point_names).encode("utf8"))
                    # Output the code
                    code_path = "%s.py" % output_path.rpartition(".")[0]
                    with open(code_path, "w") as f:
                        f.write(plotting_code.format(
                            coords_path=os.path.basename(coords_path),
                            names_path=os.path.basename(names_path),
                            lang_names=", ".join('"{}"'.format(l) for l in lang_names),
                            title=title,
                            output_path=os.path.basename(output_path),
                        ).encode("utf8"))
            plt.close()

        # Also a shortcut for computing distances, running MDS and plotting
        def plot_mds(vectors, names, title, path, additional_plotting=None, code=False):
            # Compute the pairwise distances between all the vectors
            dists = distance(vectors)
            # This should always result in a symmetric matrix, but sometimes doesn't due to precision errors
            # Sklearn's checks are too stringent, so sometimes fail for this reason
            # Solve by just averaging
            dists = (dists + dists.T) / 2.
            # Initialize MDS
            mds = MDS(n_components=2, random_state=1234, n_init=1, dissimilarity="precomputed")
            # Fit an MDS reduction to the computed distance matrix
            mds.fit(dists.astype(numpy.float64))
            # Plot the computed reduction in 2D space
            return plot(mds.embedding_, names, title, output_path=path, additional_plotting=additional_plotting, code=code), mds

        langs_and_colours = " and ".join("%s (%s)" % (l, lang_colours[l]) for l in lang_names)

        ### MDS
        mds_path = os.path.join(output_path, "mds.svg")
        self.log.info("Running MDS reduction")
        plot_mds(embed, names, "MDS of %s phones, %s distances" % (langs_and_colours, distance_metric_name), mds_path,
                 code=True)

        if not single_only:
            # Also plot single-phoneme only
            mds_single_path = os.path.join(output_path, "mds_single.svg")
            self.log.info("Running MDS reduction (single-phoneme only)")
            plot_mds(single_embed, names,
                     "MDS of %s single phones, %s distances" % (langs_and_colours, distance_metric_name),
                     mds_single_path, code=True)

        # Produce a plot each character individually, together with the nearest neighbours, so that the
        # projection is more accurate for the small region
        self.log.info("Target-by-target region visualisations")
        phoneme_reg_plots_dir = os.path.join(output_path, "phoneme_regions")
        os.mkdir(phoneme_reg_plots_dir)
        pbar = get_progress_bar(len(names), title="Reducing and plotting")
        for i in pbar(range(len(names))):
            name = names[i]
            neighbourhood = numpy.argsort(distance(embed[i].reshape(1, -1), embed))[0][:10]
            key_idx = numpy.where(neighbourhood == i)[0][0]
            char_mds_path = os.path.join(phoneme_reg_plots_dir, "mds_%s.svg" % name.replace("/", "_SLASH_"))

            def _plot_circle(fig, ax, coords):
                # Add a circle around the key point
                p = Circle(coords[key_idx], 0.05, edgecolor="r", facecolor=(0, 0, 0, 0))
                ax.add_patch(p)

            plot_mds(
                embed[neighbourhood], [names[j] for j in neighbourhood],
                "%s and its nearest neighbours, %s distances" % (name, distance_metric_name),
                char_mds_path, additional_plotting=_plot_circle)


        # Produce a 1D plot for each character individually, showing its similarity to its nearest neighbours
        self.log.info("Target-by-target nearest neighbour distributions")
        phoneme_neighbours_dir = os.path.join(output_path, "phoneme_neighbours")
        os.mkdir(phoneme_neighbours_dir)
        pbar = get_progress_bar(len(names), title="Plotting")
        n_neighbours = 20
        for i in pbar(range(len(names))):
            name = names[i]
            distances = distance(embed[i].reshape(1, -1), embed)[0]
            neighbourhood = numpy.argsort(distances)[:n_neighbours]
            key_idx = numpy.where(neighbourhood == i)[0][0]

            min_sep = distances[neighbourhood].max() / 50.
            max_y = 0.

            fig, ax = plt.subplots()
            fig.set_size_inches(10, 1)
            ax.axes.get_yaxis().set_visible(False)
            # Go left to right over the values
            last_val0 = None
            current_y = 0.
            for neighbour in neighbourhood:
                neighbour_dist = distances[neighbour]
                neighbour_name = names[neighbour]

                if last_val0 is not None and neighbour_dist - last_val0 < min_sep:
                    # Push this char up, so it isn't too close to the previous
                    current_y += 0.02
                    max_y = max(max_y, current_y)
                else:
                    last_val0 = neighbour_dist
                    current_y = 0.
                ax.annotate(point_name(neighbour_name), (neighbour_dist, current_y), color=point_colour(neighbour_name))

            plt.xlim((0., 1.1*distances[neighbourhood].max()))
            plt.ylim((-0.05, max_y+0.05))
            plt.title("Nearest neighbours to {}".format(name))
            plt.savefig(os.path.join(phoneme_neighbours_dir, "neighbours_{}.svg".format(_name_to_filename(name))), bbox_inches="tight")
            plt.close()


        # Analyse each dimension by plotting the spread of points along just that dimension
        # This gives us some idea of whether we've learned individually meaningful dimensions
        self.log.info("Plotting spread of values on each individual dimension")
        dim_plot_dir = os.path.join(output_path, "dimensions")
        os.mkdir(dim_plot_dir)
        pbar = get_progress_bar(embed.shape[1], title="Plotting")
        # Set a smaller font size for this plot, so we can see the chars separately
        matplotlib.rc('font', size=4)
        for dim, values in pbar(enumerate(embed.T)):
            min_sep = (values.max() - values.min()) / 80.
            max_y = 0.

            fig, ax = plt.subplots()
            fig.set_size_inches(10, 1)
            ax.axes.get_yaxis().set_visible(False)
            # Go left to right over the values
            last_val0 = None
            current_y = 0.
            for x, name in sorted(zip(values, names), key=itemgetter(0)):
                if last_val0 is not None and x - last_val0 < min_sep:
                    # Push this char up, so it isn't too close to the previous
                    current_y += 0.003
                    max_y = max(max_y, current_y)
                else:
                    last_val0 = x
                    current_y = 0.
                ax.annotate(point_name(name), (x, current_y), color=point_colour(name))

            plt.xlim((1.1*values.min(), 1.1*values.max()))
            plt.ylim((-0.01, max_y+0.01))
            plt.title("Dimension %d" % dim)
            plt.savefig(os.path.join(dim_plot_dir, "dimension_%03d.svg" % dim))
            plt.close()

        matplotlib.rc('font', size=10)

        if len(vocabs) != 2:
            self.log.info("Skipping various analyses that only make sense for two languages, since we have %d "
                          "languages to compare" % len(vocabs))
        else:
            # Output all pairwise similarities
            # Split up the languages' vocabularies and embeddings again
            lang0_embed = embed[numpy.array(lang_ids) == 0]
            lang1_embed = embed[numpy.array(lang_ids) == 1]
            lang0_names = [name for (name, lang) in zip(names, lang_ids) if lang == 0]
            lang1_names = [name for (name, lang) in zip(names, lang_ids) if lang == 1]

            for src, trg, src_names, src_embed, trg_names, trg_embed in [
                (lang_names[0], lang_names[1], lang0_names, lang0_embed, lang1_names, lang1_embed),
                (lang_names[1], lang_names[0], lang1_names, lang1_embed, lang0_names, lang0_embed),
            ]:
                cross_lingual_sim_file = os.path.join(output_path, "cross_lingual_sims_%s-%s.txt" % (src, trg))
                self.log.info("Outputting %s->%s cross-lingual similarities to %s (%s)" %
                              (src, trg, cross_lingual_sim_file, distance_metric_name))
                buff = StringIO()
                print >>buff, "= %s -> %s comparison = (%s)" % (src, trg, distance_metric_name)
                trg_names_no_lang = [name.partition(":")[2] for name in trg_names]

                for i in range(len(src_names)):
                    trg_dists = distance(src_embed[i].reshape(1, -1), trg_embed)
                    src_name = src_names[i]
                    src_name_no_lang = src_name.partition(":")[2]
                    print >>buff, "\nMost similar %s chars to %s%s" % (
                        trg, src_name,
                        " (absent in %s)" % trg if src_name_no_lang not in trg_names_no_lang else "",
                    )

                    for j in numpy.argsort(trg_dists)[0, :10]:
                        trg_name = trg_names[j]
                        print >>buff, "%s (%.2f)%s" % (
                            trg_name,
                            trg_dists[0, j],
                            " *" if trg_name.partition(":")[2] == src_name.partition(":")[2] else "",
                        )

                with open(cross_lingual_sim_file, "w") as f:
                    f.write(buff.getvalue().encode("utf8"))

            ######################## PLOTS ##################
            # Produce a plot each character individually, together with all of the other language's characters
            self.log.info("Target-by-target visualisations")
            phoneme_plots_dir = os.path.join(output_path, "phonemes")
            os.mkdir(phoneme_plots_dir)
            for src, trg, src_names, src_embed, trg_names, trg_embed in [
                (lang_names[0], lang_names[1], lang0_names, lang0_embed, lang1_names, lang1_embed),
                (lang_names[1], lang_names[0], lang1_names, lang1_embed, lang0_names, lang0_embed),
            ]:
                self.log.info("%s -> %s comparison (%s)" % (src, trg, distance_metric_name))
                pbar = get_progress_bar(len(src_names), title="Reducing and plotting")
                for i in pbar(range(len(src_names))):
                    char_mds_path = os.path.join(phoneme_plots_dir, "mds_%s.svg" % _name_to_filename(src_names[i]))
                    plot_mds(numpy.vstack((src_embed[i], trg_embed)),
                             [src_names[i]] + trg_names,
                             "%s and all %s phones, %s distances" % (src_names[i], trg, distance_metric_name),
                             char_mds_path)

        self.log.info("All visualizations and analyses output to %s" % output_path)


def distance_metric(name):
    def _dist_metric(xs, ys=None):
        if ys is None:
            ys = xs

        if name == "dot":
            # Don't think this makes any sense as a distance
            d = -numpy.dot(xs, ys.T)
            d -= d.min()
        elif name == "eucl":
            d = euclidean_distances(xs, ys)
        elif name == "cos":
            # Kind of reasonable
            d = cosine_distances(xs, ys)
        elif name == "man":
            # Terrible results
            d = manhattan_distances(xs, ys)
        elif name == "sig_kern":
            # This isn't too bad
            d = -sigmoid_kernel(xs, ys)
            d -= d.min()
        else:
            raise ValueError("unknown distance type %s" % name)
        return d
    return _dist_metric


def _name_to_filename(name):
    """ Sanitize a name so it can be used as a filename """
    return name.replace("/", "SLASH")


plotting_code = """
import numpy
import matplotlib
import pylab as plt
matplotlib.rc('font', family="DejaVu Sans", size=10)


colors = "bgrcmyb"
lang_colours = dict(zip([{lang_names}], colors))

def point_colour(name):
    lang_name = name.partition(":")[0]
    return lang_colours[lang_name]

def point_name(name):
    # Strip the language marker
    return name.partition(":")[2]


# Load the stored coordinates
coords = numpy.load("{coords_path}")
# Load the point names
with open("{names_path}", "r") as f:
    point_names = f.read().decode("utf8").splitlines()

fig, ax = plt.subplots()
for (x, y), name in zip(coords, point_names):
    ax.annotate(point_name(name), (x, y), color=point_colour(name))

plt.xlim((1.1*coords[:, 0].min(), 1.1*coords[:, 0].max()))
plt.ylim((1.1*coords[:, 1].min(), 1.1*coords[:, 1].max()))

# Don't show labels on axes
ax.set_yticklabels([])
ax.set_xticklabels([])

#plt.title("{title}")

plt.savefig("{output_path}", bbox_inches='tight')
plt.close()

"""
