# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
import shutil
from cStringIO import StringIO
from copy import copy
from itertools import takewhile
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_distances

import numpy

from langsim.modules.local_lm.neural_sixgram2.model import NeuralSixgramModel as NeuralSixgramModel2
from pimlico.core.modules.base import BaseModuleExecutor
from pimlico.core.modules.execute import ModuleExecutionError
from pimlico.datatypes.files import NamedFileWriter


class ModuleExecutor(BaseModuleExecutor):
    def execute(self):
        output_path = self.info.get_absolute_output_dir("analysis")

        # Allow this to be changed easily
        distance = cosine_distances
        distance_metric_name = "cos"
        dist_to_sim = lambda x: 1. - x

        oov_token = self.info.options["oov"]
        min_token_prop = self.info.options["min_token_prop"]
        lang_names = self.info.options["lang_names"]
        # We could have multiple langs with the same name, which we do sometimes for particular experiments
        old_lang_names = copy(lang_names)
        for i, ln in enumerate(lang_names):
            if old_lang_names.count(ln) > 1:
                lang_names[i] = "%s%d" % (ln, old_lang_names[:i].count(ln))
        self.log.info("Input languages: %s" % ", ".join(lang_names))

        vocabs = [v.get_data() for v in self.info.get_input("vocabs", always_list=True)]
        vocab_sizes = [len(v) for v in vocabs]
        self.log.info("Vocab sizes: %s" % ", ".join(str(s) for s in vocab_sizes))

        if len(vocabs) != 2:
            raise ModuleExecutionError("expected exactly two languages, got {} vocabs".format(len(vocabs)))

        frequency_arrays = self.info.get_input("frequencies", always_list=True)
        if len(frequency_arrays) != len(vocabs):
            raise ValueError("you must specify one frequency array per corpus")
        frequency_arrays = [a.array for a in frequency_arrays]

        # This is only for one model type
        model = self.info.get_input("model").load_model()
        # Might also want to allow NeuralSixgramModel
        if type(model) is not NeuralSixgramModel2:
            raise TypeError("embedding analysis module currently only for NeuralSixgramModel2: got %s" %
                            type(model).__name__)

        single_embed = model.get_embeddings()

        # Infer whether the model was trained with a vocab that includes OOV (more recent models) or whether OOV
        # was represented by an extra index
        oov_included = single_embed.shape[0] == sum(vocab_sizes)
        extra_indices = 0 if oov_included else 1

        # Prepare output dir
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

        self.log.info("Filtering out characters with frequency lower than {}%".format(min_token_prop*100.))
        # Normalize the frequency arrays
        frequency_arrays = [a.astype(numpy.float32) / a.sum() for a in frequency_arrays]
        if oov_token is not None:
            # Don't filter out OOV
            let_through = lambda i, voc:  voc.id2token[i] == oov_token
        else:
            let_through = lambda i, voc: False
        # Select which names we've going to allow into our vocabulary for visualization
        allowed_lang_ids = [
            # Apply proportional frequency cutoff
            [i for (i, freq) in enumerate(freqs) if let_through(i, vocab) or freq >= min_token_prop]
            for vocab, freqs in zip(vocabs, frequency_arrays)
        ]
        allowed_names = [[vocab.id2token[i] for i in lang_ids] for vocab, lang_ids in zip(vocabs, allowed_lang_ids)]

        index_shifts = reduce(lambda c, x: c + [c[-1] + x + extra_indices], vocab_sizes, [0])[:-1]
        allowed_ids = [
            i+shift for (lang_ids, shift) in zip(allowed_lang_ids, index_shifts) for i in lang_ids
        ]
        lang_ids = sum([[lang_id]*len(char_ids) for lang_id, char_ids in enumerate(allowed_lang_ids)], [])
        vocab_lookup = dict(
            (id+shift, (name, lang_num))
            for lang_num, (vocab, shift) in enumerate(zip(vocabs, index_shifts))
            for (id, name) in vocab.id2token.items() + ([(len(vocab), "OOV")] if not oov_included else [])
        )
        all_names = [(name, lang_num) for (id, (name, lang_num)) in sorted(vocab_lookup.items(), key=itemgetter(0))]
        names = [(name, lang) for lang, _lang_names in enumerate(allowed_names) for name in _lang_names]
        single_embed = single_embed[allowed_ids, :]

        def _plain_char_symbol(char):
            # Replace spaces so we can see them
            return char if char != " " else "\u21A6"

        def _char_symbol(char, lang_num):
            # Char with lang ID
            return "%s:%s" % (lang_names[lang_num], _plain_char_symbol(char))

        self.log.info("Filtered out %d chars from vocab: %s" % (len(all_names)-len(names),
                                                                ", ".join(
                                                                    _char_symbol(char, lang_num)
                                                                    for (char, lang_num) in all_names
                                                                    if (char, lang_num) not in names)
                                                                ))

        # These names are specific to the language and don't include the lang identifier
        names0 = [char for (char, lang_num) in names if lang_num == 0]
        names1 = [char for (char, lang_num) in names if lang_num == 1]

        # Get separate matrices of the two languages' embeddings
        embed0 = numpy.copy(single_embed[[i for (i, lang) in enumerate(lang_ids) if lang == 0]])
        embed1 = numpy.copy(single_embed[[i for (i, lang) in enumerate(lang_ids) if lang == 1]])

        self.log.info("Num chars in lang 1 after filters: {}".format(embed0.shape[0]))
        self.log.info("Num chars in lang 2 after filters: {}".format(embed1.shape[0]))

        # For now we only include single-character embeddings here, but we could also retrieve projections
        # of n-grams, as in the plot module

        # Look up all the characters that are in both languages
        names0_ids = dict((name, i) for (i, name) in enumerate(names0))
        names1_ids = dict((name, i) for (i, name) in enumerate(names1))
        overlapping_pairs = [
            (names0_ids[name0], names1_ids[name0]) for name0 in names0 if name0 in names1_ids
        ]
        self.log.info("Computing stats over {} overlapping pairs".format(len(overlapping_pairs)))
        # Compute the distance of every lang0 embedding to every lang1 embedding
        distances = distance(embed0, embed1)
        # Transform this to get the similarity, for output purposes
        similarities = dist_to_sim(distances)
        distance_ranks_f = numpy.argsort(distances, axis=1)
        # 1-indexed rank (i.e. 1 is the closest)
        matching_ranks_f = [numpy.where(distance_ranks_f[i] == j)[0] + 1 for (i, j) in overlapping_pairs]

        # Do the same thing in the opposite direction
        distance_ranks_r = numpy.argsort(distances, axis=0)
        matching_ranks_r = [numpy.where(distance_ranks_r[:, j] == i)[0] + 1 for (i, j) in overlapping_pairs]

        matching_ranks = matching_ranks_r + matching_ranks_f

        # Compute median and mean
        median_matching_rank = numpy.median(matching_ranks)
        mean_matching_rank = numpy.mean(matching_ranks)
        self.log.info("Median rank of overlapping chars: {}".format(median_matching_rank))
        self.log.info("Mean rank of overlapping chars: {}".format(mean_matching_rank))

        # Compute R@N for 1 and 5
        recall_at_1 = float(sum(1 for rank in matching_ranks if rank == 1)) / len(matching_ranks)
        recall_at_3 = float(sum(1 for rank in matching_ranks if rank <= 3)) / len(matching_ranks)
        recall_at_5 = float(sum(1 for rank in matching_ranks if rank <= 5)) / len(matching_ranks)
        self.log.info("R@1 for overlapping pairs: {}".format(recall_at_1))
        self.log.info("R@3 for overlapping pairs: {}".format(recall_at_3))
        self.log.info("R@5 for overlapping pairs: {}".format(recall_at_5))

        # Output the values of the correlation coefficients
        with NamedFileWriter(self.info.get_absolute_output_dir("analysis"),
                             self.info.get_output("analysis").filename) as writer:
            writer.write_data(
                "Min frequency: {}%\n"
                "Num overlapping chars: {}\n"
                "Num chars lang 1: {}\n"
                "Num chars lang 2: {}"
                "Median rank of overlapping chars: {}\n"
                "Mean rank of overlapping chars: {}\n"
                "Recall@1 for overlapping pairs: {}\n"
                "Recall@3 for overlapping pairs: {}\n"
                "Recall@5 for overlapping pairs: {}\n".format(
                    min_token_prop*100., len(overlapping_pairs),
                    int(embed0.shape[0]), int(embed1.shape[0]),
                    median_matching_rank, mean_matching_rank,
                    recall_at_1, recall_at_3, recall_at_5
                )
            )

        pair_output = StringIO()
        def _file_out(s=""):
            s = s.encode("utf8")
            print >> pair_output, s

        def _out(s=""):
            _file_out(s)
            s = s.encode("utf8")
            # Print blank lines, but don't log
            if s:
                self.log.info(s)

        # We can look at the nearest pairs across languages, skipping over identical symbols
        sorted_xs, sorted_ys = numpy.unravel_index(distances.argsort(axis=None), distances.shape)
        nearest_pairs = [(names0[x], names1[y], distances[x, y]) for (x, y) in zip(sorted_xs, sorted_ys)]
        nearest_pairs = [(x, y, dist) for (x, y, dist) in nearest_pairs if x != y]
        _out("Nearest non-identical neighbours: {}".format(
            ", ".join("{} {} ({:.2f})".format(x, y, dist) for (x, y, dist) in nearest_pairs[:50])
        ))

        # We can also look at the nearest neighbours to each character that aren't the same character
        def _non_identical_nbrs(y_ids, y_names, x_name):
            if x_name not in y_names:
                # X not in y-names, so don't show all y names
                for i in range(3):
                    yield y_names[y_ids[i]]
                yield "..."
            else:
                for y in y_ids:
                    yield y_names[y]
                    if y_names[y] == x_name:
                        return

        _out()
        _out("Neighbours from {} nearer to each in {} than identical char".format(lang_names[1], lang_names[0]))
        for x, nearest in enumerate(distance_ranks_f):
            # Skip over ones where the nearest neighbour is the identical char
            if names0[x] != names1[nearest[0]]:
                # Indicate clearly where the x char isn't in y at all
                x_name = "{}{}".format(names0[x], "" if names0[x] in names1 else "*")
                _out("{}: {}".format(
                    x_name, " ".join(_non_identical_nbrs(nearest, names1, names0[x]))
                ))

        _out()
        _out("Neighbours from {} nearer to each in {} than identical char".format(lang_names[0], lang_names[1]))
        for x, nearest in enumerate(distance_ranks_r.T):
            # Skip over ones where the nearest neighbour is the identical char
            x_name = "{}{}".format(names1[x], "" if names1[x] in names0 else "*")
            if names1[x] != names0[nearest[0]]:
                _out("{}: {}".format(
                    x_name, " ".join(_non_identical_nbrs(nearest, names0, names1[x]))
                ))

        # Now we do the same again, but for the ones where the identical char *is* the NN
        # We show the closest 5 neighbours
        num_neighbours = 5

        _out(), _out()
        _out("Neighbours of {} chars from {}, where nearest is identical".format(lang_names[0], lang_names[1]))
        neighbour_table0 = []
        for x, nearest in enumerate(distance_ranks_f):
            # Look at ones where the nearest neighbour is the identical char
            if names0[x] == names1[nearest[0]]:
                x_name = names0[x]
                y_names = [(names1[y], similarities[x, y])
                           for y in takewhile(lambda _y: similarities[x, _y] > 0.5, nearest)]
                # Put a row in a table that will produce a Latex version
                neighbour_table0.append([x_name] + y_names)
                # Also output a line
                _out("{}: {}".format(
                    x_name, " ".join("{} ({:.2f})".format(nm, sim) for (nm, sim) in y_names)
                ))

        _out()
        _out("Neighbours of {} chars from {}, where nearest is identical".format(lang_names[1], lang_names[0]))
        neighbour_table1 = []
        for x, nearest in enumerate(distance_ranks_r.T):
            # Look at ones where the nearest neighbour is the identical char
            if names1[x] == names0[nearest[0]]:
                x_name = names1[x]
                y_names = [(names0[y], similarities[y, x])
                           for y in takewhile(lambda _y: similarities[_y, x] > 0.5, nearest)]
                neighbour_table1.append([x_name] + y_names)
                _out("{}: {}".format(
                    x_name, " ".join("{} ({:.2f})".format(nm, sim) for (nm, sim) in y_names)
                ))

        # Produce a Latex version of these tables
        self.log.info("Outputting Latex tables to pairs file")
        _file_out(), _file_out("Latex tables")
        def _latex_name(s):
            if s == " ":
                return "$\\mapsto$"
            else:
                return s
        def _latex_col(sim):
            # Colour (grey) based on strength of similarity
            # Scale so s>0.75 -> 1, 0.5>s>0.75 -> 0.5-1
            #strength = int(100 * (min(1.0, ((sim - 0.5) / 0.25) * 0.5 + 0.5)))
            strength = int(100 * sim)
            return "black!{:d}".format(strength)

        def _latex_nn_table(tbl, l0, l1):
            cols = max(len(r) for r in tbl) - 1
            _file_out("\\begin{tabular}{c |" + " c"*cols + "}")
            _file_out("    {} & {} \\\\\\hline".format(l0, l1))
            for row in tbl:
                _file_out("    {} & {} \\\\".format(
                    _latex_name(row[0]), " & ".join("{{\\color{{{}}}{}}}".format(_latex_col(sim), _latex_name(nm)) for (nm, sim) in row[1:])
                ))
            _file_out("\\end{tabular}")

        _latex_nn_table(neighbour_table0, lang_names[0], lang_names[1])
        _latex_nn_table(neighbour_table1, lang_names[1], lang_names[0])

        # Output all of this text to a file
        with NamedFileWriter(self.info.get_absolute_output_dir("pairs"),
                             self.info.get_output("pairs").filename) as writer:
            writer.write_data(pair_output.getvalue())
            self.log.info("Pair analysis output to {}".format(writer.absolute_path))


def _name_to_filename(name):
    """ Sanitize a name so it can be used as a filename """
    return name.replace("/", "SLASH")
