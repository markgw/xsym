import csv
import json
import matplotlib.pyplot as plt
import sys
from cStringIO import StringIO
from operator import itemgetter

import numpy
from tabulate import tabulate

from langsim.modules.local_lm.neural_sixgram2.model import MappedPairsRankCalculator
from pimlico.core.modules.base import BaseModuleExecutor
from pimlico.datatypes.files import UnnamedFileCollectionWriter, NamedFileWriter
from pimlico.utils.progress import get_progress_bar


class ModuleExecutor(BaseModuleExecutor):
    def execute(self):
        models = self.info.get_input("models", always_list=True)
        vocab1s = self.info.get_input("vocab1s", always_list=True)
        vocab2s = self.info.get_input("vocab2s", always_list=True)
        corruption_paramses = self.info.get_input("corruption_params", always_list=True)
        mapped_pairs_lists = self.info.get_input("mapped_pairs", always_list=True)

        results = []
        corr_param_names = None

        self.log.info("Computing evaluation metric for each set of embeddings")
        pbar = get_progress_bar(len(models), title="Computing")
        for model_builder, vocab1_inp, vocab2_inp, corruption_params_inp, mapped_pairs_input in \
                pbar(zip(models, vocab1s, vocab2s, corruption_paramses, mapped_pairs_lists)):
            model = model_builder.load_model()
            vocab1 = vocab1_inp.get_data()
            vocab2 = vocab2_inp.get_data()
            mapped_pairs = json.loads(mapped_pairs_input.read_file())
            corruption_params = json.loads(corruption_params_inp.read_file())

            if corr_param_names is None:
                # The first time, get the names of the parameters
                # We assume they're the same for all results set
                corr_param_names = list(corruption_params.keys())

            # Load the trained embeddings from this model
            embeddings = model.get_embeddings()
            # The embeddings are assumed to be the whole of vocab1 followed by vocab2
            embeddings1 = embeddings[:len(vocab1)]
            embeddings2 = embeddings[len(vocab1):]

            # Get the vocab indices of the pairs that should be mapped to one another
            mapped_pairs_indices = [
                (vocab1.token2id[a], vocab2.token2id[b]) for (a, b) in mapped_pairs
            ]

            metric_calc = MappedPairsRankCalculator(mapped_pairs_indices)
            metric = metric_calc.compute(embeddings1, embeddings2)

            results.append((corruption_params, metric))

        # Collect a table of results for each setting
        results_table = [[name for name in corr_param_names] + ["pair-rank"]] + [
            [cps[param_name] for param_name in corr_param_names] + [str(metric)] for (cps, metric) in results
        ]

        results.sort(key=itemgetter(1))

        analysis = StringIO()
        def _print_and_store(x):
            print >>analysis, x
            print x

        scores_table = tabulate([
            [corruption_params[p] for p in corr_param_names] + [metric]
            for corruption_params, metric in results
        ], headers=corr_param_names+["Rank score"])
        _print_and_store(
            "\nScores for different corruption parameters, sorted by ascending (worsening) score:\n"
            "{}\n".format(scores_table)
        )

        correlations = []
        with UnnamedFileCollectionWriter(self.info.get_absolute_output_dir("files")) as files:
            def _correlate_and_plot(data, name):
                # Perform correlation and output the result
                corr = numpy.corrcoef(data)[0, 1]
                self.log.info("Correlation between {} and rank metric: {}".format(name, corr))
                # Plot the same data series
                filename = "{}_vs_rank.svg".format(name)
                path = files.get_absolute_path(filename)
                poly_coefs = plot(data, path, name.capitalize(), "Parameter {} vs rank (r={:.3f})".format(name, float(corr)))
                files.add_written_file(filename)
                self.log.info("  Plot output to {}".format(path))
                correlations.append((name, corr, poly_coefs))

            # Compute corr coef between summed corruption and metric
            metric_vals = numpy.array([
                [sum(corruption_params.values()), metric]
                for corruption_params, metric in results
            ]).T
            _correlate_and_plot(metric_vals, "sum")

            # Do the same for each param
            for corr_param_name in corr_param_names:
                metric_vals = numpy.array([
                    [corruption_params[corr_param_name], metric]
                    for corruption_params, metric in results
                ]).T
                _correlate_and_plot(metric_vals, corr_param_name)

            # Also output CSV of results, for manual analysis later
            table_data = StringIO()
            w = csv.writer(table_data)
            w.writerows(results_table)
            files.write_file("results_table.csv", table_data.getvalue())

        correlations_table = tabulate(
            [(n, "{:.3f}".format(c), "{:.2f}".format(coefs[0])) for (n, c, coefs) in correlations],
            headers=("Parameter", "Corr coef", "Slope")
        )
        _print_and_store("\nCorrelations:\n{}\n".format(correlations_table))

        with NamedFileWriter(self.info.get_absolute_output_dir("analysis"),
                             self.info.get_output("analysis").filename) as writer:
            writer.write_data(analysis.getvalue())
            self.log.info("Output analysis tables to {}".format(writer.absolute_path))


# Start off trying to display plots, but if it fails once, don't try on subsequent plots
# This is usually because we're not at a graphical terminal, so just output to a file instead
_show_plots = True
_show_plots = False


def plot(data_series, output_filename, xname, title):
    # Plot the two series together
    fig, ax = plt.subplots()
    ax.scatter(data_series[0], data_series[1])
    ax.set_xlabel(xname, fontsize=15)
    ax.set_ylabel("Mapped character rank", fontsize=15)
    ax.set_title(title)

    z = numpy.polyfit(data_series[0], data_series[1], 1)
    p = numpy.poly1d(z)
    ax.plot(data_series[0], p(data_series[0]), "r--")

    ax.grid(True)
    fig.tight_layout()

    plt.savefig(output_filename)

    global _show_plots
    if _show_plots:
        try:
            plt.show()
        except Exception, e:
            _show_plots = False
            print >>sys.stderr, "Could not show plots: {}".format(e)
            print >>sys.stderr, "Not trying to show future plots"

    # Return the polynomial coefficients
    return z
