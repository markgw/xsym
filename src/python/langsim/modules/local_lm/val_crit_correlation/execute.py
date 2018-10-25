from cStringIO import StringIO
import os
import csv
import numpy as np

import matplotlib

from langsim.working.training_metrics import final_corrcoef

matplotlib.use("svg")
import matplotlib.pyplot as plt

from pimlico.core.modules.base import BaseModuleExecutor
from pimlico.datatypes.files import NamedFileWriter


class ModuleExecutor(BaseModuleExecutor):
    def execute(self):
        model_inputs = self.info.get_input("models", always_list=True)

        nn_sims = []
        overlap_ranks = []
        final_nn_sims = []
        final_overlap_ranks = []

        for model in model_inputs:
            metrics_dir = os.path.join(model.absolute_base_dir, "metrics")

            with open(os.path.join(metrics_dir, "nearest_neighbour_sim.csv"), "r") as f:
                r = csv.reader(f)
                # Skip header
                r.next()
                model_nn_sims = [line[2] for line in r]
            with open(os.path.join(metrics_dir, "overlap_rank.csv"), "r") as f:
                r = csv.reader(f)
                r.next()
                model_overlap_ranks = [line[2] for line in r]

            nn_sims.extend(model_nn_sims)
            overlap_ranks.extend(model_overlap_ranks)
            final_nn_sims.append(model_nn_sims[-1])
            final_overlap_ranks.append(model_overlap_ranks[-1])

            if len(model_nn_sims) != len(model_overlap_ranks):
                raise ValueError("model {} has a different number of NN sims and overlap ranks ({} and {})".format(
                    metrics_dir, len(model_nn_sims), len(model_overlap_ranks)
                ))

            self.log.info("Read {} rows from {}".format(len(model_nn_sims), metrics_dir))

        self.log.info("Read total {} rows".format(len(nn_sims)))

        with NamedFileWriter(self.info.get_absolute_output_dir("metrics"), self.info.get_output("metrics").filename) as writer:
            o = StringIO()
            w = csv.writer(o)
            w.writerows(zip(nn_sims, overlap_ranks))
            writer.write_data(o.getvalue())
            self.log.info("Output all metrics to {}".format(writer.absolute_path))

            metrics_output_path = writer.data_dir

        with NamedFileWriter(self.info.get_absolute_output_dir("final_metrics"), self.info.get_output("final_metrics").filename) as writer:
            o = StringIO()
            w = csv.writer(o)
            w.writerows(zip(final_nn_sims, final_overlap_ranks))
            writer.write_data(o.getvalue())
            self.log.info("Output final metrics to {}".format(writer.absolute_path))

            final_metrics_output_path = writer.data_dir

        nn_sims = [float(x) for x in nn_sims]
        overlap_ranks = [float(x) for x in overlap_ranks]

        training_corrcoef = np.corrcoef(np.vstack((nn_sims, overlap_ranks)))[0, 1]
        self.log.info("Correlation coefficient between NN metric and actual overlap rank "
                      "during training: {}".format(training_corrcoef))

        # Plot the two series together
        fig, ax = plt.subplots()
        ax.scatter(nn_sims, overlap_ranks)
        ax.set_xlabel("Nearest neighbour distance", fontsize=15)
        ax.set_ylabel("Actual mapped character rank", fontsize=15)
        ax.set_title("Fit metrics measured during training")

        ax.grid(True)
        fig.tight_layout()

        plot_path = os.path.join(metrics_output_path, "all_metrics_plot.svg")
        plt.savefig(plot_path)
        self.log.info("Plot all metrics to {}".format(plot_path))

        # Do the same stuff for the metrics right at the end of training
        final_nn_sims = [float(x) for x in final_nn_sims]
        final_overlap_ranks = [float(x) for x in final_overlap_ranks]

        final_corrcoef = np.corrcoef(np.vstack((final_nn_sims, final_overlap_ranks)))[0,1]
        self.log.info("Correlation coefficient between NN metric and actual overlap rank "
                      "after training: {}".format(final_corrcoef))

        fig, ax = plt.subplots()
        ax.scatter(final_nn_sims, final_overlap_ranks)
        ax.set_xlabel("Nearest neighbour distance", fontsize=15)
        ax.set_ylabel("Actual mapped character rank", fontsize=15)
        ax.set_title("Fit metrics measured at end of training")

        ax.grid(True)
        fig.tight_layout()

        plot_path = os.path.join(final_metrics_output_path, "final_metrics_plot.svg")
        plt.savefig(plot_path)
        self.log.info("Plot final metrics to {}".format(plot_path))

        # Output the values of the correlation coefficients
        with NamedFileWriter(self.info.get_absolute_output_dir("correlations"),
                             self.info.get_output("correlations").filename) as writer:
            writer.write_data(
                "Correlation coefficient between NN metric and actual overlap rank "
                "during training: {}\n"
                "Correlation coefficient between NN metric and actual overlap rank "
                "after training: {}\n".format(training_corrcoef, final_corrcoef)
            )
