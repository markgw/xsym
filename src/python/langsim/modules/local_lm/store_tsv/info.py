from pimlico.core.modules.base import BaseModuleInfo
from pimlico.datatypes.embeddings import Embeddings, TSVVecFiles, TSVVecFilesWriter


class TSVVecFilesWithLangsWriter(TSVVecFilesWriter):
    """
    Write embeddings and their labels to TSV files, as used by Tensorflow.

    Include additional language data in metadata.

    """
    def write_vocab_with_langs(self, word_counts):
        import csv

        with open(self.get_absolute_path(self.filenames[1]), "w") as f:
            writer = csv.writer(f, dialect="excel-tab")
            writer.writerow(["Word", "Count", "Language", "Language num", "Word with lang"])
            languages = []
            for word, count in word_counts:
                # Split off the language ID from the word
                lang, __, word_only = word.partition(u":")
                if lang not in languages:
                    languages.append(lang)

                writer.writerow([
                    unicode(word_only).encode("utf-8"),
                    str(count),
                    unicode(lang).encode("utf-8"),
                    str(languages.index(lang)),
                    unicode(word).encode("utf-8")
                ])
        self.file_written(self.filenames[1])


class ModuleInfo(BaseModuleInfo):
    """
    Takes embeddings stored in the default format used within Pimlico pipelines
    (see :class:`~pimlico.datatypes.embeddings.Embeddings`) and stores them
    as TSV files.

    These are suitable as input to the `Tensorflow Projector <https://projector.tensorflow.org/>`_.

    Like the built-in store_tsv module, but includes some additional language information in the
    metadata to help with visualization.

    """
    module_type_name = "store_tsv"
    module_readable_name = "Store in TSV format"
    module_inputs = [("embeddings", Embeddings)]
    module_outputs = [("embeddings", TSVVecFiles)]
