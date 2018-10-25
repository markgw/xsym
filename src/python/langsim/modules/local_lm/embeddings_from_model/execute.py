from operator import itemgetter
import numpy

from pimlico.core.modules.base import BaseModuleExecutor
from pimlico.core.modules.execute import ModuleExecutionError
from pimlico.datatypes.embeddings import EmbeddingsWriter


class ModuleExecutor(BaseModuleExecutor):
    def execute(self):
        vocabs = [v.get_data() for v in self.info.get_input("vocabs", always_list=True)]
        lang_names = self.info.options["lang_names"]
        if len(vocabs) != len(lang_names):
            raise ModuleExecutionError("got a different number of vocabs and language names")
        freqs = [f.array for f in self.info.get_input("frequencies", always_list=True)]
        if len(vocabs) != len(freqs):
            raise ModuleExecutionError("got a different number of vocabs and frequency arrays")

        # Load the vectors from the model
        model = self.info.get_input("model").load_model()
        vectors = model.get_embeddings()

        # Infer whether the model was trained with a vocab that includes OOV (more recent models) or whether OOV
        # was represented by an extra index
        oov_included = vectors.shape[0] == sum(len(v) for v in vocabs)

        # Prepend the language name to each word in the vocabulary to distinguish them
        names = sum([
            [u"{}:{}".format(lang_name, name) for (id, name) in sorted(vocab.id2token.items(), key=itemgetter(0))] +
            # Add the OOV item if it's implicitly at the end of the vocabulary and not included explicitly
            ([] if oov_included else [u"{}:OOV".format(lang_name)])
            for (lang_name, vocab) in zip(lang_names, vocabs)
        ], [])
        if len(names) != vectors.shape[0]:
            raise ModuleExecutionError("computing names from vocabs resulted in a name list of length {}, but there "
                                       "are {} embeddings".format(len(names), vectors.shape[0]))

        freqs = numpy.concatenate(freqs)
        if freqs.shape[0] != vectors.shape[0]:
            raise ModuleExecutionError("number of word frequencies ({}) was not the same as the number of vectors "
                                       "({})".format(freqs.shape[0], vectors.shape[0]))
        word_counts = zip(names, freqs)

        self.log.info("Vectors, names and counts loaded for {} words".format(len(word_counts)))

        with EmbeddingsWriter(self.info.get_absolute_output_dir("embeddings")) as writer:
            writer.write_vectors(vectors)
            self.log.info("Vectors written")
            writer.write_word_counts(word_counts)
            self.log.info("Vocabulary written")
