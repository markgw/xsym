from pimlico.core.modules.base import BaseModuleExecutor
from pimlico.datatypes.embeddings import EmbeddingsWriter


class ModuleExecutor(BaseModuleExecutor):
    def execute(self):
        embeddings = self.info.get_input("embeddings")
        lang1 = self.info.options["lang1"]
        if lang1 is None:
            # Look in the vocab to decide which is language 1
            word1 = embeddings.index2word[0]
            lang1 = word1.partition(":")[0]
            self.log.info("Treating {} as language 1".format(lang1))
        else:
            self.log.info("Any words starting '{}' will be treated as language 1".format(lang1))

        lang_ids = [1 if word.startswith(lang1) else 2 for word in embeddings.index2word]
        lang1_word_counts = [
            (word.partition(":")[2], cnt) for ((word, cnt), lang) in zip(embeddings.word_counts, lang_ids) if lang == 1
        ]
        lang2_word_counts = [
            (word.partition(":")[2], cnt) for ((word, cnt), lang) in zip(embeddings.word_counts, lang_ids) if lang == 2
        ]
        # Get the embeddings for the two languages in separate arrays
        lang1_ids = [i for (i, lang) in enumerate(lang_ids) if lang == 1]
        lang2_ids = [i for (i, lang) in enumerate(lang_ids) if lang == 2]
        self.log.info("Language 1: {} vectors, language 2: {} vectors".format(len(lang1_ids), len(lang2_ids)))
        lang1_embeddings = embeddings.vectors[lang1_ids]
        lang2_embeddings = embeddings.vectors[lang2_ids]

        self.log.info("Writing language 1 embeddings")
        with EmbeddingsWriter(self.info.get_absolute_output_dir("lang1_embeddings")) as writer1:
            writer1.write_word_counts(lang1_word_counts)
            writer1.write_vectors(lang1_embeddings)

        self.log.info("Writing language 2 embeddings")
        with EmbeddingsWriter(self.info.get_absolute_output_dir("lang2_embeddings")) as writer2:
            writer2.write_word_counts(lang2_word_counts)
            writer2.write_vectors(lang2_embeddings)
