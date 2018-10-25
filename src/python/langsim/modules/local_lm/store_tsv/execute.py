from langsim.modules.local_lm.store_tsv.info import TSVVecFilesWithLangsWriter
from pimlico.core.modules.base import BaseModuleExecutor


class ModuleExecutor(BaseModuleExecutor):
    def execute(self):
        embeddings = self.info.get_input("embeddings")

        # Output to the file
        with TSVVecFilesWithLangsWriter(self.info.get_absolute_output_dir("embeddings")) as writer:
            writer.write_vectors(embeddings.vectors)
            writer.write_vocab_with_langs(embeddings.word_counts)
