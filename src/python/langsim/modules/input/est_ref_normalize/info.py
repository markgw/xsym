"""Special normalization routine for Estonian Reference Corpus.

Splits up sentences into separate lines. This is easy to do, since the
corpus puts a double space between sentences. There are also double
spaces in other places, so we only split on double spaces after punctuation.
Other double spaces are removed.

We also lower-case the whole corpus.

"""
from pimlico.core.modules.map import DocumentMapModuleInfo

from pimlico.core.modules.options import str_to_bool
from pimlico.datatypes.documents import TextDocumentType, RawTextDocumentType
from pimlico.datatypes.tar import TarredCorpusType, TarredCorpusWriter, tarred_corpus_with_data_point_type


class ModuleInfo(DocumentMapModuleInfo):
    module_type_name = "est_ref_normalize"
    module_readable_name = "Est Ref normalization"
    module_inputs = [("corpus", TarredCorpusType(TextDocumentType))]
    module_outputs = [("corpus", tarred_corpus_with_data_point_type(RawTextDocumentType))]
    module_options = {
        "forum": {
            "help": "Set to T for processing the forum data, which is slightly different to the newspaper data",
            "type": str_to_bool,
        },
    }

    def get_writer(self, output_name, output_dir, append=False):
        return TarredCorpusWriter(output_dir, append=append)
