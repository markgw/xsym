"""
Display corrupted and uncorrupted texts alongside one another

For observing the output of the corruption process, which otherwise is just a load of
integer-encoded data.

"""
from pimlico.core.modules.map import DocumentMapModuleInfo
from pimlico.datatypes.documents import RawTextDocumentType
from pimlico.datatypes.tar import TarredCorpusWriter

from pimlico.datatypes.dictionary import Dictionary
from pimlico.datatypes.ints import IntegerListsDocumentType
from pimlico.datatypes.tar import TarredCorpusType, tarred_corpus_with_data_point_type


class ModuleInfo(DocumentMapModuleInfo):
    module_type_name = "inspect_corrupted"
    module_readable_name = "Inspect corrupted text"
    module_inputs = [
        ("corpus1", TarredCorpusType(IntegerListsDocumentType)),
        ("vocab1", Dictionary),
        ("corpus2", TarredCorpusType(IntegerListsDocumentType)),
        ("vocab2", Dictionary),
    ]
    module_outputs = [
        ("inspect", tarred_corpus_with_data_point_type(RawTextDocumentType)),
    ]
    module_options = {}

    def get_writer(self, output_name, output_dir, append=False):
        return TarredCorpusWriter(output_dir, append=append)
