"""
Collect results from the corruption experiments, including models trained on corrupted
corpora, and analyse them.

"""
from pimlico.core.dependencies.python import numpy_dependency, PythonPackageOnPip
from pimlico.core.modules.base import BaseModuleInfo
from pimlico.datatypes.base import MultipleInputs
from pimlico.datatypes.dictionary import Dictionary
from pimlico.datatypes.files import NamedFile, FileInput, UnnamedFileCollection
from pimlico.datatypes.keras import KerasModelBuilderClass


class ModuleInfo(BaseModuleInfo):
    module_type_name = "corruption_results"
    module_inputs = [
        ("corruption_params", MultipleInputs(FileInput())),
        ("models", MultipleInputs(KerasModelBuilderClass)),
        ("vocab1s", MultipleInputs(Dictionary)),
        ("vocab2s", MultipleInputs(Dictionary)),
        ("mapped_pairs", MultipleInputs(FileInput())),
    ]
    module_outputs = [
        ("analysis", NamedFile("analysis.txt")),
        ("files", UnnamedFileCollection),
    ]
    module_options = {}

    def get_software_dependencies(self):
        return super(ModuleInfo, self).get_software_dependencies() + \
               [PythonPackageOnPip("matplotlib"), numpy_dependency]