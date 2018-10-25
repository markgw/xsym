from pimlico.core.dependencies.python import numpy_dependency, PythonPackageOnPip
from pimlico.core.modules.base import BaseModuleInfo
from pimlico.datatypes.base import MultipleInputs
from pimlico.datatypes.files import NamedFile
from pimlico.datatypes.keras import KerasModelBuilderClass


class ModuleInfo(BaseModuleInfo):
    """
    Compute correlation between the validation criterion and the retrieval
    of known correspondences. See the paper for more details.

    """
    module_type_name = "validation_criterion_correlation"
    module_inputs = [
        ("models", MultipleInputs(KerasModelBuilderClass)),
    ]
    module_outputs = [
        ("metrics", NamedFile("collected_metrics.csv")),
        ("final_metrics", NamedFile("collected_final_metrics.csv")),
        ("correlations", NamedFile("correlations.txt")),
    ]
    module_options = {}

    def get_software_dependencies(self):
        return super(ModuleInfo, self).get_software_dependencies() + \
               [PythonPackageOnPip("matplotlib"), numpy_dependency]
