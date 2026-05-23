import importlib.util
import os

_module_path = os.path.join(os.path.dirname(__file__), 'SA_MGSTFN.py')
_spec = importlib.util.spec_from_file_location('sa_mgstfn_impl', _module_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

SAMGSTFN = _module.SAMGSTFN
