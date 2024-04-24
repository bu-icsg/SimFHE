import os
import sys
from pathlib import Path
import importlib.util
import pytest

SIMPLIFIED_SCRIPTS_PATH = os.path.join( Path(__file__).parent.parent,'simplified_scripts')
sys.path.insert(0,SIMPLIFIED_SCRIPTS_PATH)

def importhelper(modulename):
    spec = importlib.util.spec_from_file_location(modulename, os.path.join(SIMPLIFIED_SCRIPTS_PATH,modulename+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[modulename] = module
    spec.loader.exec_module(module)
    return module

@pytest.mark.parametrize("modulename",["eval_sine_cost",
                                       "eval_sine_cost_optimized",
                                       "fft_cost_bsgs",
                                       "fft_cost_hoisted_unrolled"])
def test_run_scripts(modulename):
    print(f'-*- running module {modulename}')
    module = importhelper(modulename)
    module.main()
