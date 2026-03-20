import runpy
import tempfile
import traceback
from pathlib import Path
import importlib.util
import sys

# Preload core modules directly from source to avoid package-level imports
src_prefix = Path(__file__).resolve().parents[1] / 'src' / 'crisprscreens' / 'core'
modules_to_load = ['qc', 'report', 'plots']
for m in modules_to_load:
    path = str((src_prefix / f"{m}.py").resolve())
    spec = importlib.util.spec_from_file_location(f"crisprscreens.core.{m}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"crisprscreens.core.{m}"] = mod
    spec.loader.exec_module(mod)

# Now run the tests file which imports using package paths
mod = runpy.run_path('tests/test_report_and_qc.py')
results = {'passed': [], 'failed': []}

# Helper to call test functions

def call_test(fn, *args):
    try:
        fn(*args)
        results['passed'].append(fn.__name__)
    except Exception:
        results['failed'].append((fn.__name__, traceback.format_exc()))

# Call tests that don't need tmp_path
call_test(mod['test_qc_replicate_consistency_skips_baseline'])
call_test(mod['test_qc_logfc_distribution_skips_baseline'])
call_test(mod['test_qc_controls_neutrality_baseline_explicit'])

# For tests needing tmp_path, create temp dir
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    call_test(mod['test_result_report_mle_and_rra'], td)

print('Passed:', results['passed'])
print('Failed:', [f[0] for f in results['failed']])
for name, tb in results['failed']:
    print('\n--- Failed:', name, '---')
    print(tb)
