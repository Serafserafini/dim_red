import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def test_pipeline_light_modules_do_not_import_jax():
    code = (
        "import sys; "
        "import dim_red.pipeline.run_layout, dim_red.pipeline.compare, "
        "dim_red.pipeline.benchmark, dim_red.pipeline.cli; "
        "bad=[m for m in sys.modules if m.split('.')[0] in "
        "('jax','flax','optax','jaxlib')]; "
        "sys.exit(1 if bad else 0)"
    )
    env = dict(os.environ, PYTHONPATH=str(SRC))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
    assert proc.returncode == 0
