"""Copy the explicit provider-boundary dependency closure into an isolated stage."""
from pathlib import Path
import shutil

FILES = ('provider_project_boundary.mjs', 'project_boundary_control.mjs',
         'project_boundary_routes.mjs', 'project_boundary_files.mjs', 'paused_echo_files.mjs',
         'worker_boundary_registration.mjs', 'provider_exchange_boundary.mjs', 'exchange_client.mjs')


def copy_boundary_modules(target):
    here = Path(__file__).resolve().parent
    for name in FILES:
        shutil.copyfile(here / name, Path(target) / name)
