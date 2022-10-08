import os
from moseq2_ephys_sync import cli


def test_entrypoint():
    exit_status = os.system('moseq2_ephys_sync --help')
    assert exit_status == 0