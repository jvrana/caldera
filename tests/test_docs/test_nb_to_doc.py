import subprocess
from os.path import abspath
from os.path import dirname
from os.path import isdir
from os.path import isfile
from os.path import join
from typing import Any
from typing import Optional

import pytest


here = abspath(dirname(__file__))
fixtures = join(here, "fixtures")


class CalledProcessError(RuntimeError):
    pass


def cmd_output(*cmd: str, retcode: Optional[int] = 0, **kwargs: Any) -> str:
    kwargs.setdefault("stdout", subprocess.PIPE)
    kwargs.setdefault("stderr", subprocess.PIPE)
    proc = subprocess.Popen(cmd, **kwargs)
    stdout, stderr = proc.communicate()
    stdout = stdout.decode()
    if retcode is not None and proc.returncode != retcode:
        raise CalledProcessError(cmd, retcode, proc.returncode, stdout, stderr)
    return stdout


@pytest.mark.parametrize("use_tmp", [True])
@pytest.mark.env("doc-tools")
@pytest.mark.parametrize("temp_sys_path", ["../../docs/"], indirect=True)
def test_to_tmpdir(tmpdir, use_tmp, temp_sys_path):
    from _tools.nb_to_doc import main

    if use_tmp is False:
        outdir = None
        expected_outdir = fixtures
    else:
        outdir = tmpdir.join("notebook").mkdir()
        expected_outdir = outdir
    notebook = join(fixtures, "notebook.ipynb")
    assert not isfile(join(expected_outdir, "notebook.rst"))
    assert not isdir(join(expected_outdir, "notebook_files"))
    assert not isfile(join(expected_outdir, "notebook_files", "file.txt"))
    main(notebook, outdir=outdir)
    assert isfile(join(expected_outdir, "notebook.rst"))
    assert isdir(join(expected_outdir, "notebook_files"))
    assert isfile(join(expected_outdir, "notebook_files", "file.txt"))
