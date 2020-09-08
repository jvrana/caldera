#! /usr/bin/env python
"""Execute a .ipynb file, write out a processed .rst and clean .ipynb.

Some functions in this script were copied from the nbstripout tool:
Copyright (c) 2015 Min RK, Florian Rathgeber, Michael McNeil Forbes 2019
Casper da Costa-Luis Permission is hereby granted, free of charge, to
any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to
do so, subject to the following conditions: The above copyright notice
and this permission notice shall be included in all copies or
substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import argparse
import contextlib
import os

import nbformat
from nbconvert import RSTExporter
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import ExtractOutputPreprocessor
from nbconvert.preprocessors import TagRemovePreprocessor
from traitlets.config import Config


class MetadataError(Exception):
    pass


def pop_recursive(d, key, default=None):
    """dict.pop(key) where `key` is a `.`-delimited list of nested keys.

    >>> d = {'a': {'b': 1, 'c': 2}}
    >>> pop_recursive(d, 'a.c')
    2
    >>> d
    {'a': {'b': 1}}
    """
    nested = key.split(".")
    current = d
    for k in nested[:-1]:
        if hasattr(current, "get"):
            current = current.get(k, {})
        else:
            return default
    if not hasattr(current, "pop"):
        return default
    return current.pop(nested[-1], default)


def strip_output(nb):
    """Strip the outputs, execution count/prompt number and miscellaneous
    metadata from a notebook object, unless specified to keep either the
    outputs or counts."""
    keys = {"metadata": [], "cell": {"metadata": []}}

    nb.metadata.pop("signature", None)
    nb.metadata.pop("widgets", None)

    for field in keys["metadata"]:
        pop_recursive(nb.metadata, field)

    for cell in nb.cells:

        # Remove the outputs, unless directed otherwise
        if "outputs" in cell:

            cell["outputs"] = []

        # Remove the prompt_number/execution_count, unless directed otherwise
        if "prompt_number" in cell:
            cell["prompt_number"] = None
        if "execution_count" in cell:
            cell["execution_count"] = None

        # Always remove this metadata
        for output_style in ["collapsed", "scrolled"]:
            if output_style in cell.metadata:
                cell.metadata[output_style] = False
        if "metadata" in cell:
            for field in ["collapsed", "scrolled", "ExecuteTime"]:
                cell.metadata.pop(field, None)
        for (extra, fields) in keys["cell"].items():
            if extra in cell:
                for field in fields:
                    pop_recursive(getattr(cell, extra), field)
    return nb


def read_notebook(fpath: str):
    # Read the notebook
    print(f"Executing {fpath} ...", end=" ", flush=True)
    with open(fpath) as f:
        nb = nbformat.read(f, as_version=4)
    return nb


def run_notebook(nb, basedir: str):
    # Run the notebook
    kernel = os.environ.get("NB_KERNEL", None)
    if kernel is None:
        kernel = nb["metadata"]["kernelspec"]["name"]
    ep = ExecutePreprocessor(
        timeout=600,
        kernel_name=kernel,
        extra_arguments=["--InlineBackend.rc={'figure.dpi': 96}"],
    )
    ep.preprocess(nb, {"metadata": {"traversal": basedir}})


def remove_plain_text_output(nb):
    # Remove plain text execution result outputs
    for cell in nb.get("cells", {}):
        fields = cell.get("outputs", [])
        for field in fields:
            if field["output_type"] == "execute_result":
                data_keys = field["data"].keys()
                for key in list(data_keys):
                    if key == "text/plain":
                        field["data"].pop(key)
                if not field["data"]:
                    fields.remove(field)


def convert_to_rst(nb, basedir, fpath, fstem):
    # Convert to .rst formats
    exp = RSTExporter()
    c = Config()
    c.TagRemovePreprocessor.remove_cell_tags = {"hide"}
    c.TagRemovePreprocessor.remove_input_tags = {"hide-input"}
    c.TagRemovePreprocessor.remove_all_outputs_tags = {"hide-output"}
    c.ExtractOutputPreprocessor.output_filename_template = (
        f"{fstem}_files/{fstem}_" + "{cell_index}_{index}{extension}"
    )
    exp.register_preprocessor(TagRemovePreprocessor(config=c), True)
    exp.register_preprocessor(ExtractOutputPreprocessor(config=c), True)
    body, resources = exp.from_notebook_node(nb)

    # Clean the output on the notebook and save a .ipynb back to disk
    print(f"Writing clean {fpath} ... ", end=" ", flush=True)
    nb = strip_output(nb)
    with open(fpath, "wt") as f:
        nbformat.write(nb, f)

    # Write the .rst file
    rst_path = os.path.join(basedir, f"{fstem}.rst")
    print(f"Writing {rst_path}")
    with open(rst_path, "w") as f:
        f.write(body)

    print(resources["outputs"])

    # Write the individual image outputs
    imdir = os.path.join(basedir, f"{fstem}_files")
    if not os.path.exists(imdir):
        os.mkdir(imdir)
    for imname, imdata in resources["outputs"].items():
        if imname.startswith(fstem):
            impath = os.path.join(basedir, f"{imname}")
            with open(impath, "wb") as f:
                f.write(imdata)
                f.write(imdata)


@contextlib.contextmanager
def chdir(dirname=None):
    curdir = os.getcwd()
    try:
        if dirname is not None:
            os.chdir(dirname)
        yield
    finally:
        os.chdir(curdir)


def main(fpath: str, outdir: str = None):
    """Execute a python notebook, save files, and convert to `rst`.

    1. Read in the notebook file
    2. Set current working directory to `<outdir>/<fname>_files`
    3. Execute the notebook.
    4. Save images, save rst file

    .. warning::

        This method will change current working directory before notebook execution.
        Make sure any readable input files are absolute or relative to the notebook file itself.
        Output paths should use cwd.

    :param fpath:
    :param outdir: (optional) provide output directory to save files
    :return:
    """
    basedir, fname = os.path.split(fpath)
    if outdir is None:
        outdir = basedir
    fstem = "".join(fname.split(".")[:-1])
    nb = read_notebook(fpath)

    # create output directory
    file_out_dir = os.path.join(outdir, f"{fstem}_files")
    if not os.path.exists(file_out_dir):
        os.mkdir(file_out_dir)

    # run notebook in directory
    with chdir(file_out_dir):
        run_notebook(nb, basedir)

    remove_plain_text_output(nb)

    convert_to_rst(nb, outdir, fpath, fstem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", "-f", help="file of the .ipynb file to execute")
    parser.add_argument(
        "--outdir", "-o", help="output directory to save the files", default=None
    )
    args = parser.parse_args()

    # Get the desired ipynb file traversal and parse into components
    main(fpath=args.filename, outdir=args.outdir)
