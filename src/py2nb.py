"""Convierte cuadernos percent (.py) a .ipynb y los ejecuta: python src/py2nb.py notebooks/x.py"""
import sys
import jupytext
import nbformat
from nbclient import NotebookClient

for f in sys.argv[1:]:
    nb = jupytext.read(f)
    res = {"metadata": {"path": "notebooks"}}
    NotebookClient(nb, timeout=3600, kernel_name="heart-mlops", resources=res).execute()
    nbformat.write(nb, f.replace(".py", ".ipynb"))
    print("ok", f)
