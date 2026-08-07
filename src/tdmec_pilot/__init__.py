"""TDMEC Dataset B controlled preprocessing pilot.

Reusable, Colab-independent processing modules. Core modules run on any normal
Linux filesystem; the notebook is a thin wrapper that only supplies a mounted
Google Drive path as the persistent output root.

Nothing here downloads all 70 Dataset B files, generates embeddings, or trains
the model. The pilot processes exactly the two configured input files.
"""

__version__ = "0.1.0"
