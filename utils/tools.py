import logging

import numpy as np
import pandas as pd
from Bio.SeqIO.FastaIO import SimpleFastaParser
from scipy.sparse import csr_matrix

Logger = logging.getLogger(__name__)


def write_df_to_fasta(df: pd.DataFrame, id_col: str, seq_col: str, fasta_path: str):
    with open(fasta_path, "w") as f:
        for idx, row in df[[id_col, seq_col]].iterrows():
            f.write(f">{row[id_col]}\n{row[seq_col]}\n")


def write_fasta_to_df(fasta_path: str):
    with open(fasta_path) as fasta_file:
        sequence = []
        for title, seq in SimpleFastaParser(fasta_file):
            # identifiers.append(title.split(None, 1)[0])  # First word is ID
            sequence.append(seq)

    df = pd.Series(sequence)
    return df


def save_sparse_csr(filename, array):
    # note that .npz extension is added automatically
    np.savez(
        filename,
        data=array.data,
        indices=array.indices,
        indptr=array.indptr,
        shape=array.shape,
    )


def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + ".npz")
    return csr_matrix(
        (loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"]
    )
