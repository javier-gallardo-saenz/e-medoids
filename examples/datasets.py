"""Molecular fingerprints for the examples.

Defaults to ZINC-250k (250k drug-like molecules), downloaded once into data/.
Fingerprints are cached next to the SMILES file, since featurising takes far
longer than clustering.
"""
import csv
import urllib.request
from pathlib import Path

import numpy as np

from emedoids import pack_bits

DATA = Path(__file__).resolve().parent.parent / "data"
ZINC_URL = ("https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
            "master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv")


def zinc_smiles():
    """SMILES strings of ZINC-250k."""
    path = DATA / "zinc250k.csv"
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        print(f"downloading ZINC-250k to {path} ...")
        urllib.request.urlretrieve(ZINC_URL, path)
    with open(path, newline="") as f:
        return [row["smiles"].strip() for row in csv.DictReader(f)]


def read_smiles(path):
    """SMILES from a text file: one molecule per line, SMILES in the first column."""
    with open(path) as f:
        return [line.split()[0] for line in f if line.strip()]


def morgan_fingerprints(smiles, radius=2, n_bits=2048):
    """Bit-packed Morgan fingerprints, shape (n_valid, n_bits / 64), dtype uint64.

    radius=2 corresponds to ECFP4. Unparsable SMILES are skipped.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    mols = (Chem.MolFromSmiles(s) for s in smiles)
    bits = np.array([gen.GetFingerprintAsNumPy(m) for m in mols if m is not None])
    return pack_bits(bits)


def load_fingerprints(smiles_file=None, n=None, seed=0):
    """Fingerprints of `n` random molecules from a SMILES file (default: ZINC-250k)."""
    source = Path(smiles_file) if smiles_file else DATA / "zinc250k.csv"
    cache = source.with_suffix(".morgan2_2048.npy")
    if cache.exists():
        X = np.load(cache)
    else:
        smiles = read_smiles(source) if smiles_file else zinc_smiles()
        print(f"computing fingerprints of {len(smiles)} molecules (cached afterwards) ...")
        X = morgan_fingerprints(smiles)
        np.save(cache, X)
    if n is not None and n < len(X):
        X = X[np.random.default_rng(seed).choice(len(X), n, replace=False)]
    return np.ascontiguousarray(X)
