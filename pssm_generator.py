#!/usr/bin/env python3
"""
Generate simple PSSM-like profiles for a few protein sequences
using NCBI's remote BLASTP service (no local database).

- Uses Biopython's NCBIWWW.qblast to talk to the BLAST URL API.
- Builds a position-specific frequency + log-odds matrix from HSPs.
- Intended for *small* batches only (respect NCBI usage guidelines!).
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path  # NEW

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # NEW
from Bio.Blast import NCBIWWW, NCBIXML

# 20 standard amino acids in a fixed order
AMINO_ACIDS = list("ARNDCQEGHILKMFPSTWYV")
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Background frequencies – here uniform; you can plug in BLOSUM62 background if you want
BACKGROUND = {aa: 1.0 / 20.0 for aa in AMINO_ACIDS}


def run_remote_blastp(
    seq: str,
    db: str = "swissprot",
    expect: float = 1e-3,
    hitlist_size: int = 100,
    sleep_between: float = 0.0,
) -> object:
    """
    Submit a single BLASTP query to NCBI and return the parsed BLAST record.

    NOTE: For more than ~50-100 sequences total, you should move to local BLAST+.
    """
    # Optional courtesy sleep if you're looping
    if sleep_between > 0:
        time.sleep(sleep_between)

    # NCBIWWW.qblast is a wrapper over the BLAST URL API
    # It already implements the 10-second spacing & polling rules internally.
    handle = NCBIWWW.qblast(
        program="blastp",
        database=db,
        sequence=seq,
        expect=expect,
        hitlist_size=hitlist_size,
        format_type="XML",
    )
    blast_record = NCBIXML.read(handle)
    handle.close()
    return blast_record


def collect_column_counts(
    blast_record,
    query_seq: str,
) -> List[Dict[str, int]]:
    """
    From a BLAST record, aggregate counts of aligned residues per query position.

    We iterate each HSP and, for every query residue position (no gaps),
    we record the aligned subject residue if present.
    """
    qlen = len(query_seq)
    # One dict per query position
    col_counts: List[Dict[str, int]] = [dict() for _ in range(qlen)]

    for alignment in blast_record.alignments:
        for hsp in alignment.hsps:
            # hsp.query and hsp.sbjct are aligned strings with '-' for gaps
            q_aln = hsp.query
            s_aln = hsp.sbjct

            # BLAST positions are 1-based; convert to 0-based index
            q_pos = hsp.query_start - 1

            for q_char, s_char in zip(q_aln, s_aln):
                if q_char != "-":
                    # We advance in the *query* whenever there's a query residue
                    if s_char in AA_INDEX:
                        col_dict = col_counts[q_pos]
                        col_dict[s_char] = col_dict.get(s_char, 0) + 1
                    q_pos += 1

            # Only use the best HSP per alignment for simplicity
            break

    return col_counts


def build_pssm_from_counts(
    col_counts: List[Dict[str, int]],
    query_seq: str,
    pseudocount: float = 1.0,
    include_query_as_count: bool = True,
) -> np.ndarray:
    """
    Turn per-position residue counts into a log-odds PSSM (L x 20 matrix).

    - col_counts[i] is a dict {aa: count} from aligned hits.
    - Optionally add each query residue as +1 count for its own position.
    - P(position i, aa) = (count(aa) + pseudocount) / (total + pseudocount*20)
    - Score = log2( P(aa|i) / background(aa) )
    """
    qlen = len(query_seq)
    pssm = np.zeros((qlen, len(AMINO_ACIDS)), dtype=float)

    for i in range(qlen):
        counts = dict(col_counts[i])  # copy

        if include_query_as_count:
            qaa = query_seq[i]
            if qaa in AA_INDEX:
                counts[qaa] = counts.get(qaa, 0) + 1

        # Total counts including pseudocounts
        total = sum(counts.values()) + pseudocount * len(AMINO_ACIDS)
        if total == 0:
            # No information at this position; leave as zeros
            continue

        for aa in AMINO_ACIDS:
            c = counts.get(aa, 0)
            p = (c + pseudocount) / total
            bg = BACKGROUND[aa]
            # Avoid log(0); bg>0 always here
            pssm[i, AA_INDEX[aa]] = np.log2(p / bg)

    return pssm


def blast_pssm_for_sequence(
    seq: str,
    db: str = "swissprot",
    expect: float = 1e-3,
    hitlist_size: int = 100,
    pseudocount: float = 1.0,
    include_query_as_count: bool = True,
    sleep_between: float = 0.0,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Convenience function: run remote BLASTP for a single sequence
    and return (pssm_array, pssm_dataframe).
    """
    seq = seq.strip().upper().replace(" ", "")

    blast_record = run_remote_blastp(
        seq,
        db=db,
        expect=expect,
        hitlist_size=hitlist_size,
        sleep_between=sleep_between,
    )
    col_counts = collect_column_counts(blast_record, seq)
    pssm = build_pssm_from_counts(
        col_counts,
        query_seq=seq,
        pseudocount=pseudocount,
        include_query_as_count=include_query_as_count,
    )

    df = pd.DataFrame(pssm, columns=AMINO_ACIDS)
    df.insert(0, "position", np.arange(1, len(seq) + 1))
    df.insert(1, "query_aa", list(seq))

    return pssm, df


def pssm_for_dataframe(
    df: pd.DataFrame,
    seq_col: str,
    id_col: Optional[str] = None,
    max_queries: int = 3,
    db: str = "swissprot",
    **pssm_kwargs,
) -> Dict[str, np.ndarray]:
    """
    Generate PSSMs for up to `max_queries` sequences from a pandas DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must contain a column with sequences in FASTA or raw AA format.
    seq_col : str
        Name of the column with amino acid sequences.
    id_col : str or None
        Column used as key for the result dict; if None, uses index.
    max_queries : int
        Limit to avoid hammering NCBI. Good to keep <= 5 for testing.
    db : str
        BLAST database name (e.g. "swissprot", "nr").

    Returns
    -------
    dict: { id -> np.ndarray (L x 20) }
    """
    results: Dict[str, np.ndarray] = {}

    # NCBI strongly suggests keeping scripted usage low volume.
    # This function is intentionally capped via max_queries.
    subset = df.head(max_queries)

    for idx, row in subset.iterrows():
        seq = row[seq_col]
        key = str(row[id_col]) if id_col is not None else str(idx)

        print(f"Running BLASTP for {key} (length {len(seq)}) against {db}...")
        pssm, _ = blast_pssm_for_sequence(seq, db=db, **pssm_kwargs)
        results[key] = pssm

        print(f"  -> PSSM shape: {pssm.shape}")

    return results


# ========= NEW HELPER FUNCTIONS FOR CSV + HEATMAP =========

def pssm_to_dataframe(pssm: np.ndarray) -> pd.DataFrame:
    """
    Convert a PSSM array (L x 20) into a DataFrame with a position column.
    """
    df = pd.DataFrame(pssm, columns=AMINO_ACIDS)
    df.insert(0, "position", np.arange(1, pssm.shape[0] + 1))
    return df


def save_pssm_to_csv(pssm: np.ndarray, seq_id: str, out_dir: str = "pssm_csv") -> str:
    """
    Save a PSSM matrix to a CSV file and return the file path.
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    df = pssm_to_dataframe(pssm)
    out_path = out_dir_path / f"pssm_{seq_id}.csv"
    df.to_csv(out_path, index=False)

    return str(out_path)


def plot_pssm_heatmap(
    pssm: np.ndarray,
    seq_id: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Create a heatmap for a PSSM matrix (L x 20).

    If save_path is provided, saves the figure there; otherwise shows it.
    """
    data = pssm

    plt.figure(figsize=(8, 6))
    im = plt.imshow(data, aspect="auto", origin="lower")
    plt.colorbar(im, label="PSSM score (log2 odds)")

    # X-axis: amino acids
    plt.xticks(np.arange(len(AMINO_ACIDS)), AMINO_ACIDS)

    # Y-axis: positions (may be long; thin the labels)
    positions = np.arange(1, data.shape[0] + 1)
    step = max(1, data.shape[0] // 20)  # aim for ~20 labels
    plt.yticks(
        np.arange(0, data.shape[0], step),
        positions[::step],
    )

    plt.xlabel("Amino acid")
    plt.ylabel("Position in sequence")
    plt.title(f"PSSM heatmap for {seq_id}")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Saved heatmap to {save_path}")
        plt.close()
    else:
        plt.show()


# ======================= MAIN SCRIPT =======================

if __name__ == "__main__":
    ps4_df = pd.read_csv("dataset/data.csv")
    pssms = pssm_for_dataframe(
        ps4_df,
        seq_col="input",
        id_col="chain_id",     # or None
        max_queries=3,   # keep very small with remote NCBI
        db="swissprot",
        hitlist_size=100,
        pseudocount=1.0,
    )

    # Existing debug print (unchanged)
    for key, mat in pssms.items():
        print(f"ID: {key}, PSSM shape: {mat.shape}")
        print(mat[:5])  # first 5 positions
        break

    # NEW: take the first PSSM, save to CSV, and create a heatmap
    if pssms:
        first_id, first_pssm = next(iter(pssms.items()))

        # Save PSSM to CSV
        csv_path = save_pssm_to_csv(first_pssm, first_id)
        print(f"Saved PSSM CSV for {first_id} to {csv_path}")

        # Save heatmap as PNG
        heatmap_path = f"pssm_heatmap_{first_id}.png"
        plot_pssm_heatmap(first_pssm, first_id, save_path=heatmap_path)
