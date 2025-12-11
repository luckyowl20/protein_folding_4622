#!/usr/bin/env python3
"""
blast_msa_batch.py

Utilities to run BLASTP via the NCBI BLAST Common URL API and generate
query-anchored MSAs for multiple sequences stored in a pandas DataFrame.

Designed as a precursor step for building PSSMs over large datasets
(e.g. PS4), but **be careful**: NCBI does not intend their public BLAST
servers to be used as a massive batch backend. For very large datasets,
you should strongly consider installing BLAST+ locally instead.

Example usage:

    import pandas as pd
    from blast_msa_batch import BlastMSABatcher

    df = pd.read_csv("ps4_subset.csv")  # must contain a "sequence" column

    batcher = BlastMSABatcher(
        default_db="swissprot",
        email="you@example.com",
        tool_name="ps4_pssm_pipeline",
    )

    df_out = batcher.run_on_dataframe(
        df,
        seq_col="sequence",
        id_col="protein_id",      # optional, used for filenames
        db_col="blast_db",        # optional, per-row DB code
        out_dir="msa_outputs",
    )

    df_out.to_csv("ps4_with_msa_paths.csv", index=False)
"""

from __future__ import annotations

import time
import re
import sys
from pathlib import Path
from typing import Optional

import requests
import pandas as pd

BLAST_URL = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"

# Regex patterns to parse NCBI responses
RID_RE = re.compile(r"RID\s*=\s*([A-Z0-9]+)")
RTOE_RE = re.compile(r"RTOE\s*=\s*(\d+)")
STATUS_RE = re.compile(r"Status=\s*([^ \n\r]+)")


class BlastError(Exception):
    """Custom exception for BLAST-related failures."""
    pass


def _normalize_seq(seq: str) -> str:
    """Normalize an input amino acid sequence: strip whitespace and uppercase."""
    if seq is None:
        return ""
    return "".join(str(seq).split()).upper()


def submit_blast(
    query_seq: str,
    db: str = "swissprot",
    program: str = "blastp",
    hitlist_size: int = 100,
    email: Optional[str] = None,
    tool: Optional[str] = "blast_msa_batch",
    session: Optional[requests.Session] = None,
) -> tuple[str, int]:
    """
    Submit a BLAST search (CMD=Put) to NCBI BLAST URL API.
    Returns (RID, RTOE in seconds).
    """
    seq = _normalize_seq(query_seq)
    if not seq:
        raise BlastError("Empty query sequence.")

    fasta = f">query\n{seq}\n"

    params = {
        "CMD": "Put",
        "PROGRAM": program,
        "DATABASE": db,
        "QUERY": fasta,
        "HITLIST_SIZE": str(hitlist_size),
        "ALIGNMENTS": str(hitlist_size),
        "SHORT_QUERY_ADJUST": "true",
    }

    if email:
        params["email"] = email
    if tool:
        params["tool"] = tool

    sess = session or requests.Session()

    resp = sess.post(BLAST_URL, data=params)
    resp.raise_for_status()

    m_rid = RID_RE.search(resp.text)
    m_rtoe = RTOE_RE.search(resp.text)

    if not m_rid:
        raise BlastError(
            "Could not find RID in BLAST response.\n"
            f"Response text (truncated):\n{resp.text[:1000]}"
        )

    rid = m_rid.group(1)
    rtoe = int(m_rtoe.group(1)) if m_rtoe else 30
    return rid, rtoe


def wait_for_results(
    rid: str,
    poll_interval: int = 60,
    session: Optional[requests.Session] = None,
) -> None:
    """
    Poll NCBI until BLAST job with RID is READY or failed.
    Uses FORMAT_OBJECT=Status.
    """
    sess = session or requests.Session()

    while True:
        params = {
            "CMD": "Get",
            "RID": rid,
            "FORMAT_OBJECT": "Status",
            "FORMAT_TYPE": "Text",
        }
        resp = sess.get(BLAST_URL, params=params)
        resp.raise_for_status()

        status_text = resp.text
        m_status = STATUS_RE.search(status_text)
        status = m_status.group(1) if m_status else "UNKNOWN"

        print(f"[RID {rid}] Status: {status}", file=sys.stderr)

        if status == "READY":
            return
        elif status in {"FAILED", "UNKNOWN"}:
            raise BlastError(
                f"BLAST search {rid} failed or status unknown.\n"
                f"Status block:\n{status_text}"
            )

        time.sleep(poll_interval)


def fetch_alignment(
    rid: str,
    max_alignments: int = 100,
    alignment_view: str = "QueryAnchoredNoIdentities",
    session: Optional[requests.Session] = None,
) -> str:
    """
    Retrieve the BLAST alignment (query-anchored MSA-style view) as text.
    """
    sess = session or requests.Session()

    params = {
        "CMD": "Get",
        "RID": rid,
        "FORMAT_TYPE": "Text",
        "FORMAT_OBJECT": "Alignment",
        "ALIGNMENTS": str(max_alignments),
        "DESCRIPTIONS": str(max_alignments),
        "ALIGNMENT_VIEW": alignment_view,
    }

    resp = sess.get(BLAST_URL, params=params)
    resp.raise_for_status()
    return resp.text


class BlastMSABatcher:
    """
    High-level wrapper to apply BLASTP to a pandas DataFrame of sequences
    and store an MSA-style alignment file for each.

    Parameters
    ----------
    default_db : str
        BLAST database used when a row-specific db is not provided.
        Common values: "swissprot", "nr", etc.
    email : str, optional
        Contact email; strongly recommended by NCBI.
    tool_name : str, optional
        Tool name identifier sent to NCBI.
    max_hits : int, default 100
        Max number of alignments / hits to retrieve.
    poll_interval : int, default 60
        Seconds between status polls for each RID.
    per_query_delay : float, default 3.0
        Extra delay between *starting* different queries, to be polite.
    """

    def __init__(
        self,
        default_db: str = "swissprot",
        email: Optional[str] = None,
        tool_name: str = "blast_msa_batch",
        max_hits: int = 100,
        poll_interval: int = 60,
        per_query_delay: float = 3.0,
    ):
        self.default_db = default_db
        self.email = email
        self.tool_name = tool_name
        self.max_hits = max_hits
        self.poll_interval = poll_interval
        self.per_query_delay = per_query_delay
        self.session = requests.Session()

    def process_single_sequence(
        self,
        seq: str,
        db: Optional[str] = None,
    ) -> dict:
        """
        Run BLAST for a single sequence, wait for completion, and fetch MSA.

        Returns a dict with keys:
        - rid
        - rtoe
        - alignment (string)
        """
        effective_db = db or self.default_db

        rid, rtoe = submit_blast(
            query_seq=seq,
            db=effective_db,
            program="blastp",
            hitlist_size=self.max_hits,
            email=self.email,
            tool=self.tool_name,
            session=self.session,
        )

        # Respect server's estimate (RTOE) plus a minimum wait
        time.sleep(max(rtoe, 10))

        # Wait until job is fully ready
        wait_for_results(
            rid=rid,
            poll_interval=self.poll_interval,
            session=self.session,
        )

        # Fetch query-anchored alignment
        alignment_text = fetch_alignment(
            rid=rid,
            max_alignments=self.max_hits,
            alignment_view="QueryAnchoredNoIdentities",
            session=self.session,
        )

        return {
            "rid": rid,
            "rtoe": rtoe,
            "alignment": alignment_text,
        }

    def run_on_dataframe(
        self,
        df: pd.DataFrame,
        seq_col: str,
        id_col: Optional[str] = None,
        db_col: Optional[str] = None,
        out_dir: str | Path = "msa_outputs",
        filename_template: str = "{idx}_{id}_msa.txt",
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """
        Run BLAST/MSA generation for each row in a DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset containing sequences.
        seq_col : str
            Column name containing the amino acid sequence (FASTA body).
        id_col : str, optional
            Column used to generate nicer filenames (e.g. "protein_id").
            If None, only the row index is used.
        db_col : str, optional
            Column providing a BLAST database code per row. If None, uses
            self.default_db for all rows.
        out_dir : str or Path
            Directory where MSA text files will be written.
        filename_template : str
            Template for output filenames. Variables available:
              - idx : integer row index
              - id  : id_col value (if provided; otherwise "row{idx}")
        overwrite : bool
            If False (default) and a filename already exists, BLAST is
            skipped and existing file is left as-is.

        Returns
        -------
        pandas.DataFrame
            A copy of df with extra columns:
            - blast_rid
            - blast_rtoe
            - blast_db_used
            - blast_msa_path
            - blast_error (NaN if successful)
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        df = df.copy()

        df["blast_rid"] = pd.NA
        df["blast_rtoe"] = pd.NA
        df["blast_db_used"] = pd.NA
        df["blast_msa_path"] = pd.NA
        df["blast_error"] = pd.NA

        for idx, row in df.iterrows():
            seq = row.get(seq_col, None)
            seq_norm = _normalize_seq(seq)

            if not seq_norm:
                df.at[idx, "blast_error"] = "Empty or invalid sequence"
                continue

            db_val = row.get(db_col) if db_col is not None else None
            db_used = db_val if isinstance(db_val, str) and db_val else self.default_db

            # Build filename
            if id_col is not None and pd.notna(row.get(id_col)):
                id_val = str(row[id_col])
            else:
                id_val = f"row{idx}"

            fname = filename_template.format(idx=idx, id=id_val)
            out_path = out_dir / fname

            if out_path.exists() and not overwrite:
                print(
                    f"[row {idx}] File {out_path} exists, skipping BLAST.",
                    file=sys.stderr,
                )
                df.at[idx, "blast_msa_path"] = str(out_path)
                df.at[idx, "blast_db_used"] = db_used
                continue

            print(
                f"[row {idx}] Running BLAST for id={id_val}, db={db_used}",
                file=sys.stderr,
            )

            try:
                result = self.process_single_sequence(seq_norm, db=db_used)

                # Save alignment to disk
                out_path.write_text(result["alignment"])

                df.at[idx, "blast_rid"] = result["rid"]
                df.at[idx, "blast_rtoe"] = result["rtoe"]
                df.at[idx, "blast_db_used"] = db_used
                df.at[idx, "blast_msa_path"] = str(out_path)
                df.at[idx, "blast_error"] = pd.NA

            except Exception as exc:
                df.at[idx, "blast_error"] = str(exc)
                print(
                    f"[row {idx}] ERROR during BLAST: {exc}",
                    file=sys.stderr,
                )

            # Be polite to NCBI between *starting* jobs
            time.sleep(self.per_query_delay)

        return df
