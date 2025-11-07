#!/usr/bin/env python3
# convert_cb513_csv.py
import argparse
import pandas as pd
import numpy as np

def join_tokens(x):
    # x is a string like "A B C D ..."  -> "ABCD..."
    return "".join(x.strip().split())

def split_tokens(x):
    # "1.0 0.0 1.0" -> [1.0, 0.0, 1.0]
    return [float(t) for t in x.strip().split() if t]

def apply_mask(seq_str, mask_floats):
    # keep only positions where mask >= 0.5
    toks = seq_str.strip().split()
    if len(toks) != len(mask_floats):
        # if already no spaces (unexpected), fall back to char-wise
        toks = list(seq_str.strip())
        if len(toks) != len(mask_floats):
            raise ValueError("Length mismatch after tokenization.")
    keep = [i for i,v in enumerate(mask_floats) if v >= 0.5]
    out = "".join(toks[i] for i in keep)
    return out

def main():
    ap = argparse.ArgumentParser(description="Normalize CB513 CSV (tokenized columns) into plain strings with mask applied.")
    ap.add_argument("--in_csv", required=True, help="CB513_HHblits.csv path")
    ap.add_argument("--out_csv", default="cb513_norm.csv", help="Output CSV with chain_id,input,dssp8")
    ap.add_argument("--use_mask", action="store_true", help="Apply cb513_mask if present")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    need_cols = ["input","dssp8"]
    for c in need_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing column '{c}' in {args.in_csv}")

    out_rows = []
    for i, row in df.iterrows():
        seq_tok = str(row["input"])
        ss8_tok = str(row["dssp8"])

        if args.use_mask and ("cb513_mask" in df.columns):
            mask = split_tokens(str(row["cb513_mask"]))
            seq = apply_mask(seq_tok, mask)
            ss8 = apply_mask(ss8_tok, mask)
        else:
            # just strip spaces
            seq = join_tokens(seq_tok)
            ss8 = join_tokens(ss8_tok)

        # (Optional) ensure P->C mapping if any stray 'P' appears in labels
        ss8 = ss8.replace("P","C")

        out_rows.append({
            "chain_id": f"cb513_{i:04d}",
            "input": seq,
            "dssp8": ss8
        })

    out = pd.DataFrame(out_rows)
    # sanity: drop empty or mismatch
    out = out[(out["input"].str.len() > 0) & (out["input"].str.len() == out["dssp8"].str.len())]
    out.to_csv(args.out_csv, index=False)
    print(f"[info] wrote {args.out_csv} with {len(out)} rows")

if __name__ == "__main__":
    main()
