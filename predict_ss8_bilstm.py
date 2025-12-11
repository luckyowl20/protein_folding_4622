#!/usr/bin/env python3
# predict_ss8_bilstm.py
#
# Predict DSSP Q8/Q3 using a trained BiLSTM model (bilstm.py-compatible).

import argparse, os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID = 0

# ----------------------------
# Model (must match bilstm.py)
# ----------------------------

class BiLSTMSS(nn.Module):
    def __init__(self, vocab_size, n_classes=8, emb_dim=64, hidden=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(2 * hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, n_classes)

    def forward(self, x, mask, lengths):
        # x: (B,L) long, mask: (B,L) bool, lengths: (B,)
        emb = self.embed(x)  # (B,L,E)

        # pack by sequence lengths
        lengths_cpu = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B,L,2H)

        h = self.norm(h)
        h = self.dropout(h)
        logits = self.head(h)  # (B,L,K)

        # mask out padded positions for safety
        logits = logits.masked_fill(~mask.unsqueeze(-1), -1e4)
        return logits

# ----------------------------
# Utils
# ----------------------------

def load_ckpt(ckpt_path, device=None):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]            # dict: emb_dim, hidden, layers, dropout
    aa2id = ckpt["aa2id"]          # dict: aa -> int
    id2ss = {int(k): v for k, v in ckpt["id2ss"].items()}  # int -> 'H' etc.
    n_classes = len(ckpt["dssp8"])
    vocab_size = len(aa2id) + 1    # +PAD

    model = BiLSTMSS(
        vocab_size=vocab_size,
        n_classes=n_classes,
        emb_dim=args["emb_dim"],
        hidden=args["hidden"],
        n_layers=args["layers"],
        dropout=args["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, aa2id, id2ss, device

def encode_seq(seq, aa2id):
    # Unknown AA go to 'X' if present, otherwise a new id
    x_id = aa2id.get('X', max(aa2id.values()) + 1)
    return [aa2id.get(ch, x_id) for ch in seq.strip()]

def collate_pad_with_lengths(seqs, pad_id=PAD_ID):
    # seqs: list[list[int]]
    B = len(seqs)
    L = max(len(s) for s in seqs) if B > 0 else 0
    x = torch.full((B, L), pad_id, dtype=torch.long)
    m = torch.zeros((B, L), dtype=torch.bool)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, s in enumerate(seqs):
        n = len(s)
        x[i, :n] = torch.tensor(s, dtype=torch.long)
        m[i, :n] = True
        lengths[i] = n
    return x, m, lengths

def q8_to_q3_map():
    # Must match training label order: ['H','E','G','I','B','T','S','C']
    # Mapping: H/G/I -> H, E/B -> E, T/S/C -> C
    return torch.tensor([0, 1, 0, 0, 1, 2, 2, 2], dtype=torch.long)

def q8_string_to_q3(q8_str):
    mapping = {'H': 'H', 'G': 'H', 'I': 'H',
               'E': 'E', 'B': 'E',
               'T': 'C', 'S': 'C', 'C': 'C'}
    return ''.join(mapping.get(c, 'C') for c in q8_str)

def accuracy(a, b):
    # per-residue accuracy for two equal-length strings
    a = np.frombuffer(a.encode(), dtype='S1')
    b = np.frombuffer(b.encode(), dtype='S1')
    if len(a) == 0:
        return 0.0
    return float((a == b).sum()) / len(a)

# ----------------------------
# Prediction entry points
# ----------------------------

@torch.no_grad()
def predict_batch(model, aa2id, id2ss, device, seq_strs, batch_size=64):
    # Returns list of predicted Q8 strings in input order
    preds_all = []
    for i in range(0, len(seq_strs), batch_size):
        chunk = seq_strs[i:i + batch_size]
        toks = [encode_seq(s, aa2id) for s in chunk]
        x, m, lengths = collate_pad_with_lengths(toks)
        x, m, lengths = x.to(device), m.to(device), lengths.to(device)
        logits = model(x, m, lengths)
        pred_ids = logits.argmax(-1).cpu().numpy()  # (B,L)
        for bi, s in enumerate(chunk):
            n = len(s.strip())
            ids = pred_ids[bi][:n]
            preds_all.append(''.join(id2ss[int(k)] for k in ids))
    return preds_all

def read_csv(path):
    import pandas as pd
    df = pd.read_csv(path)
    if 'input' not in df.columns:
        raise ValueError("CSV must have an 'input' column (and optional 'chain_id','dssp8').")
    chains = df['chain_id'].astype(str).tolist() if 'chain_id' in df.columns else [f"seq{i}" for i in range(len(df))]
    seqs = df['input'].astype(str).tolist()
    golds = df['dssp8'].astype(str).tolist() if 'dssp8' in df.columns else None
    return chains, seqs, golds

def read_fasta(path):
    names, seqs = [], []
    cur_name, cur = None, []
    for line in open(path, 'r', encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if cur_name is not None:
                names.append(cur_name)
                seqs.append(''.join(cur))
            cur_name = line[1:].strip()
            cur = []
        else:
            cur.append(line.replace(' ', '').upper())
    if cur_name is not None:
        names.append(cur_name)
        seqs.append(''.join(cur))
    return names, seqs

def write_results(out_csv, names, seqs, preds, gold_q8=None):
    import pandas as pd

    assert len(names) == len(seqs) == len(preds)
    rows = []
    overall_q8_hit = overall_q8_n = 0
    overall_q3_hit = overall_q3_n = 0

    for i, (name, seq, pred_q8) in enumerate(zip(names, seqs, preds)):
        row = {
            "chain_id": name,
            "input": seq,
            "pred_q8": pred_q8,
            "pred_q3": q8_string_to_q3(pred_q8),
        }
        if gold_q8 is not None:
            gold = gold_q8[i]
            # truncate to min length if there is any mismatch
            L = min(len(gold), len(pred_q8))
            g8 = gold[:L]
            p8 = pred_q8[:L]
            row["gold_q8"] = g8
            row["gold_q3"] = q8_string_to_q3(g8)
            row["acc_q8"] = accuracy(g8, p8)
            row["acc_q3"] = accuracy(row["gold_q3"], row["pred_q3"][:L])

            overall_q8_hit += int(row["acc_q8"] * L)
            overall_q8_n += L
            overall_q3_hit += int(row["acc_q3"] * L)
            overall_q3_n += L

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    summary = {}
    if overall_q8_n > 0:
        summary["overall_Q8"] = overall_q8_hit / overall_q8_n
        summary["overall_Q3"] = overall_q3_hit / overall_q3_n
    return df, summary

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Predict DSSP Q8 (and Q3) from trained BiLSTM checkpoint.")
    ap.add_argument("--ckpt", required=True, help="Path to BiLSTM checkpoint (best.pt)")
    ap.add_argument("--seq", help="Single amino-acid sequence to predict")
    ap.add_argument("--csv", help="CSV with columns input[,chain_id][,dssp8]")
    ap.add_argument("--fasta", help="FASTA file with sequences")
    ap.add_argument("--out_csv", default="predictions_bilstm.csv", help="Output CSV for predictions")
    ap.add_argument("--batch_size", type=int, default=128, help="Batch size for inference")
    args = ap.parse_args()

    if not (args.seq or args.csv or args.fasta):
        print("Please provide one of --seq, --csv, or --fasta.", file=sys.stderr)
        sys.exit(2)

    model, aa2id, id2ss, device = load_ckpt(args.ckpt)
    print(f"[info] device: {device} | ckpt: {args.ckpt}")

    if args.seq:
        seqs = [args.seq.strip().upper()]
        names = ["seq0"]
        preds = predict_batch(model, aa2id, id2ss, device, seqs, args.batch_size)
        print(preds[0])
        print("Q3:", q8_string_to_q3(preds[0]))
        return

    if args.csv:
        names, seqs, golds = read_csv(args.csv)
        preds = predict_batch(model, aa2id, id2ss, device, seqs, args.batch_size)
        df, summary = write_results(args.out_csv, names, seqs, preds, gold_q8=golds)
        print(f"[info] wrote {args.out_csv}")
        if summary:
            print(f"[summary] overall Q8={summary['overall_Q8']:.4f} | Q3={summary['overall_Q3']:.4f}")
        return

    if args.fasta:
        names, seqs = read_fasta(args.fasta)
        preds = predict_batch(model, aa2id, id2ss, device, seqs, args.batch_size)
        df, summary = write_results(args.out_csv, names, seqs, preds, gold_q8=None)
        print(f"[info] wrote {args.out_csv}")
        return

if __name__ == "__main__":
    main()
