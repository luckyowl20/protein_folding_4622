#!/usr/bin/env python3
# predict_ss8.py
import argparse, os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Model (must match cnn.py)
# ----------------------------
PAD_ID = 0

class ResidualBlock(nn.Module):
    def __init__(self, d_model, ksize=7, dilation=1, p_drop=0.1):
        super().__init__()
        pad = ((ksize - 1)//2) * dilation
        self.norm1 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_model, ksize, padding=pad, dilation=dilation)
        self.norm2 = nn.LayerNorm(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, ksize, padding=pad, dilation=dilation)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        # x: (B,L,C)
        r = x
        x = self.norm1(x).transpose(1,2)
        x = F.gelu(self.conv1(x)).transpose(1,2)
        x = self.norm2(x).transpose(1,2)
        x = self.conv2(x).transpose(1,2)
        return self.dropout(x) + r

class CNNSecondary(nn.Module):
    def __init__(self, vocab_size, n_classes=8, emb_dim=64, d_model=256,
                 n_blocks=10, ksizes=(7,), dilations=(1,2,4,8,16), p_drop=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.in_proj = nn.Linear(emb_dim, d_model)
        blocks = []
        patt = list(zip(
            (list(ksizes) * ((n_blocks+len(ksizes)-1)//len(ksizes)))[:n_blocks],
            (list(dilations) * ((n_blocks+len(dilations)-1)//len(dilations)))[:n_blocks]
        ))
        for k,d in patt:
            blocks.append(ResidualBlock(d_model, ksize=k, dilation=d, p_drop=p_drop))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, x, mask=None):
        h = self.embed(x)      # (B,L,E)
        h = self.in_proj(h)    # (B,L,C)
        for blk in self.blocks:
            h = blk(h)
        logits = self.head(h)  # (B,L,K)
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(-1), -1e4)
        return logits

# ----------------------------
# Utils
# ----------------------------
def load_ckpt(ckpt_path, device=None):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]
    aa2id = ckpt["aa2id"]          # dict: aa -> int
    id2ss = {int(k): v for k, v in ckpt["id2ss"].items()}  # int -> 'H' etc.
    n_classes = len(ckpt["dssp8"])
    vocab_size = len(aa2id) + 1    # +PAD

    model = CNNSecondary(
        vocab_size=vocab_size,
        n_classes=n_classes,
        emb_dim=args["emb_dim"],
        d_model=args["d_model"],
        n_blocks=args["n_blocks"],
        ksizes=(args["kernel_size"],),
        dilations=tuple(args["dilations"]),
        p_drop=args["dropout"]
    )
    model.load_state_dict(ckpt["state_dict"])
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, aa2id, id2ss, device

def encode_seq(seq, aa2id):
    # Unknowns/remapped go to 'X' if present
    x_id = aa2id.get('X', max(aa2id.values())+1)
    return [aa2id.get(ch, x_id) for ch in seq.strip()]

def collate_pad(seqs, pad_id=PAD_ID):
    # seqs: list[list[int]]
    B = len(seqs)
    L = max(len(s) for s in seqs) if B > 0 else 0
    x = torch.full((B, L), pad_id, dtype=torch.long)
    m = torch.zeros((B, L), dtype=torch.bool)
    for i, s in enumerate(seqs):
        n = len(s)
        x[i, :n] = torch.tensor(s, dtype=torch.long)
        m[i, :n] = True
    return x, m

def q8_to_q3_map():
    # Must match training label order: ['H','E','G','I','B','T','S','C']
    # Mapping: H/G/I -> H, E/B -> E, T/S/C -> C
    return torch.tensor([0, 1, 0, 0, 1, 2, 2, 2], dtype=torch.long)

def q8_string_to_q3(q8_str):
    mapping = {'H':'H','G':'H','I':'H','E':'E','B':'E','T':'C','S':'C','C':'C'}
    return ''.join(mapping.get(c,'C') for c in q8_str)

def accuracy(a, b):
    # per-residue accuracy for two equal-length strings
    a = np.frombuffer(a.encode(), dtype='S1')
    b = np.frombuffer(b.encode(), dtype='S1')
    if len(a) == 0: return 0.0
    return float((a == b).sum()) / len(a)

# ----------------------------
# Prediction entry points
# ----------------------------
@torch.no_grad()
def predict_batch(model, aa2id, id2ss, device, seq_strs, batch_size=64):
    # Returns list of predicted Q8 strings in input order
    preds_all = []
    for i in range(0, len(seq_strs), batch_size):
        chunk = seq_strs[i:i+batch_size]
        toks = [encode_seq(s, aa2id) for s in chunk]
        x, m = collate_pad(toks)
        x, m = x.to(device), m.to(device)
        logits = model(x, m)
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
    seqs   = df['input'].astype(str).tolist()
    golds  = df['dssp8'].astype(str).tolist() if 'dssp8' in df.columns else None
    return chains, seqs, golds

def read_fasta(path):
    names, seqs = [], []
    cur_name, cur = None, []
    for line in open(path, 'r', encoding='utf-8'):
        line = line.strip()
        if not line: continue
        if line.startswith('>'):
            if cur_name is not None:
                names.append(cur_name); seqs.append(''.join(cur))
            cur_name = line[1:].strip()
            cur = []
        else:
            cur.append(line.replace(' ', '').upper())
    if cur_name is not None:
        names.append(cur_name); seqs.append(''.join(cur))
    return names, seqs

def write_results(out_csv, names, seqs, preds, gold_q8=None):
    import pandas as pd
    rows = []
    overall_q8_n = overall_q8_hit = 0
    overall_q3_n = overall_q3_hit = 0
    for i, (nm, s, p) in enumerate(zip(names, seqs, preds)):
        row = {"chain_id": nm, "length": len(s), "input": s, "pred_q8": p, "pred_q3": q8_string_to_q3(p)}
        if gold_q8 is not None:
            g8 = gold_q8[i]
            if len(g8) == len(p):
                q8_acc = accuracy(p, g8)
                row["gold_q8"] = g8
                row["gold_q3"] = q8_string_to_q3(g8)
                row["q8_acc"] = q8_acc
                q3_acc = accuracy(row["pred_q3"], row["gold_q3"])
                row["q3_acc"] = q3_acc
                # accumulate
                overall_q8_n += len(g8); overall_q8_hit += int(q8_acc * len(g8))
                overall_q3_n += len(g8); overall_q3_hit += int(q3_acc * len(g8))
            else:
                row["gold_q8"] = g8
                row["note"] = "length mismatch; per-seq accuracy skipped"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    summary = {}
    if overall_q8_n > 0:
        summary["overall_Q8"] = overall_q8_hit / overall_q8_n
        summary["overall_Q3"] = overall_q3_hit / overall_q3_n
    return df, summary

def main():
    ap = argparse.ArgumentParser(description="Predict DSSP Q8 (and Q3) from trained CNN checkpoint.")
    ap.add_argument("--ckpt", default="runs/ss8_cnn/best.pt", help="Path to checkpoint (best.pt)")
    ap.add_argument("--seq", help="Single amino-acid sequence to predict")
    ap.add_argument("--csv", help="CSV with columns input[,chain_id][,dssp8]")
    ap.add_argument("--fasta", help="FASTA file with sequences")
    ap.add_argument("--out_csv", default="predictions.csv", help="Output CSV for predictions")
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
