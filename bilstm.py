#!/usr/bin/env python3
# bilstm.py
# BiLSTM secondary-structure predictor (Q8/Q3) with the same UX as your cnn.py.

import os, json, argparse, math, random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Labels / vocab
# ----------------------------
AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWYXBZJUO")  # include uncommon letters -> map to X
AA2ID = {aa: i+1 for i, aa in enumerate(AA_ALPHABET)}  # 0 reserved for PAD
PAD_ID = 0

# DSSP8 (we ignore 'P' polyproline -> map to 'C')
DSSP8 = ['H','E','G','I','B','T','S','C']
SS2ID = {c:i for i,c in enumerate(DSSP8)}
ID2SS = {i:c for c,i in SS2ID.items()}

# Q3 mapping (H/G/I->H, E/B->E, T/S/C->C)
Q3_LABELS = ['H','E','C']
Q8_TO_Q3 = torch.tensor([0,1,0,0,1,2,2,2], dtype=torch.long)  # idx over DSSP8

# ----------------------------
# Encoding utils
# ----------------------------
def encode_seq(seq):
    seq = seq.strip().upper()
    x_id = AA2ID.get('X', len(AA2ID))  # unknowns -> X
    return [AA2ID.get(ch, x_id) for ch in seq]

def encode_ss(ss):
    # Map any stray 'P' to 'C'
    ss = ss.strip().upper().replace('P', 'C')
    return [SS2ID[c] for c in ss]

# ----------------------------
# Dataset
# expected CSV columns:
#   - input (string of amino acids)
#   - dssp8 (string of same length)
# optional:
#   - chain_id (for logging)
#   - first_res (ignored)
# ----------------------------
class SS8Dataset(Dataset):
    def __init__(self, df):
        self.rows = []
        for _, r in df.iterrows():
            seq = str(r['input']).strip()
            ss  = str(r['dssp8']).strip()
            if len(seq)==0 or len(seq)!=len(ss):  # skip malformed
                continue
            self.rows.append({
                "id": str(r['chain_id']) if 'chain_id' in df.columns else "",
                "x": torch.tensor(encode_seq(seq), dtype=torch.long),
                "y": torch.tensor(encode_ss(ss), dtype=torch.long),
            })

    def __len__(self): return len(self.rows)

    def __getitem__(self, i): return self.rows[i]

def collate(batch):
    B = len(batch)
    lengths = [row["x"].shape[0] for row in batch]
    L = max(lengths)
    x = torch.full((B, L), PAD_ID, dtype=torch.long)
    y = torch.full((B, L), 0, dtype=torch.long)
    m = torch.zeros((B, L), dtype=torch.bool)
    ids = []
    for i, row in enumerate(batch):
        n = row["x"].shape[0]
        x[i,:n] = row["x"]
        y[i,:n] = row["y"]
        m[i,:n] = True
        ids.append(row["id"])
    return x, y, m, torch.tensor(lengths, dtype=torch.long), ids

# ----------------------------
# Model: BiLSTM stack
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
        self.norm = nn.LayerNorm(2*hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2*hidden, n_classes)

    def forward(self, x, mask, lengths):
        # x: (B,L) longs, mask: (B,L) bool, lengths: (B,)
        emb = self.embed(x)  # (B,L,E)
        # pack by lengths (cpu tensor required)
        lengths_cpu = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B,L,2H)
        # pad_packed_sequence returns max_L that equals input max length when enforce_sorted=False
        h = self.norm(h)
        h = self.dropout(h)
        logits = self.head(h)  # (B,L,K)
        # mask out padded positions (optional, for safety in some reductions)
        logits = logits.masked_fill(~mask.unsqueeze(-1), -1e4)
        return logits

# ----------------------------
# Loss (masked cross-entropy)
# ----------------------------
def masked_ce(logits, y, mask, class_weights=None, smooth=0.0):
    # logits: (B,L,K), y: (B,L) long, mask: (B,L) bool
    B, L, K = logits.shape
    logits = logits.reshape(B*L, K)
    y = y.reshape(B*L)
    mask = mask.reshape(B*L)
    if class_weights is not None and not torch.is_tensor(class_weights):
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=logits.device)

    try:
        ce = nn.CrossEntropyLoss(weight=class_weights, reduction='none', label_smoothing=float(smooth))
    except TypeError:
        # fallback for very old torch
        ce = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    loss = ce(logits, y)
    loss = loss[mask]  # keep valid positions only
    return loss.mean()

# ----------------------------
# Metrics
# ----------------------------
@torch.no_grad()
def eval_epoch(model, loader, device, class_weights=None, smooth=0.0):
    model.eval()
    total_loss, steps = 0.0, 0
    correct_q8, total = 0, 0
    conf_q8 = np.zeros((len(DSSP8), len(DSSP8)), dtype=np.int64)

    # Q3 bookkeeping
    conf_q3 = np.zeros((3,3), dtype=np.int64)
    correct_q3 = 0

    for x, y, m, lengths, _ in loader:
        x, y, m, lengths = x.to(device), y.to(device), m.to(device), lengths.to(device)
        logits = model(x, m, lengths)
        loss = masked_ce(logits, y, m, class_weights=None, smooth=0.0)
        total_loss += loss.item(); steps += 1

        preds = logits.argmax(-1)  # (B,L)
        mask_idx = m
        correct_q8 += ((preds == y) & mask_idx).sum().item()
        total += mask_idx.sum().item()

        # confusion Q8
        t8 = y[mask_idx].view(-1).detach().cpu().numpy()
        p8 = preds[mask_idx].view(-1).detach().cpu().numpy()
        for t, p in zip(t8, p8): conf_q8[t,p] += 1

        # Q3
        q8_to_q3 = Q8_TO_Q3.to(y.device)
        t3 = q8_to_q3[y][mask_idx].view(-1)
        p3 = q8_to_q3[preds][mask_idx].view(-1)
        correct_q3 += (t3 == p3).sum().item()

        t3 = t3.cpu().numpy(); p3 = p3.cpu().numpy()
        for t,p in zip(t3,p3): conf_q3[t,p] += 1

    q8 = correct_q8 / max(1,total)
    per_class_q8 = {}
    for i,c in enumerate(DSSP8):
        denom = conf_q8[i].sum()
        per_class_q8[c] = (conf_q8[i,i]/denom) if denom>0 else 0.0

    q3 = correct_q3 / max(1,total)
    per_class_q3 = {}
    for i,c in enumerate(Q3_LABELS):
        denom = conf_q3[i].sum()
        per_class_q3[c] = (conf_q3[i,i]/denom) if denom>0 else 0.0

    return total_loss/max(1,steps), q8, per_class_q8, conf_q8, q3, per_class_q3, conf_q3

# ----------------------------
# Training
# ----------------------------
def compute_class_weights(df):
    # compute per-token frequency over dssp8
    counts = Counter()
    for ss in df['dssp8'].astype(str):
        ss = ss.strip().upper().replace('P','C')
        counts.update(ss)
    total = sum(counts[c] for c in DSSP8)
    freqs = np.array([counts.get(c,0)/max(1,total) for c in DSSP8], dtype=np.float32)
    # inverse frequency, normalize to mean=1
    inv = 1.0 / np.clip(freqs, 1e-6, 1.0)
    inv = inv * (len(inv)/inv.sum())
    return inv

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[info] device: {device}")

    # I/O
    df = pd.read_csv(args.csv)
    # split by row order (like your cnn.py): last N for val
    n = len(df)
    val_n = int(n * args.val_frac) if args.val_rows is None else int(args.val_rows)
    train_df = df.iloc[:n-val_n].reset_index(drop=True)
    val_df   = df.iloc[n-val_n:].reset_index(drop=True)

    print(f"[info] train rows: {len(train_df)} | val rows: {len(val_df)} | val chains: {len(val_df)}")

    class_weights = None
    if args.class_weights:
        class_weights = compute_class_weights(train_df)
        print(f"[info] class weights: {class_weights}")

    train_ds = SS8Dataset(train_df)
    val_ds   = SS8Dataset(val_df)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate, pin_memory=(device.type=='cuda'))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, collate_fn=collate, pin_memory=(device.type=='cuda'))

    vocab_size = len(AA2ID)+1
    model = BiLSTMSS(vocab_size=vocab_size,
                     n_classes=len(DSSP8),
                     emb_dim=args.emb_dim,
                     hidden=args.hidden,
                     n_layers=args.layers,
                     dropout=args.dropout).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.out, exist_ok=True)

    history = {"epoch":[], "train_loss":[], "val_loss":[], "val_Q8":[], "val_Q3":[]}
    best_q8, best_path = -1.0, os.path.join(args.out, "best.pt")

    for epoch in range(1, args.epochs+1):
        model.train()
        tr_loss, steps = 0.0, 0
        for x, y, m, lengths, _ in train_loader:
            x, y, m, lengths = x.to(device), y.to(device), m.to(device), lengths.to(device)
            logits = model(x, m, lengths)
            loss = masked_ce(logits, y, m, class_weights=class_weights, smooth=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tr_loss += loss.item(); steps += 1

        tr_loss /= max(1,steps)
        val_loss, q8, per_cls_q8, conf_q8, q3, per_cls_q3, conf_q3 = eval_epoch(
            model, val_loader, device, class_weights=None, smooth=0.0
        )

        print(f"[epoch {epoch:02d}] train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_Q8={q8:.4f} | val_Q3={q3:.4f}")
        print(" per-class Q8:", {k: round(v,4) for k,v in per_cls_q8.items()})
        print(" per-class Q3:", {k: round(v,4) for k,v in per_cls_q3.items()})

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_Q8"].append(q8)
        history["val_Q3"].append(q3)

        # Save best by Q8
        if q8 > best_q8:
            best_q8 = q8
            ckpt = {
                "state_dict": model.state_dict(),
                "args": {
                    "emb_dim": args.emb_dim,
                    "hidden": args.hidden,
                    "layers": args.layers,
                    "dropout": args.dropout,
                },
                "aa2id": AA2ID,
                "id2ss": {int(k):v for k,v in ID2SS.items()},
                "dssp8": DSSP8,
                "best_val_q8": float(q8),
                "best_val_q3": float(q3),
            }
            torch.save(ckpt, best_path)
            # summary json
            with open(os.path.join(args.out,"val_summary.json"),"w") as f:
                json.dump({
                    "val_Q8": float(q8),
                    "per_class_acc_Q8": {k: float(v) for k,v in per_cls_q8.items()},
                    "val_Q3": float(q3),
                    "per_class_acc_Q3": {k: float(v) for k,v in per_cls_q3.items()},
                    "val_chains": len(val_ds)
                }, f, indent=2)
            print(f"[info] saved best checkpoint (Q8={q8:.4f}) to {best_path}")

    # write history + plots
    hist_df = pd.DataFrame(history)
    hist_csv = os.path.join(args.out, "history_bilstm.csv")
    hist_df.to_csv(hist_csv, index=False)
    print(f"[info] wrote {hist_csv}")

    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(); plt.plot(history["epoch"], history["train_loss"], label="train_loss")
        plt.plot(history["epoch"], history["val_loss"], label="val_loss")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("BiLSTM Loss"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.out,"bilstm_loss.png")); plt.close(fig)

        fig = plt.figure(); plt.plot(history["epoch"], history["val_Q8"], label="val_Q8")
        plt.plot(history["epoch"], history["val_Q3"], label="val_Q3")
        plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("BiLSTM Q8/Q3"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.out,"bilstm_acc.png")); plt.close(fig)
        print("[info] wrote bilstm_loss.png and bilstm_acc.png")
    except Exception as e:
        print(f"[warn] plotting failed: {e}")

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="BiLSTM Q8/Q3 trainer (compatible with cnn.py data).")
    ap.add_argument("--csv", required=True, help="Dataset CSV (columns: input,dssp8[,chain_id,...])")
    ap.add_argument("--out", default="runs/bilstm", help="Output dir")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.10, help="If val_rows not set, use this fraction for validation.")
    ap.add_argument("--val_rows", type=int, default=None, help="Override exact number of val rows.")
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256, help="Hidden size per direction")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--class_weights", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    train(args)
