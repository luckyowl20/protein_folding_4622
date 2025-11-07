#!/usr/bin/env python3
# cnn_bilstm.py
# CNN→FC→BiLSTM→FC head for DSSP8/Q3 sequence labeling (no pooling; preserves length)

import os, json, argparse
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Vocab / labels
# ----------------------------
AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWYXBZJUO")  # include odd tokens -> map to X
AA2ID = {aa: i+1 for i, aa in enumerate(AA_ALPHABET)}  # 0 = PAD
PAD_ID = 0

DSSP8 = ['H','E','G','I','B','T','S','C']               # P→C handled below
SS2ID = {c:i for i,c in enumerate(DSSP8)}
ID2SS = {i:c for c,i in SS2ID.items()}

Q3_LABELS = ['H','E','C']
Q8_TO_Q3 = torch.tensor([0,1,0,0,1,2,2,2], dtype=torch.long)  # H,E,G,I,B,T,S,C -> H,E,C

# ----------------------------
# Encoding / dataset
# ----------------------------
def encode_seq(seq):
    seq = seq.strip().upper()
    x_id = AA2ID.get('X', len(AA2ID))
    return [AA2ID.get(ch, x_id) for ch in seq]

def encode_ss(ss):
    ss = ss.strip().upper().replace('P','C')   # map polyproline to coil
    return [SS2ID[c] for c in ss]

class SS8Dataset(Dataset):
    def __init__(self, df):
        self.rows = []
        for _, r in df.iterrows():
            seq = str(r['input']).strip()
            ss  = str(r['dssp8']).strip()
            if len(seq)==0 or len(seq)!=len(ss):
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
    lens = [row["x"].shape[0] for row in batch]
    L = max(lens)
    x = torch.full((B, L), PAD_ID, dtype=torch.long)
    y = torch.zeros((B, L), dtype=torch.long)
    m = torch.zeros((B, L), dtype=torch.bool)
    ids = []
    for i,row in enumerate(batch):
        n = row["x"].shape[0]
        x[i,:n] = row["x"]; y[i,:n] = row["y"]; m[i,:n] = True
        ids.append(row["id"])
    return x, y, m, torch.tensor(lens, dtype=torch.long), ids

# ----------------------------
# Model blocks
# ----------------------------
class ConvBranch(nn.Module):
    def __init__(self, in_dim, out_dim, k):
        super().__init__()
        pad = (k-1)//2
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=k, padding=pad)
    def forward(self, x):              # x: (B,L,C)
        y = self.conv(x.transpose(1,2))
        return y.transpose(1,2)        # (B,L,out)

class CNN_BiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_classes=8,
        emb_dim=64,
        cnn_filters=96,          # per-branch
        kernels=(3,5,7,11),
        fc_cnn=256,              # dense after CNN concat
        lstm_hidden=256,         # per direction
        lstm_layers=2,
        p_drop=0.3
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)

        # multi-kernel CNN trunk (no pooling; keep L)
        self.branches = nn.ModuleList([ConvBranch(emb_dim, cnn_filters, k) for k in kernels])
        self.norm_cnn = nn.LayerNorm(cnn_filters*len(kernels))
        self.fc_cnn   = nn.Linear(cnn_filters*len(kernels), fc_cnn)
        self.drop1    = nn.Dropout(p_drop)

        # BiLSTM for global context (pack/pad outside)
        self.lstm = nn.LSTM(
            input_size=fc_cnn,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=p_drop if lstm_layers>1 else 0.0,
            bidirectional=True,
            batch_first=True
        )

        self.norm_lstm = nn.LayerNorm(2*lstm_hidden)
        self.drop2     = nn.Dropout(p_drop)

        # final classifier head
        self.head = nn.Linear(2*lstm_hidden, n_classes)

    def forward(self, x, mask, lengths):
        # x: (B,L) long, mask: (B,L) bool, lengths: (B,)
        emb = self.embed(x)                       # (B,L,E)

        # CNN branches
        feats = [F.gelu(b(emb)) for b in self.branches]  # list of (B,L,F)
        h = torch.cat(feats, dim=-1)              # (B,L, F*k)
        h = self.norm_cnn(h)
        h = F.gelu(self.fc_cnn(h))                # (B,L, fc_cnn)
        h = self.drop1(h)

        # BiLSTM with packing
        lens_cpu = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(h, lens_cpu, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B,L, 2H)

        h = self.norm_lstm(h)
        h = self.drop2(h)
        logits = self.head(h)                     # (B,L,K)

        # safety: mask invalid positions with large negative logits (not required for CE but good practice)
        logits = logits.masked_fill(~mask.unsqueeze(-1), -1e4)
        return logits

# ----------------------------
# Loss & metrics
# ----------------------------
def masked_ce(logits, y, mask, class_weights=None, smooth=0.0):
    B,L,K = logits.shape
    logits = logits.reshape(B*L, K)
    y = y.reshape(B*L)
    mask = mask.reshape(B*L)
    if class_weights is not None and not torch.is_tensor(class_weights):
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=logits.device)
    try:
        ce = nn.CrossEntropyLoss(weight=class_weights, reduction='none', label_smoothing=float(smooth))
    except TypeError:
        ce = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    loss = ce(logits, y)
    return loss[mask].mean()

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss, steps = 0.0, 0
    correct_q8, total = 0, 0
    conf_q8 = np.zeros((len(DSSP8), len(DSSP8)), dtype=np.int64)
    conf_q3 = np.zeros((3,3), dtype=np.int64)
    correct_q3 = 0

    for x,y,m,lens,_ in loader:
        x,y,m,lens = x.to(device), y.to(device), m.to(device), lens.to(device)
        logits = model(x,m,lens)
        loss = masked_ce(logits, y, m)
        total_loss += loss.item(); steps += 1

        preds = logits.argmax(-1)
        mask = m
        correct_q8 += ((preds==y) & mask).sum().item()
        total += mask.sum().item()

        t8 = y[mask].view(-1).detach().cpu().numpy()
        p8 = preds[mask].view(-1).detach().cpu().numpy()
        for t,p in zip(t8,p8): conf_q8[t,p]+=1

        # Q3
        qmap = Q8_TO_Q3.to(y.device)
        t3 = qmap[y][mask].view(-1)
        p3 = qmap[preds][mask].view(-1)
        correct_q3 += (t3==p3).sum().item()
        t3 = t3.cpu().numpy(); p3=p3.cpu().numpy()
        for t,p in zip(t3,p3): conf_q3[t,p]+=1

    q8 = correct_q8 / max(1,total)
    q3 = correct_q3 / max(1,total)
    per_cls_q8 = {DSSP8[i]: (conf_q8[i,i]/conf_q8[i].sum() if conf_q8[i].sum()>0 else 0.0)
                  for i in range(len(DSSP8))}
    per_cls_q3 = {Q3_LABELS[i]: (conf_q3[i,i]/conf_q3[i].sum() if conf_q3[i].sum()>0 else 0.0)
                  for i in range(3)}
    return total_loss/max(1,steps), q8, per_cls_q8, conf_q8, q3, per_cls_q3, conf_q3

# ----------------------------
# Training wrapper
# ----------------------------
def compute_class_weights(df):
    from collections import Counter
    cnt = Counter()
    for ss in df['dssp8'].astype(str):
        cnt.update(ss.strip().upper().replace('P','C'))
    total = sum(cnt[c] for c in DSSP8)
    freqs = np.array([cnt.get(c,0)/max(1,total) for c in DSSP8], dtype=np.float32)
    inv = 1.0/np.clip(freqs,1e-6,1.0)
    inv = inv * (len(inv)/inv.sum())  # mean=1
    return inv

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[info] device: {device}")

    df = pd.read_csv(args.csv)
    n = len(df)
    val_n = int(args.val_rows) if args.val_rows is not None else int(n*args.val_frac)
    train_df = df.iloc[:n-val_n].reset_index(drop=True)
    val_df   = df.iloc[n-val_n:].reset_index(drop=True)
    print(f"[info] train rows: {len(train_df)} | val rows: {len(val_df)} | val chains: {len(val_df)}")

    cw = compute_class_weights(train_df) if args.class_weights else None
    if cw is not None: print(f"[info] class weights: {cw}")

    train_ds = SS8Dataset(train_df);  val_ds = SS8Dataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, pin_memory=(device.type=='cuda'))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate, pin_memory=(device.type=='cuda'))

    vocab_size = len(AA2ID)+1
    model = CNN_BiLSTM(
        vocab_size=vocab_size,
        n_classes=len(DSSP8),
        emb_dim=args.emb_dim,
        cnn_filters=args.cnn_filters,
        kernels=tuple(args.kernels),
        fc_cnn=args.fc_cnn,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        p_drop=args.dropout
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.out, exist_ok=True)
    history = {"epoch":[], "train_loss":[], "val_loss":[], "val_Q8":[], "val_Q3":[]}
    best_q8 = -1.0
    best_path = os.path.join(args.out, "best.pt")

    for epoch in range(1, args.epochs+1):
        model.train()
        tr_loss, steps = 0.0, 0
        for x,y,m,lens,_ in train_loader:
            x,y,m,lens = x.to(device), y.to(device), m.to(device), lens.to(device)
            logits = model(x,m,lens)
            loss = masked_ce(logits, y, m, class_weights=cw, smooth=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tr_loss += loss.item(); steps += 1
        tr_loss /= max(1,steps)

        val_loss, q8, per_q8, cm8, q3, per_q3, cm3 = eval_epoch(model, val_loader, device)
        print(f"[epoch {epoch:02d}] train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_Q8={q8:.4f} | val_Q3={q3:.4f}")
        print(" per-class Q8:", {k: round(v,4) for k,v in per_q8.items()})
        print(" per-class Q3:", {k: round(v,4) for k,v in per_q3.items()})

        history["epoch"].append(epoch); history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss); history["val_Q8"].append(q8); history["val_Q3"].append(q3)

        if q8 > best_q8:
            best_q8 = q8
            ckpt = {
                "state_dict": model.state_dict(),
                "args": {
                    "emb_dim": args.emb_dim,
                    "cnn_filters": args.cnn_filters,
                    "kernels": list(args.kernels),
                    "fc_cnn": args.fc_cnn,
                    "lstm_hidden": args.lstm_hidden,
                    "lstm_layers": args.lstm_layers,
                    "dropout": args.dropout
                },
                "aa2id": AA2ID,
                "id2ss": {int(k):v for k,v in ID2SS.items()},
                "dssp8": DSSP8,
                "best_val_q8": float(q8),
                "best_val_q3": float(q3),
            }
            torch.save(ckpt, best_path)
            with open(os.path.join(args.out,"val_summary.json"),"w") as f:
                json.dump({
                    "val_Q8": float(q8), "per_class_acc_Q8": {k: float(v) for k,v in per_q8.items()},
                    "val_Q3": float(q3), "per_class_acc_Q3": {k: float(v) for k,v in per_q3.items()},
                    "val_chains": len(val_ds)
                }, f, indent=2)
            print(f"[info] saved best checkpoint (Q8={q8:.4f}) to {best_path}")

    # Write history + plots
    hist = pd.DataFrame(history)
    hist.to_csv(os.path.join(args.out,"history_cnn_bilstm.csv"), index=False)
    print("[info] wrote history_cnn_bilstm.csv")

    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(); plt.plot(hist["epoch"], hist["train_loss"], label="train_loss")
        plt.plot(hist["epoch"], hist["val_loss"], label="val_loss")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("CNN-BiLSTM Loss"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.out,"cnn_bilstm_loss.png")); plt.close(fig)

        fig = plt.figure(); plt.plot(hist["epoch"], hist["val_Q8"], label="val_Q8")
        plt.plot(hist["epoch"], hist["val_Q3"], label="val_Q3")
        plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("CNN-BiLSTM Q8/Q3"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.out,"cnn_bilstm_acc.png")); plt.close(fig)
        print("[info] wrote cnn_bilstm_loss.png and cnn_bilstm_acc.png")
    except Exception as e:
        print(f"[warn] plotting failed: {e}")

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="CNN→BiLSTM Q8/Q3 trainer (no pooling; preserves length).")
    ap.add_argument("--csv", required=True, help="Dataset CSV with columns: input,dssp8[,chain_id]")
    ap.add_argument("--out", default="runs/cnn_bilstm", help="Output directory")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--val_rows", type=int, default=None)
    # model sizes
    ap.add_argument("--emb_dim", type=int, default=64)
    ap.add_argument("--cnn_filters", type=int, default=96)
    ap.add_argument("--kernels", type=int, nargs="+", default=[3,5,7,11])
    ap.add_argument("--fc_cnn", type=int, default=256)
    ap.add_argument("--lstm_hidden", type=int, default=256)
    ap.add_argument("--lstm_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    # training
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
