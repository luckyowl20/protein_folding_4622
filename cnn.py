# ss8_cnn.py
import argparse, os, math, random, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ----------------------------
# Config / vocab
# ----------------------------
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"  # standard 20
PAD_ID = 0
AA2ID = {aa:i+1 for i,aa in enumerate(AA_VOCAB)}  # 1..20
AA2ID['X'] = len(AA2ID)+1                           # unknown
VOCAB_SIZE = len(AA2ID)+1                           # +PAD

DSSP8 = ['H','E','G','I','B','T','S','C']          # labels
SS2ID = {c:i for i,c in enumerate(DSSP8)}
ID2SS = {i:c for c,i in SS2ID.items()}
N_CLASSES = len(DSSP8)

# DSSP8 order used:
# ['H','E','G','I','B','T','S','C']  -> indices 0..7
Q3_LABELS = ['H', 'E', 'C']
# Map each Q8 index to a Q3 index: H/G/I -> H(0), E/B -> E(1), T/S/C -> C(2)
Q8_TO_Q3 = torch.tensor([0, 1, 0, 0, 1, 2, 2, 2], dtype=torch.long)  # shape [8]

def encode_seq(seq):
    return [AA2ID.get(ch, AA2ID['X']) for ch in seq]

def encode_ss(ss):
    remap = {'P': 'C'}  # map polyproline to coil
    return [SS2ID[remap.get(c, c)] for c in ss]

# ----------------------------
# Dataset & Collate
# ----------------------------
class SS8Dataset(Dataset):
    def __init__(self, rows):
        # rows: pandas DataFrame slice
        self.examples = []
        for _, r in rows.iterrows():
            seq = str(r['input']).strip()
            ss  = str(r['dssp8']).strip()
            if not seq or not ss or len(seq) != len(ss):  # skip bad rows
                continue
            self.examples.append({
                "chain_id": str(r['chain_id']),
                "x": torch.tensor(encode_seq(seq), dtype=torch.long),
                "y": torch.tensor(encode_ss(ss), dtype=torch.long)
            })

    def __len__(self): return len(self.examples)
    def __getitem__(self, i):
        e = self.examples[i]
        return e["x"], e["y"], e["chain_id"]

def collate_pad(batch):
    xs, ys, gids = zip(*batch)
    L = max(len(x) for x in xs)
    B = len(xs)
    xpad = torch.full((B, L), PAD_ID, dtype=torch.long)
    ypad = torch.zeros((B, L), dtype=torch.long)
    mask = torch.zeros((B, L), dtype=torch.bool)
    for i,(x,y) in enumerate(zip(xs,ys)):
        n = len(x)
        xpad[i,:n] = x
        ypad[i,:n] = y
        mask[i,:n] = True
    return xpad, ypad, mask, list(gids)

# ----------------------------
# Model
# ----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, d_model, ksize=7, dilation=1, p_drop=0.1):
        super().__init__()
        pad = ((ksize-1)//2) * dilation
        self.norm1 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_model, ksize, padding=pad, dilation=dilation)
        self.norm2 = nn.LayerNorm(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, ksize, padding=pad, dilation=dilation)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        # x: (B,L,C)
        r = x
        x = self.norm1(x).transpose(1,2)     # (B,C,L)
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
        # build repeating (ksize, dilation) pattern
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
        h = self.embed(x)            # (B,L,E)
        h = self.in_proj(h)          # (B,L,C)
        for blk in self.blocks:
            h = blk(h)
        logits = self.head(h)        # (B,L,K)
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(-1), -1e4)
        return logits

# ----------------------------
# Loss & Metrics
# ----------------------------
def masked_ce(logits, targets, mask, class_weights=None, smooth=0.05):
    # logits: (B,L,K), targets: (B,L), mask: (B,L)
    B,L,K = logits.shape
    logits = logits.reshape(B*L, K)
    targets = targets.reshape(B*L)
    mask = mask.reshape(B*L)
    if class_weights is not None:
        ce = nn.CrossEntropyLoss(weight=class_weights, reduction='none', label_smoothing=smooth)
    else:
        ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=smooth)
    loss_all = ce(logits, targets)
    loss = (loss_all * mask.float()).sum() / (mask.float().sum() + 1e-8)
    return loss

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss, steps = 0.0, 0
    correct_q8, count = 0, 0
    conf_q8 = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    conf_q3 = np.zeros((3, 3), dtype=np.int64)
    correct_q3 = 0

    for x, y, m, _ in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        logits = model(x, m)
        loss = masked_ce(logits, y, m, class_weights=None, smooth=0.0)
        total_loss += loss.item(); steps += 1

        preds = logits.argmax(-1)         # (B,L)
        mask_idx = m                      # (B,L) bool

        # ----- Q8 -----
        correct_q8 += ((preds == y) & mask_idx).sum().item()
        count += mask_idx.sum().item()
        t8 = y[mask_idx].view(-1).detach().cpu().numpy()
        p8 = preds[mask_idx].view(-1).detach().cpu().numpy()
        for t, p in zip(t8, p8):
            conf_q8[t, p] += 1

        # ----- Q3 (FIX: move mapping tensor to same device) -----
        q8_to_q3 = Q8_TO_Q3.to(y.device)      # <--- important line
        t3 = q8_to_q3[y][mask_idx].view(-1)   # tensor of ints on device
        p3 = q8_to_q3[preds][mask_idx].view(-1)

        correct_q3 += (t3 == p3).sum().item()

        t3 = t3.detach().cpu().numpy()
        p3 = p3.detach().cpu().numpy()
        for t, p in zip(t3, p3):
            conf_q3[t, p] += 1

    q8 = correct_q8 / max(count, 1)
    per_class_acc_q8 = {c: (conf_q8[i, i] / conf_q8[i].sum()) if conf_q8[i].sum() > 0 else 0.0
                        for i, c in enumerate(DSSP8)}

    q3 = correct_q3 / max(count, 1)
    Q3_LABELS = ['H', 'E', 'C']
    per_class_acc_q3 = {c: (conf_q3[i, i] / conf_q3[i].sum()) if conf_q3[i].sum() > 0 else 0.0
                        for i, c in enumerate(Q3_LABELS)}

    return total_loss / max(steps, 1), q8, per_class_acc_q8, conf_q8, q3, per_class_acc_q3, conf_q3

def class_weights_from_loader(loader, device):
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for _,y,m,_ in loader:
        y = y[m].view(-1).numpy()
        for v in y:
            counts[v]+=1
    w = (counts.sum() / (counts + 1e-8))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float, device=device)

# ----------------------------
# Split by chain_id (no leakage)
# ----------------------------
def split_by_chain(df, val_frac=0.1, seed=42):
    chains = df['chain_id'].astype(str).unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(chains)
    n_val = max(1, int(len(chains)*val_frac))
    val_chains = set(chains[:n_val])
    train_df = df[~df['chain_id'].isin(val_chains)].reset_index(drop=True)
    val_df   = df[df['chain_id'].isin(val_chains)].reset_index(drop=True)
    return train_df, val_df, sorted(list(val_chains))

# ----------------------------
# Training
# ----------------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"[info] device: {device}")

    df = pd.read_csv(args.csv)
    needed = {'chain_id','input','dssp8'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    train_df, val_df, val_chains = split_by_chain(df, val_frac=args.val_frac, seed=args.seed)
    print(f"[info] train rows: {len(train_df)} | val rows: {len(val_df)} | val chains: {len(val_chains)}")

    train_ds = SS8Dataset(train_df)
    val_ds   = SS8Dataset(val_df)

    # bucket by length (sort) to reduce padding
    def length_key(e): return len(e[0])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_pad, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_pad, num_workers=0)

    model = CNNSecondary(
        vocab_size=VOCAB_SIZE,
        n_classes=N_CLASSES,
        emb_dim=args.emb_dim,
        d_model=args.d_model,
        n_blocks=args.n_blocks,
        ksizes=(args.kernel_size,),
        dilations=tuple(args.dilations),
        p_drop=args.dropout
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=max(1,len(train_loader))
    )

    # class weights (from train)
    cw = class_weights_from_loader(train_loader, device) if args.class_weights else None
    print(f"[info] class weights: {cw.detach().cpu().numpy() if cw is not None else 'disabled'}")

    best_val = -1.0
    os.makedirs(args.out, exist_ok=True)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_Q8": [], "val_Q3": []} # for plotting

    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0; steps = 0
        for x,y,m,_ in train_loader:
            x,y,m = x.to(device), y.to(device), m.to(device)
            opt.zero_grad()
            logits = model(x, m)
            loss = masked_ce(logits, y, m, class_weights=cw, smooth=args.label_smoothing)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            running += loss.item(); steps += 1
        tr_loss = running/max(steps,1)

        val_loss, q8, per_cls_q8, conf_q8, q3, per_cls_q3, conf_q3 = eval_epoch(model, val_loader, device)
        print(f"[epoch {epoch:02d}] train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_Q8={q8:.4f} | val_Q3={q3:.4f}")
        print(" per-class Q8:", {k: round(v,4) for k,v in per_cls_q8.items()})
        print(" per-class Q3:", {k: round(v,4) for k,v in per_cls_q3.items()})


        # record history
        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_Q8"].append(q8)
        history["val_Q3"].append(q3)

        # save best by Q8
        if q8 > best_val:
            best_val = q8
            ckpt = {
                "state_dict": model.state_dict(),
                "args": vars(args),
                "aa2id": AA2ID,
                "id2ss": ID2SS,
                "dssp8": DSSP8
            }
            torch.save(ckpt, os.path.join(args.out, "best.pt"))
            with open(os.path.join(args.out,"val_summary.json"),"w") as f:
                json.dump({
                    "val_Q8": float(q8),
                    "per_class_acc_Q8": {k: float(v) for k,v in per_cls_q8.items()},
                    "val_Q3": float(q3),
                    "per_class_acc_Q3": {k: float(v) for k,v in per_cls_q3.items()},
                    "val_chains": val_chains
                }, f, indent=2)
            print(f"[info] saved best checkpoint (Q8={best_val:.4f}) to {args.out}/best.pt")

    print("[done] training complete.")

    # ---- save history.csv
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(args.out, "history.csv"), index=False)

    # ---- make learning-curve plots
    fig1 = plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss vs. Epoch")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "loss_curve.png")); plt.close(fig1)

    fig2 = plt.figure()
    plt.plot(history["epoch"], history["val_Q8"], label="val_Q8")
    plt.xlabel("epoch"); plt.ylabel("accuracy")
    plt.title("Validation Q8 vs. Epoch")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "q8_curve.png")); plt.close(fig2)

    fig3 = plt.figure()
    plt.plot(history["epoch"], history["val_Q3"], label="val_Q3")
    plt.xlabel("epoch"); plt.ylabel("accuracy")
    plt.title("Validation Q3 vs. Epoch")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "q3_curve.png")); plt.close(fig3)

    print(f"[info] wrote {os.path.join(args.out, 'history.csv')}")
    print(f"[info] wrote {os.path.join(args.out, 'loss_curve.png')}")
    print(f"[info] wrote {os.path.join(args.out, 'q8_curve.png')}")
    print(f"[info] wrote {os.path.join(args.out, 'q3_curve.png')}")


# ----------------------------
# Inference helper
# ----------------------------
@torch.no_grad()
def predict_string(seq, ckpt_path="best.pt", device=None):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]
    model = CNNSecondary(
        vocab_size=len(ckpt["aa2id"])+1,
        n_classes=len(ckpt["dssp8"]),
        emb_dim=args["emb_dim"], d_model=args["d_model"],
        n_blocks=args["n_blocks"], ksizes=(args["kernel_size"],),
        dilations=tuple(args["dilations"]), p_drop=args["dropout"]
    )
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x = torch.tensor([encode_seq(seq)], dtype=torch.long, device=device)
    m = torch.ones_like(x, dtype=torch.bool)
    logits = model(x, m)
    pred = logits.argmax(-1)[0].tolist()
    id2ss = {int(k):v for k,v in ckpt["id2ss"].items()}
    return "".join(id2ss[i] for i in pred)

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="DSSP8 CNN baseline")
    p.add_argument("--csv", required=True, help="CSV with columns chain_id,input,dssp8")
    p.add_argument("--out", default="runs/ss8_cnn", help="output dir")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")

    # model
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_blocks", type=int, default=10)
    p.add_argument("--kernel_size", type=int, default=7)
    p.add_argument("--dilations", type=int, nargs="+", default=[1,2,4,8,16])
    p.add_argument("--dropout", type=float, default=0.1)

    # training tricks
    p.add_argument("--class_weights", action="store_true", help="use class-imbalance weights")
    p.add_argument("--label_smoothing", type=float, default=0.05)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    train(args)
