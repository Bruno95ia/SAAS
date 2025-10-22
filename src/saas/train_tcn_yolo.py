# src/saas/train_tcn_yolo.py
from __future__ import annotations
import argparse, csv, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class ClipDataset(Dataset):
    def __init__(self, labels_csv: Path, feats_dir: Path, seq=64, stride=2, classes=None):
        rows=[]
        with open(labels_csv) as f:
            for r in csv.reader(f):
                if not r or r[0].strip().startswith("#"): continue
                rows.append((r[0].strip(), r[1].strip()))
        self.classes = classes or sorted({lab for _,lab in rows})
        self.c2i = {c:i for i,c in enumerate(self.classes)}
        self.items=[]
        for rel, lab in rows:
            stem = Path(rel).stem
            fpath = feats_dir / f"{stem}.npz"
            if fpath.exists():
                self.items.append((fpath, self.c2i[lab]))
        self.seq, self.stride = seq, stride

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        fpath, y = self.items[idx]
        d = np.load(fpath)
        X = d["X"].astype(np.float32)      # (T,F)  F=5
        mask = d["mask"].astype(bool)      # (T,)
        # janela temporal
        valid = np.where(mask)[0]
        if len(valid) >= 4:
            t0 = int(np.clip(valid[0] - self.seq//4, 0, max(0, X.shape[0]-self.seq)))
        else:
            t0 = 0
        Xs = X[t0:t0+self.seq:self.stride]
        need = self.seq//self.stride - Xs.shape[0]
        if need > 0:
            Xs = np.concatenate([Xs, np.zeros((need, X.shape[1]), dtype=np.float32)], axis=0)
        return torch.from_numpy(Xs), torch.tensor(y, dtype=torch.long)

class TCN(nn.Module):
    def __init__(self, in_ch, n_classes, hid=64, levels=3, kernel=5, dropout=0.1):
        super().__init__()
        layers=[]; ch=in_ch
        for L in range(levels):
            out=hid*(2**L)
            layers += [
                nn.Conv1d(ch, out, kernel_size=kernel, padding="same"),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(out, out, kernel_size=kernel, padding="same"),
                nn.ReLU(),
            ]
            ch=out
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(ch, n_classes)

    def forward(self, x):   # x: (B,T,F)
        x = x.transpose(1,2)  # (B,F,T)
        h = self.net(x)       # (B,C,T)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ClipDataset(args.labels, args.feats, seq=args.seq, stride=args.stride)
    assert len(ds) > 1, "dataset vazio ou labels.csv sem pares válidos"
    n_classes = len(ds.classes)
    idx = list(range(len(ds))); random.shuffle(idx)
    n_tr = max(1, int(0.8*len(idx))); tr_idx=idx[:n_tr]; va_idx=idx[n_tr:]

    tr = DataLoader(torch.utils.data.Subset(ds,tr_idx), batch_size=args.bs, shuffle=True)
    va = DataLoader(torch.utils.data.Subset(ds,va_idx), batch_size=args.bs*2, shuffle=False)

    # inferir F (n_feats)
    sample_X, _ = next(iter(tr))
    in_feat = sample_X.shape[-1]

    model = TCN(in_ch=in_feat, n_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    cri = nn.CrossEntropyLoss()

    best_acc, best_sd = 0.0, None
    for ep in range(1, args.epochs+1):
        model.train(); los=0.0
        for X,y in tr:
            X,y = X.to(device), y.to(device)
            opt.zero_grad(); logits = model(X); loss=cri(logits,y); loss.backward(); opt.step()
            los += loss.item()*X.size(0)
        los /= len(tr.dataset)

        model.eval(); correct=0; total=0
        with torch.no_grad():
            for X,y in va:
                X,y = X.to(device), y.to(device)
                pred = model(X).argmax(1)
                correct += (pred==y).sum().item(); total += y.numel()
        acc = correct/max(1,total)
        print(f"epoch {ep} loss {los:.4f} val_acc {acc:.3f}")
        if acc > best_acc: best_acc, best_sd = acc, model.state_dict()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir/"tcn_best.pt"
    torch.save({"state_dict": best_sd, "classes": ds.classes, "in_feat": in_feat}, ckpt)
    print("saved:", ckpt)
    # export ONNX
    dummy = torch.zeros(1, args.seq//args.stride, in_feat)  # (B,T',F)
    model.load_state_dict(best_sd); model.eval()
    onnx_path = out_dir/"tcn.onnx"
    torch.onnx.export(model, dummy, onnx_path, input_names=["x"], output_names=["logits"], opset_version=17)
    with open(out_dir/"classes.txt","w") as f: f.write("\n".join(ds.classes))
    print("exported:", onnx_path)

def build_argparser():
    ap = argparse.ArgumentParser(description="Treino TCN em features YOLO (T×F)")
    ap.add_argument("--labels", type=Path, default=Path("labels.csv"))
    ap.add_argument("--feats",  type=Path, default=Path("runs/feats"))
    ap.add_argument("--out",    type=Path, default=Path("runs/models"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs",     type=int, default=32)
    ap.add_argument("--lr",     type=float, default=3e-4)
    ap.add_argument("--seq",    type=int, default=64, help="janela de frames")
    ap.add_argument("--stride", type=int, default=2,  help="subamostragem temporal")
    return ap

if __name__ == "__main__":
    train(build_argparser().parse_args())