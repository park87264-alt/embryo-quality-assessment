
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '08_sfu_segmentation_expert' / 'src'))
from model import ResNet18UNet

PHASES = ['tPB2','tPNa','tPNf','t2','t3','t4','t5','t6','t7','t8','t9plus','tM','tSB','tB','tEB','tHB']
STRUCTURES = ['ICM','TE','ZP']
RUN_RE = re.compile(r'RUN(\d+)')


def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def frame_no(path: Path) -> int:
    m = RUN_RE.search(path.name)
    return int(m.group(1)) if m else -1


def phase_for_frame(row, frame:int) -> int:
    # Use Nantes event-derived phase intervals from the existing 16-label manifest.
    for i, ph in enumerate(PHASES):
        if int(row.get(f'{ph}_present', 0) or 0) != 1:
            continue
        s = row.get(f'{ph}_start_frame', np.nan); e = row.get(f'{ph}_end_frame', np.nan)
        if pd.isna(s) or pd.isna(e):
            continue
        if int(float(s)) <= frame <= int(float(e)):
            return i
    present = []
    for i, ph in enumerate(PHASES):
        if int(row.get(f'{ph}_present', 0) or 0) == 1 and not pd.isna(row.get(f'{ph}_start_frame', np.nan)):
            present.append((abs(frame - int(float(row[f'{ph}_start_frame']))), i))
    return min(present)[1] if present else -1


def sample_images(img_dir: Path, frames_per_embryo:int):
    files = sorted(img_dir.glob('*.jpeg'), key=frame_no)
    if not files:
        return []
    if len(files) <= frames_per_embryo:
        return files
    idx = np.linspace(0, len(files)-1, frames_per_embryo).round().astype(int)
    return [files[i] for i in idx]


def resize_mask(m: torch.Tensor, size:int):
    return F.interpolate(m.unsqueeze(0), size=(size,size), mode='bilinear', align_corners=False).squeeze(0)


def mask_features(mask_probs: np.ndarray, gray: np.ndarray):
    feats=[]
    eps=1e-6
    h,w=gray.shape
    union = np.zeros((h,w), dtype=bool)
    areas=[]
    for c in range(mask_probs.shape[0]):
        prob=mask_probs[c]
        binary=prob>0.5
        union |= binary
        area=float(binary.mean()); areas.append(area)
        soft_area=float(prob.mean())
        if binary.any():
            ys,xs=np.where(binary)
            cx=float(xs.mean()/max(w-1,1)); cy=float(ys.mean()/max(h-1,1))
            bw=float((xs.max()-xs.min()+1)/w); bh=float((ys.max()-ys.min()+1)/h)
            bbox_area=bw*bh
            # Perimeter proxy: edge transitions normalized by image area.
            per=float((np.abs(np.diff(binary.astype(np.float32),axis=0)).sum()+np.abs(np.diff(binary.astype(np.float32),axis=1)).sum())/(h*w))
            mean_int=float(gray[binary].mean()/255.0); std_int=float(gray[binary].std()/255.0)
        else:
            cx=cy=bw=bh=bbox_area=per=mean_int=std_int=0.0
        feats.extend([area, soft_area, cx, cy, bw, bh, bbox_area, per, mean_int, std_int])
    total=sum(areas)+eps
    feats.extend([areas[0]/total, areas[1]/total, areas[2]/total, (areas[0]+areas[1])/(areas[2]+eps), float(union.mean())])
    return feats


def extract_curves(args):
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    manifest=pd.read_csv(args.manifest)
    if args.max_embryos and args.max_embryos < len(manifest):
        manifest=manifest.sample(n=args.max_embryos, random_state=args.seed).sort_values('embryo_id')
    device=torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    model=ResNet18UNet(out_channels=3, pretrained=False).to(device)
    ck=torch.load(args.seg_ckpt, map_location=device)
    model.load_state_dict(ck.get('model', ck), strict=True)
    model.eval()
    tf=transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    X=[]; Y=[]; M=[]; embryo_ids=[]; records=[]
    feature_names=[]
    for s in STRUCTURES:
        feature_names += [f'{s}_area_bin',f'{s}_area_soft',f'{s}_cx',f'{s}_cy',f'{s}_bbox_w',f'{s}_bbox_h',f'{s}_bbox_area',f'{s}_perimeter_proxy',f'{s}_int_mean',f'{s}_int_std']
    feature_names += ['ICM_area_share','TE_area_share','ZP_area_share','ICM_TE_to_ZP_ratio','union_area']
    with torch.no_grad():
        total_rows=len(manifest)
        for ridx, row in manifest.iterrows():
            eid=str(row['embryo_id'])
            img_dir=Path(row['processed_F0_dir']) if isinstance(row.get('processed_F0_dir'), str) else Path(args.nantes_root)/'embryo_dataset'/eid
            imgs=sample_images(img_dir, args.frames_per_embryo)
            if len(imgs)<args.min_frames:
                continue
            seq_x=[]; seq_y=[]; seq_m=[]
            loaded=[]; grays=[]; frames=[]
            for p in imgs:
                img=Image.open(p).convert('RGB')
                loaded.append(tf(img))
                grays.append(np.array(img.resize((args.image_size,args.image_size)).convert('L')))
                frames.append(frame_no(p))
            probs=[]
            for b0 in range(0, len(loaded), args.extract_batch_size):
                batch=torch.stack(loaded[b0:b0+args.extract_batch_size], dim=0).to(device)
                probs.append(torch.sigmoid(model(batch)).detach().cpu().numpy())
            probs=np.concatenate(probs, axis=0)
            for t, (p, prob, gray, fr) in enumerate(zip(imgs, probs, grays, frames)):
                feats=mask_features(prob, gray)
                y=phase_for_frame(row, fr)
                seq_x.append(feats); seq_y.append(y if y>=0 else 0); seq_m.append(1 if y>=0 else 0)
                records.append({'embryo_id':eid,'sample_index':t,'frame':fr,'phase':PHASES[y] if y>=0 else 'NA', **{k:v for k,v in zip(feature_names,feats)}})
            while len(seq_x)<args.frames_per_embryo:
                seq_x.append([0.0]*len(feature_names)); seq_y.append(0); seq_m.append(0)
            X.append(seq_x[:args.frames_per_embryo]); Y.append(seq_y[:args.frames_per_embryo]); M.append(seq_m[:args.frames_per_embryo]); embryo_ids.append(eid)
            if len(embryo_ids) % 50 == 0:
                print(f'extracted {len(embryo_ids)} embryos / {total_rows}', flush=True)
    X=np.asarray(X,dtype=np.float32); Y=np.asarray(Y,dtype=np.int64); M=np.asarray(M,dtype=np.float32)
    # Add temporal deltas and normalized time as explicit curve dynamics.
    dX=np.zeros_like(X); dX[:,1:,:]=X[:,1:,:]-X[:,:-1,:]
    time=np.linspace(0,1,X.shape[1],dtype=np.float32)[None,:,None].repeat(X.shape[0],axis=0)
    X_aug=np.concatenate([X,dX,time],axis=2)
    feature_names_aug=feature_names+[f'delta_{n}' for n in feature_names]+['time_norm']
    np.savez_compressed(out_dir/'nantes_structure_curve_sequences.npz', X=X_aug, y=Y, mask=M, embryo_ids=np.asarray(embryo_ids), feature_names=np.asarray(feature_names_aug), phases=np.asarray(PHASES))
    pd.DataFrame(records).to_csv(out_dir/'nantes_structure_curve_frame_features.csv', index=False)
    summary={
        'n_embryos': int(len(embryo_ids)), 'frames_per_embryo': int(args.frames_per_embryo),
        'valid_frame_labels': int(M.sum()), 'feature_dim': int(X_aug.shape[-1]),
        'segmentation_checkpoint': str(args.seg_ckpt), 'feature_file': str(out_dir/'nantes_structure_curve_sequences.npz'),
        'csv_file': str(out_dir/'nantes_structure_curve_frame_features.csv'),
        'note': 'Pseudo masks are generated by the SFU-trained ICM/TE/ZP structure expert and used only as structural temporal pretraining signals on Nantes.'
    }
    (out_dir/'curve_extraction_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))


class CurveDataset(Dataset):
    def __init__(self, X,y,m,indices):
        self.X=torch.from_numpy(X[indices]).float(); self.y=torch.from_numpy(y[indices]).long(); self.m=torch.from_numpy(m[indices]).float()
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i], self.y[i], self.m[i]


class TemporalCurveEncoder(nn.Module):
    def __init__(self, in_dim:int, hidden:int, n_classes:int, dropout:float=0.2):
        super().__init__()
        self.proj=nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru=nn.GRU(hidden, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.head=nn.Sequential(nn.LayerNorm(hidden*2), nn.Dropout(dropout), nn.Linear(hidden*2, n_classes))
    def forward(self,x):
        z=self.proj(x); z,_=self.gru(z); return self.head(z)


def eval_model(model, loader, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,y,m in loader:
            x=x.to(device); logits=model(x).cpu(); pred=logits.argmax(-1)
            active=m>0
            ys.extend(y[active].numpy().tolist()); ps.extend(pred[active].numpy().tolist())
    return {
        'accuracy': float(accuracy_score(ys, ps)),
        'balanced_accuracy': float(balanced_accuracy_score(ys, ps)),
        'macro_f1': float(f1_score(ys, ps, average='macro', zero_division=0)),
        'n_frames': int(len(ys)),
    }


def train_pretrain(args):
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data=np.load(args.curve_npz, allow_pickle=True)
    X=data['X'].astype(np.float32); y=data['y']; m=data['mask']; phases=[str(x) for x in data['phases']]
    n=len(X); all_idx=np.arange(n)
    results=[]
    for seed in args.seeds:
        set_seed(seed)
        train_idx, test_idx=train_test_split(all_idx, test_size=0.3, random_state=seed)
        train_ds=CurveDataset(X,y,m,train_idx); test_ds=CurveDataset(X,y,m,test_idx)
        train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True)
        test_loader=DataLoader(test_ds,batch_size=args.batch_size,shuffle=False)
        device=torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
        model=TemporalCurveEncoder(X.shape[-1], args.hidden, len(phases), args.dropout).to(device)
        counts=np.bincount(y[train_idx][m[train_idx]>0].reshape(-1), minlength=len(phases)).astype(np.float32)
        weights=counts.sum()/np.maximum(counts,1); weights=weights/weights.mean()
        weights=torch.from_numpy(weights).float().to(device)
        opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best={'macro_f1':-1}; best_state=None; history=[]
        for epoch in range(1,args.epochs+1):
            model.train(); losses=[]
            for x,yy,mm in train_loader:
                x=x.to(device); yy=yy.to(device); mm=mm.to(device)
                logits=model(x)
                active=mm>0
                loss=F.cross_entropy(logits[active], yy[active], weight=weights)
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                losses.append(float(loss.item()))
            metrics=eval_model(model,test_loader,device); metrics['epoch']=epoch; metrics['train_loss']=float(np.mean(losses))
            history.append(metrics)
            if metrics['macro_f1']>best['macro_f1']:
                best=metrics; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        seed_dir=out_dir/f'seed_{seed}'; seed_dir.mkdir(exist_ok=True)
        torch.save({'model':best_state,'in_dim':X.shape[-1],'hidden':args.hidden,'phases':phases,'best_metrics':best}, seed_dir/'temporal_curve_encoder_best.pt')
        pd.DataFrame(history).to_csv(seed_dir/'history.csv',index=False)
        (seed_dir/'best_metrics.json').write_text(json.dumps(best,indent=2,ensure_ascii=False),encoding='utf-8')
        results.append({'seed':seed, **best, 'checkpoint':str(seed_dir/'temporal_curve_encoder_best.pt')})
    keys=['accuracy','balanced_accuracy','macro_f1']
    summary={'seeds':args.seeds,'task':'Nantes 16-phase prediction from pseudo-segmentation structural curves','per_seed':results,'aggregate':{}}
    for k in keys:
        vals=[r[k] for r in results]
        summary['aggregate'][k]={'mean':float(np.mean(vals)),'std':float(np.std(vals))}
    summary['interpretation']='If macro-F1 is above random chance (~0.0625 for 16 classes), the generated structural curves carry usable temporal development information. This is a pretraining signal, not a direct Gardner endpoint.'
    (out_dir/'temporal_curve_pretrain_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='cmd', required=True)
    e=sub.add_parser('extract')
    e.add_argument('--manifest', default='/path/to/embryo_data/05_Metadata_Manifests/nantes_16phase_manifest.csv')
    e.add_argument('--nantes_root', default='/path/to/embryo_data/04_Nantes_Processed_1024_Final')
    e.add_argument('--seg_ckpt', default='/path/to/checkpoints/sfu_segmentation_best.pt')
    e.add_argument('--out_dir', required=True)
    e.add_argument('--frames_per_embryo', type=int, default=40)
    e.add_argument('--min_frames', type=int, default=12)
    e.add_argument('--image_size', type=int, default=224)
    e.add_argument('--extract_batch_size', type=int, default=32)
    e.add_argument('--max_embryos', type=int, default=0)
    e.add_argument('--device', default='cuda:0')
    e.add_argument('--seed', type=int, default=42)
    t=sub.add_parser('train')
    t.add_argument('--curve_npz', required=True)
    t.add_argument('--out_dir', required=True)
    t.add_argument('--epochs', type=int, default=35)
    t.add_argument('--batch_size', type=int, default=64)
    t.add_argument('--hidden', type=int, default=128)
    t.add_argument('--dropout', type=float, default=0.25)
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--weight_decay', type=float, default=1e-4)
    t.add_argument('--device', default='cuda:0')
    t.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
    args=p.parse_args()
    if args.cmd=='extract': extract_curves(args)
    else: train_pretrain(args)

if __name__=='__main__': main()
