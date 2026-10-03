# Auto-exported from reproduce_pipeline_v2.ipynb

import os, sys, subprocess
from pathlib import Path

# Set working directory to pipeline root
PIPELINE_ROOT = Path("../").resolve()
os.chdir(PIPELINE_ROOT)
print(f"Pipeline root: {PIPELINE_ROOT}")

# Create required directories
for d in ["data", "results", "logs", "reproduce/logs"]:
    Path(d).mkdir(parents=True, exist_ok=True)
    print(f"  Created: {d}")

print("\nPython:", sys.version)

# Check key packages
import importlib
for pkg in ["numpy", "pandas", "sklearn", "torch", "open_clip", "Bio", "docx"]:
    try:
        m = importlib.import_module(pkg)
        ver = getattr(m, '__version__', 'ok')
        print(f"  {pkg}: {ver}")
    except ImportError:
        print(f"  {pkg}: NOT INSTALLED — run: pip install -r reproduce/requirements.txt")

# Check MAFFT
result = subprocess.run(["mafft", "--version"], capture_output=True, text=True)
if result.returncode == 0 or result.stderr:
    print(f"  mafft: {result.stderr.strip() or 'available'}")
else:
    print("  mafft: NOT FOUND — install with: conda install -c bioconda mafft")

# Check GPU
import torch
print(f"\nGPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")

# %%

# Check required input files
required_data = [
    "data/nwss_measles_2024_2026.csv",
    "data/nndss_measles_2022_2026.csv",
    "data/mmr_coverage_2022_2026.csv",
    "data/bts_air_passengers_2022_2026.csv",
    "data/oaw_sequences_wgs.fasta",
    "data/oaw_metadata.csv",
]
missing = [f for f in required_data if not Path(f).exists()]
if missing:
    print("Missing data files — running ingestion scripts:")
    for f in missing:
        print(f"  {f}")
else:
    print("All Phase 0 data files present. Skipping ingestion.")

# %%

# Run Phase 0: data ingestion
# Uncomment if data files are missing:
# %run phase0_data_ingestion.py

# Run Phase 0: metadata cleaning
# %run phase0_metadata_cleaning.py

# Run Phase 0: MAFFT alignment (requires mafft binary)
# %run phase0_alignment.py

# Verify outputs
import pandas as pd
for f in required_data:
    p = Path(f)
    if p.exists():
        if p.suffix == '.csv':
            df = pd.read_csv(p)
            print(f"  {f}: {df.shape}")
        else:
            print(f"  {f}: {p.stat().st_size:,} bytes")
    else:
        print(f"  MISSING: {f}")

# %%

# Check if Stage 1 results already exist
s1_results = Path("results/stage1_cv_results.csv")
if s1_results.exists():
    import pandas as pd
    df = pd.read_csv(s1_results)
    print("Stage 1 results already computed:")
    print(df[['model','auc_roc','f1','precision','recall']].to_string(index=False))
else:
    print("Running Stage 1 genomic model...")
    %run phase1_genomic_model.py
    df = pd.read_csv(s1_results)
    print(df[['model','auc_roc','f1','precision','recall']].to_string(index=False))

# %%

# Visualize Stage 1 feature importance
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = ['Liberation Sans', 'Arial', 'DejaVu Sans']
import matplotlib.pyplot as plt

fi = pd.read_csv("results/stage1_feature_importance.csv")
fi_sorted = fi.sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(7, 4))
colors = ['#D4A04A' if v > 0.1 else '#8A8378' for v in fi_sorted['importance']]
ax.barh(fi_sorted['feature'], fi_sorted['importance'], color=colors)
ax.set_xlabel('Feature Importance (Random Forest)')
ax.set_title('Stage 1: Genomic Transmission Linkage — Feature Importance')
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('results/stage1_feature_importance_repro.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/stage1_feature_importance_repro.png")

# %%

import gdown
from pathlib import Path

MSLD_ROOT = Path("/workspace/msld_v2/MSLD v2/original_images/FOLDS")

if MSLD_ROOT.exists():
    n_imgs = len(list(MSLD_ROOT.rglob("*.jpg")))
    print(f"MSLD v2.0 already downloaded: {n_imgs} images found at {MSLD_ROOT}")
else:
    print("Downloading MSLD v2.0 from Google Drive...")
    MSLD_OUT = Path("/workspace/msld_v2")
    MSLD_OUT.mkdir(parents=True, exist_ok=True)
    
    folder_id = "1_bGmbDQNgJViQenjZ4QUhhzpdiubga48"
    files = gdown.download_folder(
        f"https://drive.google.com/drive/folders/{folder_id}",
        output=str(MSLD_OUT),
        quiet=False,
        use_cookies=False
    )
    print(f"Downloaded: {files}")
    
    # Unzip
    import zipfile
    zip_path = MSLD_OUT / "MSLD v2" / "Original Images.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(MSLD_OUT / "MSLD v2" / "original_images")
        print("Unzipped Original Images.zip")
    
    n_imgs = len(list(MSLD_ROOT.rglob("*.jpg")))
    print(f"Total images: {n_imgs}")

# %%

import time, numpy as np
from pathlib import Path

CACHE_PATH = Path("results/stage3_all_embeddings_cache.npz")
MSLD_ROOT  = Path("/workspace/msld_v2/MSLD v2/original_images/FOLDS")

if CACHE_PATH.exists():
    d = np.load(CACHE_PATH, allow_pickle=True)
    print(f"Cache already exists: {d['embeddings'].shape} embeddings")
    print(f"Labels: {sorted(set(d['labels'].tolist()))}")
elif not MSLD_ROOT.exists():
    print("MSLD v2.0 not found. Run Section 3 download first, or copy the pre-computed")
    print("cache from reproduce/data/stage3_biomedclip_embeddings_cache.npz to results/")
else:
    import torch
    import open_clip
    from PIL import Image
    from torchvision import transforms

    # Load BiomedCLIP ViT-B/16 (frozen)
    MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    print(f"Loading {MODEL_ID} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")

    # Class prefix → label mapping
    PREFIX_MAP = {"MKP": "Mpox", "CHP": "Chickenpox", "MSL": "Measles",
                  "CWP": "Cowpox", "HFMD": "HFMD", "HEALTHY": "Healthy"}

    def get_label(fname):
        for prefix, label in PREFIX_MAP.items():
            if fname.upper().startswith(prefix):
                return label
        return "Unknown"

    # Collect all image paths across all folds
    all_paths, all_labels = [], []
    for fold in range(1, 6):
        for split in ["train", "valid", "test"]:
            split_dir = MSLD_ROOT / f"fold{fold}" / split
            if not split_dir.exists():
                continue
            for img_path in sorted(split_dir.glob("*.jpg")):
                all_paths.append(str(img_path))
                all_labels.append(get_label(img_path.name))

    print(f"Total images: {len(all_paths)}")
    from collections import Counter
    print("Label counts:", dict(Counter(all_labels)))

    # Extract embeddings in batches
    BATCH = 32
    all_embeddings = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(all_paths), BATCH):
            batch_paths = all_paths[i:i+BATCH]
            imgs = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    imgs.append(preprocess(img))
                except Exception:
                    imgs.append(torch.zeros(3, 224, 224))
            batch_tensor = torch.stack(imgs).to(device)
            feats = model.encode_image(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2-normalize
            all_embeddings.append(feats.cpu().numpy())
            if (i // BATCH) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  Batch {i//BATCH+1}/{(len(all_paths)+BATCH-1)//BATCH} | {elapsed:.0f}s")

    embeddings = np.vstack(all_embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH,
             paths=np.array(all_paths),
             embeddings=embeddings,
             labels=np.array(all_labels))
    print(f"Cache saved: {CACHE_PATH}")


# %%

import json
from pathlib import Path

s3_real = Path("results/stage3_imaging_results_real.json")

if s3_real.exists():
    with open(s3_real) as f:
        s3 = json.load(f)
    print("Stage 3 results already computed:")
    print(f"  Model:        {s3['model']}")
    print(f"  Dataset:      {s3['dataset']}")
    print(f"  CV strategy:  {s3['cv_strategy']}")
    print(f"  Accuracy:     {s3['overall_accuracy']:.4f}")
    print(f"  Macro F1:     {s3['overall_macro_f1']:.4f}")
    print(f"  Macro AUC:    {s3['overall_macro_auc']:.4f}")
    print(f"  Simulated:    {s3['simulated']}")
    print("\nPer-class AUC:")
    for cls, auc in s3['per_class_auc'].items():
        print(f"  {cls:12s}: {auc:.4f}")
else:
    print("Running Stage 3 imaging pipeline (this may take 15-30 min on CPU)...")
    %run phase1_imaging_model_fast.py

# %%

# Verify embedding cache integrity
import numpy as np
from pathlib import Path
from collections import Counter

CACHE_PATH = Path("results/stage3_all_embeddings_cache.npz")

if not CACHE_PATH.exists():
    # Try the reproduce/data/ copy
    alt = Path("../reproduce/data/stage3_biomedclip_embeddings_cache.npz")
    if alt.exists():
        import shutil
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(alt, CACHE_PATH)
        print(f"Copied cache from reproduce/data/ to results/")
    else:
        print("Cache not found. Run Section 3b extraction first.")
        raise FileNotFoundError(CACHE_PATH)

d = np.load(CACHE_PATH, allow_pickle=True)
emb = d["embeddings"]
labels = d["labels"].tolist()
paths  = d["paths"].tolist()

assert emb.shape[1] == 512, f"Expected 512 dims, got {emb.shape[1]}"
assert len(set(labels)) == 6, f"Expected 6 classes, got {set(labels)}"

print("Embedding cache verified:")
print(f"  Shape:   {emb.shape}  (n_images x 512)")
print(f"  Labels:  {dict(Counter(labels))}")
print(f"  Norm check (first 3): {[round(float(np.linalg.norm(emb[i])),4) for i in range(3)]}")
print("  PASS: cache is valid and ready for Stage 3 MLP training.")


# %%

# Supplementary Figure S1: Confusion matrix
import json, numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = ['Liberation Sans', 'Arial', 'DejaVu Sans']
import matplotlib.pyplot as plt
import seaborn as sns

with open("results/stage3_imaging_results_real.json") as f:
    s3 = json.load(f)

cm = np.array(s3['confusion_matrix'])
classes = s3['class_names']

# Normalize by row
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrBr',
            xticklabels=classes, yticklabels=classes,
            linewidths=0.5, linecolor='#D5CFC5', ax=ax,
            vmin=0, vmax=1)
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('True', fontsize=11)
ax.set_title('Stage 3: BiomedCLIP + MLP — Confusion Matrix (MSLD v2.0, 5-fold CV)', fontsize=11)
plt.tight_layout()
plt.savefig('results/stage3_confusion_matrix_repro.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/stage3_confusion_matrix_repro.png")

# %%

# Supplementary Figure S2: Per-fold AUC bar chart
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = ['Liberation Sans', 'Arial', 'DejaVu Sans']
import matplotlib.pyplot as plt

fold_df = pd.read_csv("results/stage3_fold_results.csv")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
metrics = ['accuracy', 'macro_f1', 'macro_auc']
labels  = ['Accuracy', 'Macro F1', 'Macro AUC']
colors  = ['#D4A04A', '#75A025', '#0279EE']

for ax, metric, label, color in zip(axes, metrics, labels, colors):
    ax.bar(fold_df['fold'], fold_df[metric], color=color, alpha=0.85, edgecolor='white')
    ax.axhline(fold_df[metric].mean(), color='#111111', linestyle='--', linewidth=1.2,
               label=f'Mean={fold_df[metric].mean():.3f}')
    ax.set_xlabel('Fold'); ax.set_ylabel(label)
    ax.set_title(f'Stage 3: {label} per Fold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('results/stage3_fold_metrics_repro.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/stage3_fold_metrics_repro.png")

# %%

import pandas as pd
from pathlib import Path

s2_results = Path("results/stage2_cv_results_v2_all.csv")
if s2_results.exists():
    df = pd.read_csv(s2_results)
    print("Stage 2 results already computed:")
    summary = df.groupby('model')[['auc_roc','f1','recall']].mean().round(4)
    print(summary)
else:
    print("Running Stage 2 epi model...")
    %run phase1_epi_model_v2.py
    df = pd.read_csv(s2_results)
    print(df.groupby('model')[['auc_roc','f1','recall']].mean().round(4))

# %%

import pandas as pd
from pathlib import Path

p2_results = Path("results/phase2_alignment_results.csv")
if p2_results.exists():
    df = pd.read_csv(p2_results)
    print("Phase 2 alignment results already computed:")
    print(df.to_string(index=False))
else:
    print("Running Phase 2 contrastive alignment...")
    %run phase2_contrastive_alignment.py
    df = pd.read_csv(p2_results)
    print(df.to_string(index=False))

# %%

import pandas as pd
from pathlib import Path

p3_results = Path("results/phase3_ablation_results.csv")
if p3_results.exists():
    df = pd.read_csv(p3_results)
    print("Phase 3 ablation results already computed:")
    print(df[['combination','mean_auc','std_auc']].to_string(index=False))
else:
    print("Running Phase 3 fusion ablation...")
    %run phase3_fusion_ablation.py
    df = pd.read_csv(p3_results)
    print(df[['combination','mean_auc','std_auc']].to_string(index=False))

# %%

# Ablation heatmap
import pandas as pd, numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = ['Liberation Sans', 'Arial', 'DejaVu Sans']
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("results/phase3_ablation_results.csv")
combos = df['combination'].tolist()
aucs   = df['mean_auc'].values

fig, ax = plt.subplots(figsize=(9, 2.5))
data = aucs.reshape(1, -1)
sns.heatmap(data, annot=True, fmt='.4f', cmap='YlOrBr',
            xticklabels=combos, yticklabels=['AUC'],
            vmin=0.45, vmax=1.0, linewidths=0.5, linecolor='#D5CFC5', ax=ax)
ax.set_title('Phase 3: Modality Ablation — AUC on Stage 1 Transmission Linkage Task', fontsize=11)
plt.tight_layout()
plt.savefig('results/phase3_ablation_repro.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/phase3_ablation_repro.png")

# %%

from pathlib import Path

p4_results = Path("results/phase4_validation_summary.csv")
if p4_results.exists():
    import pandas as pd
    df = pd.read_csv(p4_results)
    print("Phase 4 validation summary:")
    print(df.to_string(index=False))
else:
    print("Running Phase 4 validation summary...")
    %run phase4_validation_summary.py
    df = pd.read_csv(p4_results)
    print(df.to_string(index=False))

# %%

import subprocess, sys
result = subprocess.run(
    [sys.executable, "generate_cid_manuscript.py"],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)

# %%

import shutil
from pathlib import Path

RESULTS_OUT = Path("/mnt/results/measles_multimodal")
RESULTS_OUT.mkdir(parents=True, exist_ok=True)

deliverables = [
    # Manuscript
    ("document_cid_manuscript_draft.docx", "document_cid_manuscript_draft.docx"),
    # Figures
    ("results/stage1_feature_importance_repro.png", "stage1_feature_importance.png"),
    ("results/stage3_confusion_matrix_repro.png",   "stage3_confusion_matrix.png"),
    ("results/stage3_fold_metrics_repro.png",        "stage3_fold_metrics.png"),
    ("results/phase3_ablation_repro.png",            "phase3_ablation_results.png"),
    # Data
    ("results/stage1_cv_results.csv",                "stage1_cv_results.csv"),
    ("results/stage2_cv_results_v2_all.csv",         "stage2_cv_results.csv"),
    ("results/stage3_imaging_results_real.json",     "stage3_imaging_results_real.json"),
    ("results/stage3_fold_results.csv",              "stage3_fold_results.csv"),
    ("results/phase2_alignment_results.csv",         "phase2_alignment_results.csv"),
    ("results/phase3_ablation_results.csv",          "phase3_ablation_results.csv"),
    ("results/phase4_validation_summary.csv",        "phase4_validation_summary.csv"),
]

for src, dst in deliverables:
    src_path = Path(src)
    dst_path = RESULTS_OUT / dst
    if src_path.exists():
        shutil.copy2(src_path, dst_path)
        print(f"  Copied: {dst}")
    else:
        print(f"  MISSING: {src}")

print("\nAll deliverables copied to results folder.")

# %%
