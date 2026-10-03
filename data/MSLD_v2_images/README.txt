MSLD v2.0 Image Dataset — Download Instructions
================================================

Dataset:  Mpox Skin Lesion Dataset Version 2.0 (MSLD v2.0)
Authors:  Ali et al.
Cite:     Ali, S.N. et al. (2024). Multiclass skin lesion classification using
          ensemble of fine-tuned deep learning models. Biomed Signal Process Control.
License:  CC BY-NC 4.0 (non-commercial use only)
Images:   755 original images across 6 classes, 5 official folds

Why images are not included here
---------------------------------
The 3,775 image files (including augmented copies across folds) total ~500 MB,
which exceeds the practical size limit for a reproducibility data package.
Pre-extracted BiomedCLIP embeddings are provided instead:
  reproduce/data/stage3_biomedclip_embeddings_cache.npz  (8.5 MB)
This cache lets you skip the ~27-minute extraction step entirely.

Download instructions (no credentials required)
------------------------------------------------
Step 1: Install gdown
  pip install gdown>=4.7

Step 2: Download from Google Drive
  python -c "
  import gdown
  gdown.download_folder(
      id='1_bGmbDQNgJViQenjZ4QUhhzpdiubga48',
      output='/workspace/msld_v2',
      quiet=False
  )"

Step 3: Verify structure
  Expected path: /workspace/msld_v2/MSLD v2/original_images/FOLDS/
  Expected subfolders: fold1/ fold2/ fold3/ fold4/ fold5/
  Each fold contains: train/ valid/ test/
  Total images: 755 originals (3,775 including augmented copies)

Class prefix mapping
--------------------
  MKP_*     -> Mpox        (284 images)
  CHP_*     -> Chickenpox  ( 75 images)
  MSL_*     -> Measles     ( 55 images)
  CWP_*     -> Cowpox      ( 66 images)
  HFMD_*    -> HFMD        (161 images)
  HEALTHY_* -> Healthy     (114 images)

Alternative: use pre-extracted embeddings
-----------------------------------------
If you only need to reproduce the Stage 3 MLP training (not re-extract embeddings),
copy the cache file to the results directory and run phase1_imaging_model_fast.py:

  cp reproduce/data/stage3_biomedclip_embeddings_cache.npz \
     /workspace/measles_multimodal/results/stage3_all_embeddings_cache.npz

The pipeline script auto-detects the cache and skips extraction.

Contact
-------
For questions about this reproducibility package:
  Jiada Li, PhD — AI Scientist, Albany, NY, USA 12205 (jiadali2017@gmail.com)
