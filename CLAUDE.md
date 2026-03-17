# CLAUDE.md — PINKCC Challenge 2026 Agent

## Identité du projet

Ce projet est notre soumission au **PINKCC Challenge 2026**, un data challenge international de segmentation et détection automatique du cancer du pancréas à partir d'imagerie médicale (CT et IRM). Organisé par le SIRIC Montpellier Cancer / Laboratoire PINKCC (ICM/IRCM).

- **Site** : https://pinkcc-challenge.com
- **Discord** : canal Q&A officiel (toute communication passe par Discord)
- **Période Phase 1** : 9 mars → 27 avril 2026 (7 semaines)
- **Période Phase 2** : 4 mai → 22 mai 2026 (3 semaines, sur invitation)
- **Finale** : 27 mai 2026, Montpellier

---

## Les deux tâches obligatoires

### Tâche 1 — Classification au niveau examen
- Prédire la probabilité qu'un examen (CT ou IRM) corresponde à une tumeur pancréatique maligne primitive.
- Sortie : `classification.csv` avec colonnes `Exam_id, probability`
- **Métrique : AUPRC** (Area Under Precision-Recall Curve), calculée séparément CT et IRM.

### Tâche 2 — Segmentation tumorale 3D
- Produire un masque binaire 3D (0=background, 1=tumeur) par examen.
- Dimensions du masque = dimensions exactes de l'image originale.
- Pour les cas négatifs : masque vide obligatoire.
- Sortie : `segmentations.zip` contenant des fichiers `.nii.gz` convertis en `.npz.gz` via le script fourni.
- **Métriques : Dice coefficient + F2-score**, calculées séparément CT et IRM.

### Formule de scoring

```
CT_score   = mean(CT_classification_AUPRC, CT_segmentation)
MRI_score  = mean(MRI_classification_AUPRC, MRI_segmentation)
Final_score = (2 × CT_score + MRI_score) / 3
```

Le CT pèse **deux fois plus** que l'IRM. En cas d'égalité : score de Brier + timestamp.

---

## Données

### Phase 1
- **Train_1** : images + labels (label binaire + masque segmentation pour cas positifs)
- **Test_1** : images sans labels
- **12 264 CT** + **515 IRM** (données open source annotées par ICM)
- Format : **NIfTI** (.nii, .nii.gz)
- CT : intensités en **Unités Hounsfield (HU)**
- IRM : intensités **non standardisées** (séquences T1, phase artérielle)

### Phase 2
- Train_2 = Train_1 + Test_1 + labels Test_1 + données supplémentaires (ICM)
- Test_2 = nouveau jeu sans labels

### Contraintes critiques
- ⛔ **Datasets pancréatiques CT/IRM externes INTERDITS**
- ✅ **Modèles pré-entraînés open-source AUTORISÉS** (doivent être déclarés)
- ⚠️ Toute source de pré-entraînement doit être déclarée
- Le leaderboard public = **10% du test set** uniquement
- **5 soumissions max par jour** par équipe
- Évaluations : **lundi et vendredi à 12h UTC+1**

---

## Structure du projet

```
pinkcc-2026/
├── CLAUDE.md                    # Ce fichier
├── README.md
├── configs/                     # Configurations nnU-Net et modèles
│   ├── nnunet/
│   ├── segresnet/
│   └── swin_unetr/
├── data/
│   ├── raw/                     # Données brutes NIfTI (ne PAS commit)
│   │   ├── CT/
│   │   └── MRI/
│   ├── nnunet_raw/              # Format nnU-Net (Dataset001_PancreasCT, etc.)
│   ├── pancreas_rois/           # ROIs recadrés après étage 1
│   └── preprocessed/
├── pretrained/                  # Checkpoints pré-entraînés (ne PAS commit)
│   ├── suprem/
│   ├── r_super/
│   └── totalsegmentator/
├── src/
│   ├── preprocessing/
│   │   ├── convert_to_nnunet.py
│   │   ├── extract_pancreas_roi.py
│   │   ├── normalize_mri.py
│   │   └── run_totalsegmentator.py
│   ├── training/
│   │   ├── train_ct_segmentation.py
│   │   ├── train_mri_segmentation.py
│   │   ├── train_classification.py
│   │   └── multitask_head.py
│   ├── inference/
│   │   ├── predict.py
│   │   ├── ensemble.py
│   │   └── tta.py
│   ├── postprocessing/
│   │   ├── anatomical_filter.py
│   │   ├── connected_components.py
│   │   └── calibration.py
│   ├── evaluation/
│   │   ├── metrics.py           # AUPRC, Dice, F2
│   │   └── cross_validation.py
│   └── submission/
│       ├── format_submission.py
│       └── validate_submission.py
├── scripts/
│   ├── setup_environment.sh
│   ├── download_pretrained.sh
│   ├── run_pipeline.sh
│   └── batch_totalsegmentator.sh
├── notebooks/
│   ├── eda.ipynb
│   ├── visualize_predictions.ipynb
│   └── error_analysis.ipynb
├── results/
│   ├── cv_scores/
│   ├── predictions/
│   └── submissions/
├── requirements.txt
└── pyproject.toml
```

---

## Stack technique

### Environnement
- **Python 3.10+**, **PyTorch 2.x** avec CUDA
- **nnU-Net v2** (`pip install nnunetv2`) — framework principal
- **MONAI 1.3+** (`pip install monai[all]`) — Auto3DSeg, transforms, métriques
- **TotalSegmentator** (`pip install TotalSegmentator`) — localisation pancréas
- **nibabel**, **SimpleITK**, **pydicom** — manipulation NIfTI/DICOM
- **scikit-learn** — métriques, calibration
- GPU : Scaleway (fourni par les organisateurs)

### Modèles pré-entraînés à télécharger
```bash
# SuPreM (ICLR 2024) — initialisation principale
# Depuis HuggingFace : https://huggingface.co/MrGiovanni/SuPreM
# Poids disponibles : U-Net, SegResNet, Swin-UNETR (9 262 CTs, 25 classes)

# R-Super (MICCAI 2025) — meilleure init spécifique pancréas
# https://github.com/MrGiovanni/R-Super
# 5000 CTs lésionnels, 2200 paires CT-rapport pancréatiques

# TotalSegmentator — localisation pancréas (étage 1)
# Installé via pip, poids auto-téléchargés
TotalSegmentator -i input.nii.gz -o output --roi_subset pancreas --device gpu
# API Python :
# from totalsegmentator.python_api import totalsegmentator
```

---

## Pipeline en cascade (architecture gagnante)

### Étage 1 — Localisation du pancréas
```bash
# TotalSegmentator pour chaque volume (CT et IRM)
TotalSegmentator -i scan.nii.gz -o seg/ --roi_subset pancreas --device gpu
# Pour IRM : ajouter --task total_mr
TotalSegmentator -i mri.nii.gz -o seg/ --task total_mr --roi_subset pancreas --device gpu
```
Extraire bounding box du masque pancréas avec **20mm de marge**, recadrer le volume original.

### Étage 2 — Segmentation tumorale dans le ROI
- nnU-Net v2 3D_fullres initialisé avec poids SuPreM ou R-Super
- Entraîné sur les volumes recadrés autour du pancréas
- 5-fold cross-validation

### Classification
- Tête multi-tâche sur l'encodeur partagé (Global Average Pooling → MLP → sigmoid)
- Signaux dérivés de la segmentation : max proba, volume tumeur, nb composantes connexes
- Perte combinée : `L = 0.7 × L_seg + 0.3 × L_cls`

---

## Conventions de code

### Style
- Python : **Black** (88 chars), **isort**, **type hints** partout
- Docstrings : **Google style**
- Noms de fichiers : `snake_case.py`
- Noms de classes : `PascalCase`
- Variables de config : `SCREAMING_SNAKE_CASE`

### Git
- Branches : `feature/xxx`, `fix/xxx`, `exp/xxx` (expériences)
- Commits : préfixer par `feat:`, `fix:`, `data:`, `exp:`, `refactor:`, `docs:`
- **Ne JAMAIS commit** : données brutes, poids de modèles, prédictions volumineuses
- `.gitignore` doit exclure : `data/raw/`, `pretrained/`, `results/predictions/`, `*.nii.gz`, `*.npz.gz`

### Logging et reproductibilité
- Utiliser **Weights & Biases** ou **MLflow** pour tracker toutes les expériences
- Chaque run doit logger : config complète, seed, métriques par fold, temps d'entraînement
- Seed fixe : `42` par défaut pour la reproductibilité
- Sauvegarder les configs nnU-Net dans `configs/`

---

## Commandes nnU-Net essentielles

```bash
# Variables d'environnement obligatoires
export nnUNet_raw="/path/to/data/nnunet_raw"
export nnUNet_preprocessed="/path/to/data/preprocessed"
export nnUNet_results="/path/to/results"

# Conversion des données au format nnU-Net
python src/preprocessing/convert_to_nnunet.py \
    --input data/raw/CT \
    --output $nnUNet_raw/Dataset001_PancreasCT \
    --modality CT

# Planification et prétraitement
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity

# Entraînement (5-folds)
nnUNetv2_train 001 3d_fullres FOLD -p nnUNetPlans
# FOLD = 0, 1, 2, 3, 4

# Entraînement avec pré-entraînement SuPreM
nnUNetv2_train 001 3d_fullres FOLD -p nnUNetPlans \
    -pretrained_weights pretrained/suprem/supervised_suprem_unet.pth

# Déterminer la meilleure configuration
nnUNetv2_find_best_configuration 001 -c 3d_fullres

# Inférence avec ensemble + TTA
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER \
    -d 001 -c 3d_fullres -f 0 1 2 3 4 \
    --save_probabilities  # IMPORTANT : garder les probas pour l'ensemble
```

---

## Métriques d'évaluation (à implémenter dans `src/evaluation/metrics.py`)

```python
import numpy as np
from sklearn.metrics import average_precision_score, fbeta_score

def compute_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUPRC pour la classification."""
    return average_precision_score(y_true, y_prob)

def compute_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice coefficient pour la segmentation."""
    intersection = np.sum(pred * target)
    return (2.0 * intersection) / (np.sum(pred) + np.sum(target) + 1e-8)

def compute_f2(pred: np.ndarray, target: np.ndarray) -> float:
    """F2-score (favorise le rappel)."""
    pred_flat = pred.flatten().astype(bool)
    target_flat = target.flatten().astype(bool)
    tp = np.sum(pred_flat & target_flat)
    fp = np.sum(pred_flat & ~target_flat)
    fn = np.sum(~pred_flat & target_flat)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    beta = 2
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)

def compute_final_score(
    ct_cls: float, ct_seg_dice: float, ct_seg_f2: float,
    mri_cls: float, mri_seg_dice: float, mri_seg_f2: float
) -> float:
    """Score final PINKCC."""
    ct_seg = (ct_seg_dice + ct_seg_f2) / 2
    mri_seg = (mri_seg_dice + mri_seg_f2) / 2
    ct_score = (ct_cls + ct_seg) / 2
    mri_score = (mri_cls + mri_seg) / 2
    return (2 * ct_score + mri_score) / 3
```

---

## Format de soumission

### 1. classification.csv (tab-separated en réalité → `.tsv`)
```
exam_id	probability
CT_000	0.87
CT_001	0.12
MRI_0001	0.95
MRI_0002	0.03
```

### 2. segmentations.zip
- Contient un masque `.nii.gz` par examen, converti en `.npz.gz` via le script fourni par les organisateurs.
- **Dimensions = exactement celles de l'image originale**
- Convention : 0 = background, 1 = tumeur
- **Masque vide obligatoire pour les cas négatifs**

### Validation avant soumission
```python
# Toujours vérifier avant de soumettre :
# 1. Toutes les images de test ont une prédiction
# 2. Dimensions masque == dimensions image originale
# 3. Valeurs masque ∈ {0, 1} uniquement
# 4. Cas négatifs → masque vide (tous zéros)
# 5. classification.csv contient TOUS les exam_id
# 6. Probabilités ∈ [0, 1]
```

---

## Stratégie d'entraînement

### Prétraitement CT
- Clip HU aux percentiles 0.5 et 99.5 des voxels foreground
- Z-score global (mean/std sur le training set)
- Plage HU pertinente pancréas : [-150, +250]

### Prétraitement IRM
1. Correction biais N4ITK (`SimpleITK.N4BiasFieldCorrection`)
2. Clip percentiles 0.5/99.5
3. Z-score par volume (foreground only)

### Fonctions de perte
- **Segmentation** : Dice + Cross-Entropy (default nnU-Net), ou Dice + Focal Loss (γ=2)
- **Classification** : Binary Focal Loss ou SOAP (LibAUC) pour optimiser directement l'AUPRC
- **Combinée** : `L = 0.7 × L_seg + 0.3 × L_cls`

### Augmentation
- Pipeline nnU-Net : rotation ±15°, scaling 0.85-1.15, elastic deformation, bruit gaussien, flou, gamma, mirroring
- **Foreground oversampling : 50-67%** (vs 33% par défaut nnU-Net)
- Optionnel : DiffTumor (tumeurs synthétiques) → `https://github.com/MrGiovanni/DiffTumor`

### Hyperparamètres
- **SGD** : lr=0.01, momentum=0.99, weight_decay=3e-5, polynomial decay, 1000 epochs
- **AdamW** (transformers) : lr=1e-4, warmup linéaire + cosine decay
- Batch size : 2 (patches 3D, taille auto-configurée par nnU-Net)
- Deep supervision activée

---

## Ensemble et inférence

### Stratégie d'ensemble
```python
# Soft voting : moyenne des probabilités softmax
# 5 folds × 2-3 architectures = 10-15 modèles
ensemble_prob = np.mean([model_probs for model_probs in all_predictions], axis=0)
final_mask = (ensemble_prob > threshold).astype(np.uint8)
```

### Test-Time Augmentation (TTA)
```python
# 8 combinaisons de flips (x, y, z)
flips = [(False,False,False), (True,False,False), (False,True,False), ...]
predictions = []
for flip in flips:
    augmented = apply_flips(input_volume, flip)
    pred = model(augmented)
    pred = undo_flips(pred, flip)
    predictions.append(pred)
tta_result = np.mean(predictions, axis=0)
```

### Post-traitement
1. Seuillage (optimiser sur CV, typiquement 0.5)
2. Composantes connexes : retirer < 10-50 voxels
3. Contrainte anatomique : ≥15% overlap avec masque pancréas (TotalSegmentator)
4. Fermeture morphologique optionnelle

---

## Règles d'or pour l'agent

1. **Ne JAMAIS utiliser de datasets pancréatiques CT/IRM externes** — c'est éliminatoire.
2. **Déclarer tous les modèles pré-entraînés** utilisés dans un fichier `PRETRAINED_MODELS.md`.
3. **Faire confiance au CV 5-folds, PAS au leaderboard public** (10% du test seulement).
4. **Limiter les soumissions à ~3-4 par semaine** — ne pas overfitter le leaderboard.
5. **Le CT compte double** — toujours prioriser la performance CT.
6. **Tester le format de soumission tôt** — une soumission invalide = temps perdu.
7. **Vérifier les dimensions des masques** avant chaque soumission — doit correspondre exactement à l'image originale.
8. **Documenter chaque expérience** — les finalistes doivent être reproductibles.
9. **Garder le temps d'inférence raisonnable** — compatible avec usage clinique.
10. **Ne pas extraire, copier ou diffuser les données** hors de la plateforme.

---

## Liens utiles

| Ressource | URL |
|-----------|-----|
| PINKCC Challenge Platform | https://pinkcc-challenge.com |
| SuPreM (poids pré-entraînés) | https://github.com/MrGiovanni/SuPreM |
| R-Super (pancréas spécifique) | https://github.com/MrGiovanni/R-Super |
| DiffTumor (tumeurs synthétiques) | https://github.com/MrGiovanni/DiffTumor |
| PanTS dataset/benchmark | https://github.com/MrGiovanni/PanTS |
| nnU-Net v2 | https://github.com/MIC-DKFZ/nnUNet |
| MONAI | https://monai.io |
| TotalSegmentator | https://github.com/wasserth/TotalSegmentator |
| MONAI métriques | https://monai.readthedocs.io/en/stable/metrics.html |
| LibAUC (AUPRC optim) | https://github.com/Optimization-AI/LibAUC |
| HuggingFace SuPreM | https://huggingface.co/MrGiovanni/SuPreM |