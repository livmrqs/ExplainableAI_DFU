import pandas as pd
import numpy as np
import os
import cv2
import time
import json
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from xgboost import XGBClassifier
from codecarbon import EmissionsTracker


# ============================================================
# CONFIGURATION
# ============================================================

base_path = "../data/ischaemia"
image_size = (256, 256)
task_name = "ischaemia"

n_components_pca = 50
num_samples_to_explain = 10

random_state = 42


# ============================================================
# DATASET LOADING
# ============================================================

dataset = []

for classe, label in zip(["Aug-Positive", "Aug-Negative"], [1, 0]):
    pasta = os.path.join(base_path, classe)

    for imagem in os.listdir(pasta):
        caminho_imagem = os.path.join(pasta, imagem)
        dataset.append((caminho_imagem, label))

df = pd.DataFrame(dataset, columns=["imagem", "label"])


def load_images(df, image_size):
    """
    Loads images, resizes them, normalizes pixel values to [0, 1],
    and flattens each image into a 1D vector.

    Output:
        X: shape (n_images, height * width * channels)
        y: shape (n_images,)
    """
    imagens, labels = [], []

    for _, row in df.iterrows():
        img = cv2.imread(row["imagem"])

        if img is not None:
            img = cv2.resize(img, image_size)

            # Convert BGR to RGB so matplotlib displays correctly
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = img.astype("float32") / 255.0
            imagens.append(img.flatten())
            labels.append(row["label"])
        else:
            print(f"Imagem não carregada: {row['imagem']}")

    return np.array(imagens), np.array(labels)


X_all, y_all = load_images(df, image_size)
paths_all = df["imagem"].values


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
    X_all,
    y_all,
    paths_all,
    test_size=0.2,
    random_state=random_state,
    stratify=y_all
)


# ============================================================
# CROSS-VALIDATION
# ============================================================

kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

accs, precs, recs, f1s, aucs = [], [], [], [], []

print("\n[Cross-Validation - Training]")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    pca = PCA(n_components=n_components_pca)
    X_tr_pca = pca.fit_transform(X_tr)
    X_val_pca = pca.transform(X_val)

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1
    )

    clf.fit(X_tr_pca, y_tr)

    y_pred = clf.predict(X_val_pca)
    y_proba = clf.predict_proba(X_val_pca)[:, 1]

    accs.append(accuracy_score(y_val, y_pred))
    precs.append(precision_score(y_val, y_pred))
    recs.append(recall_score(y_val, y_pred))
    f1s.append(f1_score(y_val, y_pred))
    aucs.append(roc_auc_score(y_val, y_proba))


print("\n[Average Metrics - Cross-Validation]")
print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
print(f"Recall:   {np.mean(recs):.4f} ± {np.std(recs):.4f}")
print(f"F1-Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")


# ============================================================
# FINAL TRAINING AND TEST EVALUATION
# ============================================================

print("\n[Final Training and Test Evaluation]")

tracker = EmissionsTracker(log_level="ERROR")
tracker.start()

pca_final = PCA(n_components=n_components_pca)
X_train_pca = pca_final.fit_transform(X_train)
X_test_pca = pca_final.transform(X_test)

clf_final = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    eval_metric="logloss",
    tree_method="hist",
    random_state=random_state,
    n_jobs=-1
)

clf_final.fit(X_train_pca, y_train)

emissions = tracker.stop()

emissions_dir = "../reports/emissions"
os.makedirs(emissions_dir, exist_ok=True)

emissions_path = os.path.join(emissions_dir, f"{task_name}_xgb_pca.json")

with open(emissions_path, "w") as f:
    json.dump({"emissions_kgCO2eq": emissions}, f)

print("\n[Carbon Footprint]")
print(f"Estimated emissions during training: {emissions:.6f} kg CO₂eq")


# ============================================================
# TEST SET PERFORMANCE
# ============================================================

y_pred_test = clf_final.predict(X_test_pca)
y_proba_test = clf_final.predict_proba(X_test_pca)[:, 1]

print("\n[Test Set Performance]")
print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test):.4f}")
print(f"Recall:   {recall_score(y_test, y_pred_test):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_test):.4f}")
print(f"AUC:      {roc_auc_score(y_test, y_proba_test):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))


# ============================================================
# SAVE FINAL MODEL
# ============================================================

output_dir = f"../models/{task_name}"
os.makedirs(output_dir, exist_ok=True)

pca_path = os.path.join(output_dir, f"xgboost_PCA_{task_name}.pkl")
clf_path = os.path.join(output_dir, f"xgboost_model_PCA_{task_name}.pkl")

joblib.dump(pca_final, pca_path)
joblib.dump(clf_final, clf_path)

print(f"\nSaved PCA to: {pca_path}")
print(f"Saved XGBoost model to: {clf_path}")


# ============================================================
# AUXILIARY FUNCTIONS
# ============================================================

feature_names = [f"PC{i+1}" for i in range(X_train_pca.shape[1])]


def normalize_heatmap(heatmap):
    """
    Normalizes a heatmap to [0, 1].
    """
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    return heatmap


def get_confusion_group(true_label, pred_label):
    """
    Returns TP, TN, FP, or FN.
    """
    if true_label == 1 and pred_label == 1:
        return "TP"
    if true_label == 0 and pred_label == 0:
        return "TN"
    if true_label == 0 and pred_label == 1:
        return "FP"
    if true_label == 1 and pred_label == 0:
        return "FN"


def get_image_name(idx):
    """
    Returns a clean image filename.
    """
    return os.path.basename(paths_test[idx]).split(".")[0]


def get_original_image_from_X_test(idx):
    """
    Reshapes the flattened X_test vector back to image format.
    """
    img = X_test[idx].reshape(image_size[0], image_size[1], 3)
    img = np.clip(img, 0, 1)
    return img


def save_xai_figure(original_img, heatmap, save_path, title):
    """
    Saves an explanation figure in the same visual style as CAM outputs:
        Original | Heatmap | Overlay
    """
    heatmap = normalize_heatmap(heatmap)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="jet")
    plt.title("Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(original_img)
    plt.imshow(heatmap, cmap="jet", alpha=0.45)
    plt.title("Overlay")
    plt.axis("off")

    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 2. SHAP RECONSTRUCTED INTO IMAGE SPACE
# ============================================================
#
# What this script does:
#   - Trains PCA + XGBoost using the same pipeline as your original code.
#   - Computes SHAP values for PC1...PC50.
#   - Projects the SHAP values back to the original pixel space using:
#         shap_pixel_importance = shap_pc_values @ pca_final.components_
#   - Saves figures in CAM-like style:
#         Original | Heatmap | Overlay
#
# Expected output:
#   ../reports/xai/ischaemia_xgb_pca/2_shap_reconstructed_image/
#       shap_reconstructed_0_TP_...png
#       shap_reconstructed_1_TN_...png
#       ...
#
# Interpretation:
#   - Red/yellow regions are associated with PCA directions that most affected
#     the XGBoost output.
#   - This is not Grad-CAM. It is a PCA-backprojected SHAP map.

import shap

xai_dir = f"../reports/xai/{task_name}_xgb_pca"
shap_recon_dir = os.path.join(xai_dir, "2_shap_reconstructed_image")
os.makedirs(shap_recon_dir, exist_ok=True)

print("\n[2] Running SHAP reconstructed into image space...")

shap_explainer = shap.TreeExplainer(clf_final)
shap_values = shap_explainer.shap_values(X_test_pca)

if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]
else:
    shap_values_class1 = shap_values


def shap_reconstructed_heatmap(idx):
    """
    Converts SHAP values from PCA space into a 2D image heatmap.
    """
    shap_pc_values = shap_values_class1[idx]

    pixel_importance_flat = np.dot(shap_pc_values, pca_final.components_)

    pixel_importance_img = pixel_importance_flat.reshape(
        image_size[0],
        image_size[1],
        3
    )

    heatmap = np.mean(np.abs(pixel_importance_img), axis=2)

    return normalize_heatmap(heatmap)


for idx in range(min(num_samples_to_explain, len(X_test_pca))):
    original_img = get_original_image_from_X_test(idx)

    true_label = int(y_test[idx])
    pred_label = int(y_pred_test[idx])
    prob_pos = float(y_proba_test[idx])
    prob_neg = 1.0 - prob_pos
    group = get_confusion_group(true_label, pred_label)
    image_name = get_image_name(idx)

    heatmap = shap_reconstructed_heatmap(idx)

    title = (
        f"Method=SHAP reconstructed via PCA | Group={group} | "
        f"True={true_label} | Pred={pred_label} | "
        f"P1={prob_pos:.3f} | P0={prob_neg:.3f}"
    )

    save_path = os.path.join(
        shap_recon_dir,
        f"shap_reconstructed_{idx}_{group}_{image_name}_true{true_label}_pred{pred_label}.png"
    )

    save_xai_figure(
        original_img=original_img,
        heatmap=heatmap,
        save_path=save_path,
        title=title
    )

print(f"\n[Done] SHAP reconstructed image outputs saved in: {shap_recon_dir}")
