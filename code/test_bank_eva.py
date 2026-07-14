"""
Évaluation NAS sur Credit Card Fraud (dataset déséquilibré ~0.2% de positifs).
Version "VRAM Optimisée" : 100% des données (34 MB) résident sur le GPU. 
Zéro DataLoader, Zéro goulot d'étranglement CPU-GPU.
"""

import os
import time
import json
import random
import gc

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             average_precision_score, classification_report)
import types

from layer_classes import LinearCfg
from optimizer import ABCOptimizer, TransformerOptimizer
from model import DynamicNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------- configuration ------------------------------
SEEDS = [42, 43, 44]
TRANSFORMER_ITERS = 10     # 2 iters x batch 16 = 32 archis
ABC_ITERS = 20
ABC_POP_SIZE = 20
ABC_LIMIT = 5
ABC_PATIENCE = 5
PROXY_EPOCHS = 10
FINAL_EPOCHS = 100
MAX_LAYERS = 50
# -------------------------------------------------------------------------

def set_global_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Optimisations Ampere (RTX 30XX)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

# ---------------------------- chargement données -------------------------
print(f"Utilisation du device : {DEVICE}")
print("Chargement et transfert des données en VRAM...")

df = pd.read_csv("data/creditcard.csv")
y_all = df['Class'].values
X_all = df.drop(columns=['Class', 'Time']).values
X_scaled = StandardScaler().fit_transform(X_all)

# Split fixe
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y_all, test_size=0.2, random_state=42, stratify=y_all)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42,
    stratify=y_train_full)

print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

# ---> PRÉCHARGEMENT MASSIF SUR LE GPU <---
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(DEVICE)

X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(DEVICE)

X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(DEVICE)

POS_WEIGHT = torch.tensor(
    [(y_train == 0).sum() / max(1, (y_train == 1).sum())],
    dtype=torch.float32).to(DEVICE)

# Dummy dataset pour initialiser l'Optimiseur de base
dummy_dataset = [(torch.zeros(29), torch.zeros(1))]

# ---------------------------- proxy d'évaluation -------------------------
def evaluate_fraud_proxy(self, genome, train_epochs=PROXY_EPOCHS):
    self.eval_count += 1
    try:
        model = DynamicNet(genome, input_shape=(29,)).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Batch Size plus grand autorisé car les données tabulaires sont très légères
        bs = 1024 
        n = len(self.X_train)

        for _ in range(train_epochs):
            model.train()
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, bs):
                idx = perm[i:i+bs]
                optimizer.zero_grad()
                loss = criterion(model(self.X_train[idx]), self.y_train[idx])
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            # Inférence instantanée sur TOUT le set de validation d'un coup (56 000 lignes)
            probs = torch.sigmoid(model(self.X_val)).cpu().numpy().ravel()
            targets = self.y_val.cpu().numpy().ravel()

        if sum(targets) == 0:
            return 0.0
        return float(average_precision_score(targets, probs)) * 100.0
    except Exception:
        return -float('inf')

# ---------------------------- boucle expérimentale -----------------------
os.makedirs("results", exist_ok=True)
all_metrics = []
all_results = {}

for current_seed in SEEDS:
    print(f"\n{'='*50}\nFRAUD - SEED: {current_seed}\n{'='*50}")
    set_global_seed(current_seed)

    initial_arch = [
        LinearCfg(in_features=0, out_features=32, activation=nn.ReLU),
        LinearCfg(in_features=0, out_features=1,  activation=None),
    ]

    # --- Phase 1 : Transformer ------------------------------------------
    print("Phase Transformer (recherche macro)...")
    start_t = time.time()
    opt_trans = TransformerOptimizer(layers=initial_arch, max_layers=MAX_LAYERS, dataset=dummy_dataset, entropy_fct="default")
    # Injection des données VRAM
    opt_trans.X_train, opt_trans.y_train = X_train_t, y_train_t
    opt_trans.X_val, opt_trans.y_val = X_val_t, y_val_t
    opt_trans.evaluate = types.MethodType(evaluate_fraud_proxy, opt_trans)
    
    best_arch_trans, _ = opt_trans.run(TRANSFORMER_ITERS)
    evals_trans = opt_trans.eval_count
    time_trans = time.time() - start_t

    # --- Phase 2 : ABC warm-started --------------------------------------
    print("Phase ABC (exploitation locale)...")
    start_abc = time.time()
    opt_abc = ABCOptimizer(layers=best_arch_trans, dataset=dummy_dataset, pop_size=ABC_POP_SIZE, limit=ABC_LIMIT, patience=ABC_PATIENCE)
    # Injection des données VRAM
    opt_abc.X_train, opt_abc.y_train = X_train_t, y_train_t
    opt_abc.X_val, opt_abc.y_val = X_val_t, y_val_t
    opt_abc.evaluate = types.MethodType(evaluate_fraud_proxy, opt_abc)
    
    best_sol_final, _ = opt_abc.run(ABC_ITERS)
    evals_abc = opt_abc.eval_count
    time_abc = time.time() - start_abc

    total_evals = evals_trans + evals_abc
    total_search_time = time_trans + time_abc
    print(f"Bilan recherche : {total_evals} évaluations proxy en {total_search_time/60:.1f} min")

    # --- Nettoyage mémoire pré-entraînement ---
    del opt_trans, opt_abc
    gc.collect()

    # --- Phase 3 : Entraînement final ------------------------------------
    print(f"Entraînement final ({FINAL_EPOCHS} epochs)...")
    start_train = time.time()
    final_model = DynamicNet(best_sol_final, input_shape=(29,)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)
    optimizer = optim.Adam(final_model.parameters(), lr=0.001)

    bs_final = 512
    n_train = len(X_train_t)

    for _ in range(FINAL_EPOCHS):
        final_model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        for i in range(0, n_train, bs_final):
            idx = perm[i:i+bs_final]
            optimizer.zero_grad()
            loss = criterion(final_model(X_train_t[idx]), y_train_t[idx])
            loss.backward()
            optimizer.step()
            
    time_train = time.time() - start_train
    n_params = sum(p.numel() for p in final_model.parameters())

    # --- Phase 4 : Choix du seuil sur la VALIDATION ----------------------
    final_model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(final_model(X_val_t)).cpu().numpy().ravel()
        val_targets = y_val_t.cpu().numpy().ravel()

    best_f1, best_threshold = -1, 0.5
    for thresh in np.arange(0.05, 1.0, 0.01):
        f1 = f1_score(val_targets, (val_probs > thresh).astype(float), zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(thresh)
            
    print(f"Seuil optimal sur validation : {best_threshold:.2f} (F1 val = {best_f1:.4f})")

    # --- Phase 5 : Évaluation finale sur le TEST -------------------------
    with torch.no_grad():
        test_probs = torch.sigmoid(final_model(X_test_t)).cpu().numpy().ravel()
        test_targets = y_test_t.cpu().numpy().ravel()
        
    test_preds = (test_probs > best_threshold).astype(float)

    test_f1 = f1_score(test_targets, test_preds, zero_division=0)
    test_precision = precision_score(test_targets, test_preds, zero_division=0)
    test_recall = recall_score(test_targets, test_preds, zero_division=0)
    test_auprc = average_precision_score(test_targets, test_probs)
    
    report_dict = classification_report(test_targets, test_preds, target_names=["Normal", "Fraud"], output_dict=True, zero_division=0)

    print(f"--> Seed {current_seed} | F1={test_f1:.4f} | Precision={test_precision:.4f} | Recall={test_recall:.4f} | AUPRC={test_auprc:.4f} | {n_params:,} paramètres")
    print(classification_report(test_targets, test_preds, target_names=["Normal", "Fraud"], zero_division=0))

    all_metrics.append((test_f1, test_precision, test_recall, test_auprc, n_params, total_evals, total_search_time))

    # Sauvegardes
    torch.save(final_model.state_dict(), f"results/best_model_fraud_seed_{current_seed}.pth")
    with open(f"results/academic_results_fraud_{current_seed}.json", "w", encoding="utf-8") as f:
        json.dump({
            "seed": current_seed,
            "search_times": {
                "transformer_time_s": time_trans,
                "abc_time_s": time_abc,
                "total_search_time_s": total_search_time,
            },
            "final_training_time_s": time_train,
            "evaluations": {
                "transformer_evals": evals_trans,
                "abc_evals": evals_abc,
                "total_evals": total_evals,
            },
            "best_threshold": best_threshold,
            "n_parameters": n_params,
            "test_metrics": {
                "f1_score": float(test_f1),
                "precision": float(test_precision),
                "recall": float(test_recall),
                "auprc": float(test_auprc),
            },
            "report": report_dict,
        }, f, indent=4)

    # --- Nettoyage VRAM fin de seed ---
    del final_model
    gc.collect()
    torch.cuda.empty_cache()

# ---------------------------- Bilan multi-seed ---------------------------
f1s, precs, recs, auprcs, params, evals, times = map(np.array, zip(*all_metrics))

print(f"\n{'='*50}\nBILAN GLOBAL FRAUD ({len(SEEDS)} seeds)\n{'='*50}")
print(f"F1-Score  : {f1s.mean():.4f} ± {f1s.std(ddof=1):.4f}")
print(f"Precision : {precs.mean():.4f} ± {precs.std(ddof=1):.4f}")
print(f"Recall    : {recs.mean():.4f} ± {recs.std(ddof=1):.4f}")
print(f"AUPRC     : {auprcs.mean():.4f} ± {auprcs.std(ddof=1):.4f}")
print(f"Params    : {params.mean():.0f} ± {params.std(ddof=1):.0f}")
print(f"Évaluations: {evals.mean():.0f} ± {evals.std(ddof=1):.0f}")
print(f"Temps NAS : {times.mean()/60:.1f} ± {times.std(ddof=1)/60:.1f} min")

with open("results/academic_results_fraud_summary.json", "w", encoding="utf-8") as f:
    json.dump({
        "seeds": SEEDS,
        "f1_mean": float(f1s.mean()),     "f1_std":    float(f1s.std(ddof=1)),
        "precision_mean": float(precs.mean()), "precision_std":  float(precs.std(ddof=1)),
        "recall_mean": float(recs.mean()),    "recall_std":  float(recs.std(ddof=1)),
        "auprc_mean": float(auprcs.mean()),   "auprc_std":  float(auprcs.std(ddof=1)),
        "n_params_mean": float(params.mean()),"n_params_std":  float(params.std(ddof=1)),
        "total_evals_mean": float(evals.mean()), "search_time_s_mean": float(times.mean()),
    }, f, indent=4)

print("\n--> Résultats sauvegardés dans results/academic_results_fraud_*.json")