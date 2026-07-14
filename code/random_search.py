"""
Baseline Random Search pour nas-torch — version "proxy en VRAM".

Cette version contourne la lenteur des DataLoader Windows en préchargeant
les images du proxy directement sur le GPU. Sur RTX 3060, le gain est
typiquement x3 à x5 par rapport à num_workers=0.

Choix méthodologique important :
  - Le proxy utilise un dataset SANS augmentation (transform_eval). C'est
    le standard NAS : l'augmentation servirait à généraliser sur 100
    époques, pas à comparer ~940 architectures sur 5 époques. Si on
    préchargeait le dataset augmenté, on figerait UNE seule version
    augmentée pour tous les runs, ce qui annulerait l'effet régularisant
    de l'augmentation tout en bruitant la validation.
  - L'entraînement final 100 époques garde l'augmentation complète, comme
    dans Cifar_hybrid.py.

Si tu veux un alignement strict avec Cifar_hybrid.py (qui charge val avec
augmentation, par accident), c'est faisable mais déconseillé : il vaut
mieux corriger aussi Cifar_hybrid.py pour utiliser transform_eval sur la
val.

Protocole reste identique :
  - seeds [42, 43, 44], set_global_seed
  - split sans fuite : 40% train proxy / 10% val proxy
  - proxy 5 époques, patience 2, batch 512
  - eval_count incrémenté à chaque évaluation
  - entraînement final 100 époques + CosineAnnealingLR
  - sorties results/academic_results_RS_{seed}.json

Budget : nombre d'évaluations proxy. Lu automatiquement depuis le JSON
de l'hybride pour égalité parfaite, fallback N_EVALUATIONS_DEFAULT.
"""

import os
import time
import json
import types
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from layer_classes import Conv2dCfg, MaxPool2dCfg, FlattenCfg, LinearCfg
from optimizer import TransformerOptimizer
from model import DynamicNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------- configuration -----------------------------
SEEDS = [42, 43, 44]
PROXY_EPOCHS = 10
FINAL_EPOCHS = 100
N_EVALUATIONS_DEFAULT = 940
READ_BUDGET_FROM_HYBRID = True
HYBRID_JSON_PATTERN = "results/academic_results_hybride_{seed}.json"
MAX_LAYERS = 20

# Accélérations Ampere (RTX 3060). Désactive le déterminisme bit-exact :
# tes 3 seeds restent reproductibles à l'écart-type près, ce qui suffit
# pour le papier. Si tu veux le déterminisme bit-exact, mets à False.
USE_TF32 = True
# -------------------------------------------------------------------------


def set_global_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if USE_TF32:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if USE_TF32:
    # TF32 sur les matmul : ~30% de débit gratuit sur Ampere.
    torch.set_float32_matmul_precision('high')


def to_gpu_tensors(dataset, indices):
    """Précharge un sous-ensemble du dataset directement en VRAM.
    Le transform du dataset est appliqué UNE seule fois ici, donc le
    dataset passé doit être SANS augmentation."""
    xs, ys = [], []
    for i in indices:
        x, y = dataset[int(i)]
        xs.append(x)
        ys.append(y)
    X = torch.stack(xs).to(DEVICE)
    y = torch.tensor(ys, dtype=torch.long).to(DEVICE)
    return X, y


def evaluate_cifar_proxy(self, genome, train_epochs=PROXY_EPOCHS):
    """Proxy 100% sur GPU, sans DataLoader : itération par indices."""
    self.eval_count += 1
    try:
        model = DynamicNet(genome, input_shape=(3, 32, 32)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        bs = 512
        n = len(self.X_train)
        best_val_acc, patience, patience_counter = 0.0, 2, 0

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
                correct = 0
                for i in range(0, len(self.X_val), bs):
                    out = model(self.X_val[i:i+bs])
                    _, p = out.max(1)
                    correct += p.eq(self.y_val[i:i+bs]).sum().item()
            current_acc = 100. * correct / len(self.X_val)

            if current_acc > best_val_acc:
                best_val_acc, patience_counter = current_acc, 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break

        return best_val_acc
    except Exception:
        return -float('inf')


class RandomSearchOptimizer(TransformerOptimizer):
    """Échantillonnage uniforme sur le même espace de tokens que le
    contrôleur Transformer. Le petit Transformer construit par le
    constructeur parent est simplement ignoré."""

    def generate_architecture(self):
        generated_cfg = []
        is_linear_context = False
        for _ in range(self.max_layers):
            token_str = random.choice(self.vocab)
            if token_str == "stop":
                break
            if (token_str == "flatten" or token_str == "avgpool"
                    or token_str.startswith("linear")):
                is_linear_context = True
            cfg = self._token_to_cfg(token_str, is_linear_context)
            if cfg is not None:
                generated_cfg.append(cfg)
        if not any(isinstance(l, LinearCfg) for l in generated_cfg):
            generated_cfg.append(FlattenCfg())
        generated_cfg.append(LinearCfg(in_features=0,
                                       out_features=self.out_features,
                                       activation=None))
        return generated_cfg, None, None

    def run(self, n_evaluations=None, time_budget_s=None, verbose=True):
        assert n_evaluations or time_budget_s
        t0 = time.time()
        history = []
        n_invalid = 0

        while True:
            if n_evaluations is not None and self.eval_count >= n_evaluations:
                break
            if time_budget_s is not None and time.time() - t0 >= time_budget_s:
                break

            arch_cfg, _, _ = self.generate_architecture()
            raw_score = self.evaluate(arch_cfg)

            if raw_score == -float('inf'):
                n_invalid += 1
            elif raw_score > self.best_score:
                self.best_score = raw_score
                self.best_arch = copy.deepcopy(arch_cfg)
                if verbose:
                    print(f"RS eval {self.eval_count}: nouveau best "
                          f"{self.best_score:.2f} "
                          f"(profondeur {len(arch_cfg)}, "
                          f"{(time.time() - t0) / 60:.1f} min)")

            history.append((self.eval_count, raw_score, len(arch_cfg),
                            self.best_score, round(time.time() - t0, 1)))

        return self.best_arch, {
            "best_proxy_score": self.best_score,
            "n_evaluations": self.eval_count,
            "n_invalid": n_invalid,
            "search_time_s": time.time() - t0,
            "history": history,
        }


def resolve_budget(seed):
    if READ_BUDGET_FROM_HYBRID:
        path = HYBRID_JSON_PATTERN.format(seed=seed)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    n = int(json.load(f)["evals"])
                print(f"[seed {seed}] budget lu depuis {path} : "
                      f"{n} évaluations")
                return n
            except (KeyError, ValueError, json.JSONDecodeError):
                print(f"[seed {seed}] {path} illisible, fallback "
                      f"{N_EVALUATIONS_DEFAULT}")
    return N_EVALUATIONS_DEFAULT


if __name__ == "__main__":
    print(DEVICE)
    os.makedirs("results", exist_ok=True)

    # transform_eval : sans augmentation, utilisé pour le proxy et le test.
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    # transform_train : avec augmentation, utilisé UNIQUEMENT pour
    # l'entraînement final 100 époques (où l'augmentation est rafraîchie
    # à chaque accès via DataLoader).
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # Deux instances du même dataset : une pour préchargement (eval),
    # une pour l'entraînement final (train avec augmentation).
    trainset_eval  = torchvision.datasets.CIFAR10(
        root='./data', train=True,  download=True, transform=transform_eval)
    trainset_aug   = torchvision.datasets.CIFAR10(
        root='./data', train=True,  download=True, transform=transform_train)
    testset        = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_eval)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False,
                             num_workers=0, pin_memory=True)

    all_accuracies = []

    for current_seed in SEEDS:
        print(f"\n{'='*50}\nRANDOM SEARCH - SEED: {current_seed}\n{'='*50}")
        set_global_seed(current_seed)

        # Split identique aux autres scripts CIFAR
        indices = np.random.permutation(len(trainset_eval))
        train_proxy_idx = indices[:int(0.4 * len(trainset_eval))]
        val_proxy_idx   = indices[int(0.4 * len(trainset_eval)):
                                  int(0.5 * len(trainset_eval))]

        # Préchargement GPU des deux sous-ensembles (~245 MB + ~60 MB)
        print("Préchargement du proxy sur GPU...")
        t_preload = time.time()
        X_train_proxy, y_train_proxy = to_gpu_tensors(trainset_eval,
                                                     train_proxy_idx)
        X_val_proxy,   y_val_proxy   = to_gpu_tensors(trainset_eval,
                                                     val_proxy_idx)
        print(f"  fait en {time.time() - t_preload:.1f}s | "
              f"train {tuple(X_train_proxy.shape)} | "
              f"val {tuple(X_val_proxy.shape)}")

        # DataLoader pour l'entraînement final uniquement (augmentation
        # rafraîchie à chaque epoch via __getitem__).
        train_loader_final = DataLoader(trainset_aug, batch_size=512,
                                        shuffle=True, num_workers=0,
                                        pin_memory=True)

        # Architecture initiale (sert seulement au constructeur de
        # l'optimiseur pour inférer out_features).
        initial_arch = [
            Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3,
                      padding=1, activation=nn.ReLU),
            MaxPool2dCfg(kernel_size=2, stride=2, padding=0),
            FlattenCfg(),
            LinearCfg(in_features=0, out_features=10, activation=None)
        ]

        # On lui passe un DataLoader minimal juste pour que
        # Optimizer.__init__ puisse inférer out_features (classes CIFAR-10).
        # Ce loader n'est jamais utilisé pour itérer pendant la recherche.
        dummy_loader = DataLoader(trainset_eval, batch_size=512)

        print("Début Random Search...")
        budget = resolve_budget(current_seed)
        start_rs = time.time()
        opt_rs = RandomSearchOptimizer(layers=initial_arch,
                                       dataset=dummy_loader,
                                       max_layers=MAX_LAYERS)
        opt_rs.X_train, opt_rs.y_train = X_train_proxy, y_train_proxy
        opt_rs.X_val,   opt_rs.y_val   = X_val_proxy,   y_val_proxy
        opt_rs.evaluate = types.MethodType(evaluate_cifar_proxy, opt_rs)

        best_sol_final, rs_stats = opt_rs.run(n_evaluations=budget)
        total_search_time = time.time() - start_rs
        print(f"Bilan recherche : {rs_stats['n_evaluations']} évaluations "
              f"({rs_stats['n_invalid']} invalides) en "
              f"{total_search_time/3600:.2f} h | "
              f"best proxy = {rs_stats['best_proxy_score']:.2f}")

        # Libération de la VRAM avant l'entraînement final
        del X_train_proxy, y_train_proxy, X_val_proxy, y_val_proxy
        torch.cuda.empty_cache()

        print("Entraînement final (100 Epochs)...")
        final_model = DynamicNet(best_sol_final,
                                 input_shape=(3, 32, 32)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(final_model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=FINAL_EPOCHS)

        for epoch in range(FINAL_EPOCHS):
            final_model.train()
            for inputs, targets in train_loader_final:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()
                loss = criterion(final_model(inputs), targets)
                loss.backward()
                optimizer.step()
            scheduler.step()

        final_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = final_model(inputs.to(DEVICE))
                _, preds = outputs.max(1)
                total += targets.size(0)
                correct += preds.eq(targets.to(DEVICE)).sum().item()

        final_accuracy = correct / total
        n_params = final_model.count_parameters()
        print(f"--> Accuracy finale Seed {current_seed} : "
              f"{final_accuracy*100:.2f}% | {n_params:,} paramètres")
        all_accuracies.append(final_accuracy)

        torch.save(final_model.state_dict(),
                   f"results/best_model_RS_seed_{current_seed}.pth")

        with open(f"results/academic_results_RS_{current_seed}.json", "w",
                  encoding="utf-8") as f:
            json.dump({
                "accuracy": final_accuracy,
                "evals": rs_stats["n_evaluations"],
                "n_invalid": rs_stats["n_invalid"],
                "time": total_search_time,
                "n_parameters": n_params,
                "best_proxy_score": rs_stats["best_proxy_score"],
                "history": rs_stats["history"],
            }, f)

    print(f"\nMoyenne: {np.mean(all_accuracies)*100:.2f}% "
          f"± {np.std(all_accuracies, ddof=1)*100:.2f}%")