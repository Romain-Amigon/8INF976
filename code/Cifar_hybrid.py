import os
import time
import json
import types
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from layer_classes import Conv2dCfg, MaxPool2dCfg, FlattenCfg, LinearCfg
from optimizer import ABCOptimizer, TransformerOptimizer
from model import DynamicNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_global_seed(seed):
    import random
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

def to_gpu_tensors(dataset, indices):
    """Précharge un sous-ensemble du dataset directement en VRAM."""
    xs, ys = [], []
    for i in indices:
        x, y = dataset[int(i)]
        xs.append(x)
        ys.append(y)
    X = torch.stack(xs).to(DEVICE)
    y = torch.tensor(ys, dtype=torch.long).to(DEVICE)
    return X, y

def evaluate_cifar_proxy(self, genome, train_epochs=10):
    """Proxy 100% sur GPU, sans DataLoader : itération par indices."""
    self.eval_count += 1 
    try:
        model = DynamicNet(genome, input_shape=(3, 32, 32)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        bs = 512
        n = len(self.X_train)
        best_val_acc, patience, patience_counter = 0.0, 2, 0
        
        for epoch in range(train_epochs):
            model.train()
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, bs):
                idx = perm[i:i+bs]
                optimizer.zero_grad()
                loss = criterion(model(self.X_train[idx]), self.y_train[idx])
                loss.backward()
                optimizer.step()
                
            model.eval()
            correct = 0
            with torch.no_grad():
                for i in range(0, len(self.X_val), bs):
                    out = model(self.X_val[i:i+bs])
                    _, p = out.max(1)
                    correct += p.eq(self.y_val[i:i+bs]).sum().item()
            
            current_acc = 100. * correct / len(self.X_val)
            if current_acc > best_val_acc:
                best_val_acc = current_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience: break
                
        return best_val_acc
    except Exception as e:
        return -float('inf')

if __name__ == "__main__":
    print(f"Utilisation du device : {DEVICE}")
    os.makedirs("results", exist_ok=True)

    # transform_eval : SANS augmentation, utilisé pour le proxy et le test final.
    transform_eval = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # transform_train : AVEC augmentation, utilisé UNIQUEMENT pour l'entraînement final 100 époques.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset_eval = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_eval)
    trainset_aug  = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset       = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_eval)
    
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    seeds = [42, 43, 44]
    all_accuracies, all_results = [], {}

    for current_seed in seeds:
        print(f"\n{'='*50}\nRECHERCHE HYBRIDE - SEED: {current_seed}\n{'='*50}")
        set_global_seed(current_seed)
        
        indices = np.random.permutation(len(trainset_eval))
        train_proxy_idx = indices[:int(0.4 * len(trainset_eval))]
        val_proxy_idx   = indices[int(0.4 * len(trainset_eval)):int(0.5 * len(trainset_eval))]
        
        # Préchargement GPU
        print("Préchargement du proxy sur GPU...")
        t_preload = time.time()
        X_train_proxy, y_train_proxy = to_gpu_tensors(trainset_eval, train_proxy_idx)
        X_val_proxy,   y_val_proxy   = to_gpu_tensors(trainset_eval, val_proxy_idx)
        print(f"  Fait en {time.time() - t_preload:.1f}s")
        
        # DataLoader pour l'entraînement final 100 epochs
        train_loader_final = DataLoader(trainset_aug, batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
        # Dummy loader pour initialiser les optimiseurs
        dummy_loader = DataLoader(trainset_eval, batch_size=512)

        initial_arch = [
            Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=nn.ReLU),
            MaxPool2dCfg(kernel_size=2, stride=2, padding=0),
            FlattenCfg(),
            LinearCfg(in_features=0, out_features=10, activation=None)
        ]

        print("\nDébut Transformer...")
        start_t = time.time()
        opt_trans = TransformerOptimizer(layers=initial_arch, max_layers=20, dataset=dummy_loader, entropy_fct="default")
        opt_trans.X_train, opt_trans.y_train = X_train_proxy, y_train_proxy
        opt_trans.X_val,   opt_trans.y_val   = X_val_proxy,   y_val_proxy
        opt_trans.evaluate = types.MethodType(evaluate_cifar_proxy, opt_trans)
        
        best_arch_trans, _ = opt_trans.run(20)
        evals_trans = opt_trans.eval_count
        time_trans = time.time() - start_t

        print("\nDébut ABC...")
        start_abc = time.time()
        opt_abc = ABCOptimizer(layers=best_arch_trans, dataset=dummy_loader, pop_size=20, limit=5, patience=15)
        opt_abc.X_train, opt_abc.y_train = X_train_proxy, y_train_proxy
        opt_abc.X_val,   opt_abc.y_val   = X_val_proxy,   y_val_proxy
        opt_abc.evaluate = types.MethodType(evaluate_cifar_proxy, opt_abc)
        
        best_sol_final, _ = opt_abc.run(15)
        evals_abc = opt_abc.eval_count
        time_abc = time.time() - start_abc

        total_evals = evals_trans + evals_abc
        total_time = time_trans + time_abc
        print(f"Bilan recherche : {total_evals} évaluations proxy réalisées.")

        # =========================================================
        # NETTOYAGE VRAM AVANT L'ENTRAÎNEMENT FINAL
        # =========================================================
        print("Libération de la VRAM Proxy...")
        del X_train_proxy, y_train_proxy, X_val_proxy, y_val_proxy
        del opt_trans, opt_abc
        gc.collect()
        torch.cuda.empty_cache()

        print("\nEntraînement Final (100 epochs)...")
        final_model = DynamicNet(best_sol_final, input_shape=(3, 32, 32)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(final_model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        for epoch in range(100):
            final_model.train()
            for inputs, targets in train_loader_final:
                inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
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
        print(f"--> Accuracy Seed {current_seed} : {final_accuracy*100:.2f}% | {n_params:,} paramètres")
        all_accuracies.append(final_accuracy)

        with open(f"results/academic_results_hybride_{current_seed}.json", "w") as f:
            json.dump({
                "accuracy": final_accuracy, 
                "evals": total_evals, 
                "time": total_time,
                "n_parameters": n_params
            }, f)
            
        torch.save(final_model.state_dict(), f"results/best_model_hybride_seed_{current_seed}.pth")

        # =========================================================
        # NETTOYAGE VRAM FIN DE SEED
        # =========================================================
        del final_model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nMoyenne: {np.mean(all_accuracies)*100:.2f}% ± {np.std(all_accuracies, ddof=1)*100:.2f}%")