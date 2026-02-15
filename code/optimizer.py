import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod
from layer_classes import Conv2dCfg, DropoutCfg, FlattenCfg, LinearCfg, MaxPool2dCfg, GlobalAvgPoolCfg, BatchNorm1dCfg, BatchNorm2dCfg, ResBlockCfg
from model import DynamicNet

class Optimizer(ABC):
    def __init__(self, layers, search_space=None, dataset=None):
        self.layers = layers
        self.search_space = search_space
        self.dataset = dataset
        self.best_score = -float('inf')
        self.best_arch = None

    def evaluate(self, genome, train_epochs=10):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            input_shape = (3, 32, 32)
            model = DynamicNet(genome, input_shape=input_shape)
            model.to(device)
            
            if self.dataset is None:
                inputs = torch.randn(64, 3, 32, 32)
                targets = torch.randint(0, 2, (64,))
                train_loader = DataLoader(TensorDataset(inputs, targets), batch_size=16)
            else:
                train_loader = self.dataset

            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(train_epochs):
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            if total == 0: return 0.0
            return 100. * correct / total

        except Exception:
            return -float('inf')

    def neighbor(self, current_configs):
        new_configs = copy.deepcopy(current_configs)
        options = ["param", "add_layer", "remove_layer", "swap_activation"]
        mutation_type = random.choice(options)

        def is_linear_context_check(target_list, idx):
            if idx == 0: 
                if len(target_list) > 0 and isinstance(target_list[0], (LinearCfg, FlattenCfg)):
                    return True
                return False
            
            prev_layer = target_list[idx - 1]
            if isinstance(prev_layer, (LinearCfg, FlattenCfg, GlobalAvgPoolCfg)):
                return True
            return False

        def get_mutable_layers(current_list, is_root=True):
            candidates = []
            limit = len(current_list) - 1 if is_root else len(current_list)
            for i in range(limit):
                layer = current_list[i]
                if isinstance(layer, ResBlockCfg):
                    candidates.extend(get_mutable_layers(layer.sub_layers, is_root=False))
                elif hasattr(layer, 'activation') or hasattr(layer, 'out_channels') or hasattr(layer, 'out_features'):
                    candidates.append(layer)
            return candidates

        def get_mutable_lists(current_list, is_root=True):
            candidates = [(current_list, is_root)]
            for layer in current_list:
                if isinstance(layer, ResBlockCfg):
                    candidates.extend(get_mutable_lists(layer.sub_layers, is_root=False))
            return candidates

        if mutation_type == "param":
            candidates = get_mutable_layers(new_configs, is_root=True)
            if candidates:
                target = random.choice(candidates)
                self._mutate_layer_param(target)

        elif mutation_type == "swap_activation":
            candidates = get_mutable_layers(new_configs, is_root=True)
            valid = [l for l in candidates if hasattr(l, 'activation')]
            if valid:
                target = random.choice(valid)
                acts = [nn.ReLU, nn.Tanh, nn.LeakyReLU, None] 
                target.activation = random.choice(acts)

        elif mutation_type == "add_layer":
            list_candidates = get_mutable_lists(new_configs, is_root=True)
            if list_candidates:
                target_list, is_root = random.choice(list_candidates)
                
                if is_root:
                    if len(target_list) >= 1:
                        idx = random.randint(0, len(target_list) - 1)
                    else:
                        idx = 0
                else:
                    idx = random.randint(0, len(target_list))

                is_linear = is_linear_context_check(target_list, idx)
                new_layer = self._get_random_layer(linear_context=is_linear)
                target_list.insert(idx, new_layer)

        elif mutation_type == "remove_layer":
            list_candidates = get_mutable_lists(new_configs, is_root=True)
            valid_candidates = []
            for lst, is_root in list_candidates:
                if is_root:
                    if len(lst) > 2: valid_candidates.append((lst, True))
                else:
                    if len(lst) > 0: valid_candidates.append((lst, False))
            
            if valid_candidates:
                target_list, is_root = random.choice(valid_candidates)
                if is_root:
                    idx = random.randint(0, len(target_list) - 2)
                else:
                    idx = random.randint(0, len(target_list) - 1)
                del target_list[idx]

        return new_configs

    def _mutate_layer_param(self, layer):
        if isinstance(layer, Conv2dCfg):
            choice = random.choice(["kernel", "channels"])
            if choice == "kernel":
                delta = random.choice([-2, 2])
                new_k = layer.kernel_size + delta
                layer.kernel_size = int(np.clip(new_k, 1, 7))
                layer.padding = layer.kernel_size // 2
            elif choice == "channels":
                delta = random.choice([-8, -4, 4, 8])
                layer.out_channels = max(4, int(layer.out_channels + delta))

        elif isinstance(layer, LinearCfg):
            delta = random.choice([-16, 8, 8, 16])
            layer.out_features = max(4, int(layer.out_features + delta))

        elif isinstance(layer, DropoutCfg):
            delta = random.uniform(-0.1, 0.1)
            layer.p = np.clip(layer.p + delta, 0.0, 0.8)

    def _get_random_layer(self, linear_context=False):
        if linear_context:
            type_ = random.choice(["linear", "dropout", "bn1d"])
            if type_ == "linear":
                return LinearCfg(in_features=0, out_features=random.randint(16, 128), activation=nn.ReLU)
            elif type_ == "dropout":
                return DropoutCfg(p=0.3)
            elif type_ == "bn1d":
                return BatchNorm1dCfg(num_features=0)
        else:
            type_ = random.choice(["conv", "pool", "bn2d", "dropout"])
            if type_ == "conv":
                k = random.choice([3, 5])
                return Conv2dCfg(in_channels=0, out_channels=random.randint(8, 64), 
                                 kernel_size=k, padding=k//2, activation=nn.ReLU)
            elif type_ == "pool":
                return MaxPool2dCfg(kernel_size=2, stride=2, padding=0)
            elif type_ == "bn2d":
                return BatchNorm2dCfg(num_features=0)
            elif type_ == "dropout":
                return DropoutCfg(p=0.2)
        return DropoutCfg(p=0.1)

    @abstractmethod
    def run(self, n_iterations):
        pass


class SAOptimizer(Optimizer):
    def __init__(self, layers=None, search_space=None, temp_init=100, cooling_rate=0.95, **kwargs):
        super().__init__(layers, search_space, **kwargs)
        self.T = temp_init
        self.alpha = cooling_rate

    def run(self, n_iterations):
        current_sol = copy.deepcopy(self.layers)
        current_score = self.evaluate(current_sol)
        
        if current_score == -float('inf'):
            current_score = 0.0 
        
        initial_score = current_score
        self.best_arch = current_sol
        self.best_score = current_score
        
        best_iter = -1

        for i in range(n_iterations):
            neighbor = self.neighbor(current_sol)
            neighbor_score = self.evaluate(neighbor)
            
            if neighbor_score == -float('inf'): continue 

            delta = neighbor_score - current_score
            if delta > 0 or np.random.rand() < np.exp(delta / self.T):
                current_sol = neighbor
                current_score = neighbor_score
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_arch = copy.deepcopy(current_sol)
                    best_iter = i
            self.T *= self.alpha
            
        stats = {
            "initial_score": initial_score,
            "best_score": self.best_score,
            "best_iter": best_iter,
            "gain": self.best_score - initial_score
        }
        return self.best_arch, stats

class GeneticOptimizer(Optimizer):
    def __init__(self, layers=None, search_space=None, pop_size=10, mutation_rate=0.1, **kwargs):
        super().__init__(layers, search_space, **kwargs)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.population = []

    def run(self, n_generations):
        print("Starting Genetic Algorithm...")
        self.population = [copy.deepcopy(self.layers) for _ in range(self.pop_size)]
        
        for g in range(n_generations):
            scores = []
            valid_pop = []
            
            for ind in self.population:
                score = self.evaluate(ind)
                if score > -float('inf'):
                    scores.append(score)
                    valid_pop.append(ind)
                    if score > self.best_score:
                        self.best_score = score
                        self.best_arch = copy.deepcopy(ind)
                        print(f"Gen {g}: New Best! Score {self.best_score:.2f}")
            
            if not valid_pop:
                self.population = [copy.deepcopy(self.layers) for _ in range(self.pop_size)]
                continue

            sorted_indices = np.argsort(scores)[::-1]
            top_k = max(2, len(valid_pop) // 2)
            parents = [valid_pop[i] for i in sorted_indices[:top_k]]
            
            next_gen = parents[:]
            while len(next_gen) < self.pop_size:
                parent = random.choice(parents)
                child = self.neighbor(parent)
                next_gen.append(child)
            
            self.population = next_gen
            
        return self.best_arch
    
layers = []

depth=3
layers.append(Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=nn.ReLU))
layers.append(BatchNorm2dCfg(num_features=16)) 
for _ in range(depth):

    sub_block = [
        Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=nn.ReLU),
        BatchNorm2dCfg(num_features=16),
        Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=None) 
    ]
        

    layers.append(ResBlockCfg(sub_layers=sub_block))
    layers.append(BatchNorm2dCfg(num_features=16)) 


        
layers.append(GlobalAvgPoolCfg())
layers.append(LinearCfg(in_features=5, out_features=2, activation=None))

net=DynamicNet(layers)

print(net)
"""
print(net.get_graph())
print(net.get_graph()[0].shape)
print(net.get_graph()[1])

net.save_model('test')

net2=DynamicNet.load_model('test')
print(net)
print(net2)
"""
#def random_sol(max_depth, )

opti = SAOptimizer(layers)

print(DynamicNet(opti.neighbor(layers)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Importez vos classes ici si elles sont dans un autre fichier
# from model import DynamicNet
# from optimizer import SAOptimizer
# from layer_classes import *

def run_test():
    print("=== 1. Génération du Dataset Factice ===")
    # On simule 100 images de 32x32 avec 3 canaux (RGB) pour une classification binaire
    # Batch size = 10
    X_train = torch.randn(100, 3, 32, 32)
    y_train = torch.randint(0, 2, (100,)) # Labels 0 ou 1
    
    # Création du DataLoader
    dataset = DataLoader(TensorDataset(X_train, y_train), batch_size=10, shuffle=True)
    print("Dataset créé : 100 images (3, 32, 32)")

    print("\n=== 2. Définition de l'Architecture Initiale ===")
    depth=3
    layers.append(Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=nn.ReLU))
    layers.append(BatchNorm2dCfg(num_features=16)) 
    for _ in range(depth):
    
        sub_block = [
            Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=nn.ReLU),
            BatchNorm2dCfg(num_features=16),
            Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=None) 
        ]
            
    
        layers.append(ResBlockCfg(sub_layers=sub_block))
        layers.append(BatchNorm2dCfg(num_features=16)) 
    
    
            
    layers.append(GlobalAvgPoolCfg())
    layers.append(LinearCfg(in_features=5, out_features=2, activation=None))
    
    net=DynamicNet(layers)
    print(f"Architecture de départ valide. Params: {net.count_parameters()}")

    print("\n=== 3. Lancement de l'Optimisation (Recuit Simulé) ===")
    optimizer = SAOptimizer(
        layers=layers,
        dataset=dataset,
        temp_init=10,       
        cooling_rate=0.5,   
    )


    best_genome = optimizer.run(n_iterations=50)

    print("\n=== 4. Validation du Résultat ===")
    print("Construction du modèle final issu de la recherche...")
    
    final_model = DynamicNet(best_genome, input_shape=(3, 32, 32))
    
    dummy_input = torch.randn(1, 3, 32, 32)
    try:
        output = final_model(dummy_input)
        print("modele trouvé")
        print(f"Shape de sortie : {output.shape} (Attendu: [1, 2])")
        print(f"Nombre de paramètres : {final_model.count_parameters()}")
        print("\nStructure finale du réseau :")
        print(final_model)
        
        final_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            y_pred=final_model(X_train)
            _, pred= torch.max(y_pred, 1)
            total += y_pred.size(0)
            correct += (pred == y_train).sum().item()
        print( 100 * correct / total)
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    run_test()