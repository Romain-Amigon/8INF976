# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 19:41:48 2026

@author: amigo
"""

import numpy as np
import random
import copy
from model import DynamicNet
import copy
from abc import ABC, abstractmethod
from layer_classes import Conv2dCfg, DropoutCfg, FlattenCfg, LinearCfg, MaxPool2dCfg, GlobalAvgPoolCfg, BatchNorm1dCfg, BatchNorm2dCfg, ResBlockCfg
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
DEFAULT_SEARCH_SPACE = {
    'max_depth': 10,
    'min_depth': 2,
    'CNN': False,
    'min_kernel_size': 3,
    'max_kernel_size': 11,
    'max_features_linear': 100,
    'min_features_linear': 4
}

class Optimizer(ABC):

    def __init__(self, layers, search_space=DEFAULT_SEARCH_SPACE              
                 ,  dataset=None):
        self.search_space = search_space # {CNN : False, max_kernel_size = 11, max_features_linear= 100, ...}
        self.dataset = dataset
        self.history = [] 
        self.best_arch = None
        self.best_score = -float('inf')
        self.layers=layers

    def evaluate(self, genome, train_epochs=1):
        try:

            input_shape = (3, 32, 32) 
            model = DynamicNet(genome, input_shape=input_shape)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            total_loss = 0
            correct = 0
            total = 0
            
            for epoch in range(train_epochs):
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            accuracy = 100. * correct / total
            
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_arch = genome
                print(f"   >>> New Best Found: Acc {accuracy:.2f}% | Loss {total_loss:.4f}")
            
            return accuracy

        except Exception as e:
            print(f"Architecture invalide générée : {e}")
            return -float('inf')
    
    def neighbor(self, current_configs):
        """
        Génère une architecture voisine en modifiant la liste de configurations,
        y compris à l'intérieur des blocs résiduels (récursif).
        """
        # 1. Copie profonde pour ne pas casser l'architecture actuelle
        new_configs = copy.deepcopy(current_configs)
        
        # 2. Choix du type de mutation
        options = ["param", "param", "add_layer", "remove_layer", "swap_activation"]
        mutation_type = random.choice(options)
        
        print(mutation_type)        
        
        def get_all_mutable_layers(layer_list):
            """Récupère récursivement toutes les couches atomiques (Conv, Linear, etc.)"""
            candidates = []
            for layer in layer_list:
                if isinstance(layer, (Conv2dCfg, LinearCfg, DropoutCfg, MaxPool2dCfg, BatchNorm2dCfg)):
                    candidates.append(layer)
                elif isinstance(layer, ResBlockCfg):
                    candidates.extend(get_all_mutable_layers(layer.sub_layers))
            return candidates

        def get_all_mutable_lists(layer_list):
            """Récupère récursivement toutes les LISTES de couches (racine + sub_layers)"""
            lists = [layer_list]
            for layer in layer_list:
                if isinstance(layer, ResBlockCfg):
                    lists.extend(get_all_mutable_lists(layer.sub_layers))
            return lists

        if mutation_type == "param":
            all_layers = get_all_mutable_layers(new_configs)
            
            if all_layers:
                target_layer = random.choice(all_layers[1:-1])
                print(target_layer)
                self._mutate_layer_param(target_layer)

        elif mutation_type == "swap_activation":
            all_layers = get_all_mutable_layers(new_configs)
            candidates = [l for l in all_layers if hasattr(l, 'activation')]
            
            if candidates:
                target_layer = random.choice(candidates[1:-1])
                acts = [nn.ReLU, nn.Tanh, nn.LeakyReLU, None]
                target_layer.activation = random.choice(acts)

        # --- MUTATION 3 : AJOUT DE COUCHE (Dans n'importe quel bloc) ---
        elif mutation_type == "add_layer":
            # On choisit D'ABORD dans quelle liste on veut insérer (Main ou ResBlock)
            all_lists = get_all_mutable_lists(new_configs)
            target_list = random.choice(all_lists)
            
            # Contrainte de profondeur max (optionnelle, ici simplifiée)
            if len(target_list) < self.search_space.get('max_depth', 20):
                insert_idx = random.randint(1, len(target_list)-1)
                
                # Contexte (Linear ou Conv ?)
                # Simplification : On regarde le type de la couche précédente dans la liste cible
                is_linear = False
                if insert_idx > 0 and isinstance(target_list[insert_idx-1], (LinearCfg, FlattenCfg)):
                    is_linear = True
                
                new_layer = self._get_random_layer(linear_context=is_linear)
                target_list.insert(insert_idx, new_layer)

        elif mutation_type == "remove_layer":
            all_lists = get_all_mutable_lists(new_configs)
            # On ne garde que les listes qui ne sont pas vides (ou trop petites)
            valid_lists = [lst for lst in all_lists if len(lst) > 0]
            
            if valid_lists:
                target_list = random.choice(valid_lists[1:-1])
                # On évite de vider complètement un ResBlock s'il doit contenir min 1 couche
                if len(target_list) > 1: 
                    idx = random.randint(0, len(target_list)-1)
                    del target_list[idx]

        return new_configs

    def _mutate_layer_param(self, layer):
        """Helper pour modifier les paramètres d'un objet Cfg spécifique"""
        
        if isinstance(layer, Conv2dCfg):
            choice = random.choice(["kernel", "channels"])
            
            if choice == "kernel":
                # Kernel impair entre min et max
                delta = random.choice([-2, 2])
                new_k = layer.kernel_size + delta
                min_k = self.search_space.get('min_kernel_size', 3)
                max_k = self.search_space.get('max_kernel_size', 7)
                layer.kernel_size = np.clip(new_k, min_k, max_k)
                layer.padding = layer.kernel_size // 2 # Maintenir la dimension spatiale
                
            elif choice == "channels":
                delta = random.choice([-8, -4, 4, 8]) # Pas de petit pas pour les channels
                new_c = layer.out_channels + delta
                layer.out_channels = max(4, int(new_c))

        elif isinstance(layer, LinearCfg):
            delta = random.choice([-16, 8, 8, 16])
            new_f = layer.out_features + delta
            min_f = self.search_space.get('min_features_linear', 10)
            max_f = self.search_space.get('max_features_linear', 512)
            layer.out_features = np.clip(new_f, min_f, max_f)

        elif isinstance(layer, DropoutCfg):
            delta = random.uniform(-0.1, 0.1)
            layer.p = np.clip(layer.p + delta, 0.0, 0.8)

    def _get_random_layer(self, linear_context=False):
        """Génère une couche aléatoire valide"""
        if linear_context:
            # Si on est après un Flatten, on ne peut mettre que Linear, Dropout, BatchNorm1d
            type_ = random.choice(["linear", "dropout", "bn1d"])
            if type_ == "linear":
                return LinearCfg(in_features=0, out_features=random.randint(16, 128), activation=nn.ReLU)
            elif type_ == "dropout":
                return DropoutCfg(p=0.3)
            elif type_ == "bn1d":
                return BatchNorm1dCfg(num_features=0) # Sera calculé auto
        else:
            # Contexte Image (avant Flatten)
            type_ = random.choice(["conv", "pool", "bn2d", "dropout"])
            if type_ == "conv":
                k = random.choice([3, 5, 7])
                return Conv2dCfg(in_channels=0, out_channels=random.randint(8, 64), 
                                 kernel_size=k, padding=k//2, activation=nn.ReLU)
            elif type_ == "pool":
                return MaxPool2dCfg(kernel_size=2, stride=2, padding=0)
            elif type_ == "bn2d":
                return BatchNorm2dCfg(num_features=0)
            elif type_ == "dropout":
                return DropoutCfg(p=0.2)
        
        return DropoutCfg(p=0.1) # Fallback
        
        

    @abstractmethod
    def run(self, n_iterations):
        """Chaque méthode enfant devra implémenter sa propre boucle"""
        pass

class GeneticOptimizer(Optimizer):
    def __init__(self, search_space=DEFAULT_SEARCH_SPACE , pop_size=50, mutation_rate=0.1, **kwargs):
        super().__init__(search_space, **kwargs)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.population = []

    def run(self, n_generations):
        print(f"--- Starting Genetic Search for {n_generations} generations ---")
        
        self.population = [self.search_space.random_arch() for _ in range(self.pop_size)]
        
        for g in range(n_generations):
            print(f"Gen {g+1}/{n_generations}")
            # 2. Evaluate
            scores = [self.evaluate(ind) for ind in self.population]
            
            # 3. Select & Reproduce (Logique spécifique GA)
            # parents = selection(...)
            # children = crossover(...)
            # mutation(...)
            # self.population = next_gen
            pass

class SAOptimizer(Optimizer):
    def __init__(self,layers=None, search_space=DEFAULT_SEARCH_SPACE , temp_init=100, cooling_rate=0.95, **kwargs):
        super().__init__(layers,search_space, **kwargs)
        self.T = temp_init
        self.alpha = cooling_rate

    def run(self, n_iterations):
        print(f"Starting Simulated Annealing")
        current_sol = copy.deepcopy(layers)
        current_score = self.evaluate(current_sol)
        
        for i in range(n_iterations):
            # 1. Voisinage (Mutation légère)
            neighbor = self.neighbor(current_sol)
            neighbor_score = self.evaluate(neighbor)
            
            # 2. Critère d'acceptation (Metropolis)
            delta = neighbor_score - current_score
            if delta > 0 or np.random.rand() < np.exp(delta / self.T):
                current_sol = neighbor
                current_score = neighbor_score
            
            # 3. Refroidissement
            self.T *= self.alpha
        return current_sol

    
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
    # On lance une recherche très courte (5 itérations) pour tester
    optimizer = SAOptimizer(
        layers=layers,
        dataset=dataset,
        temp_init=10,       # Température initiale
        cooling_rate=0.5,   # Refroidissement rapide pour le test
    )

    # Lancement de la boucle
    # Cela va afficher "Accept" ou "Reject" et le score (Accuracy)
    best_genome = optimizer.run(n_iterations=5)

    print("\n=== 4. Validation du Résultat ===")
    print("Construction du modèle final issu de la recherche...")
    
    # On instancie le modèle final avec l'input shape pour calculer les dimensions
    final_model = DynamicNet(best_genome, input_shape=(3, 32, 32))
    
    # Test d'inférence (Forward pass)
    dummy_input = torch.randn(1, 3, 32, 32)
    try:
        output = final_model(dummy_input)
        print("✅ Succès ! Le modèle final a réussi une passe avant (forward).")
        print(f"Shape de sortie : {output.shape} (Attendu: [1, 2])")
        print(f"Nombre de paramètres : {final_model.count_parameters()}")
        print("\nStructure finale du réseau :")
        print(final_model)
    except Exception as e:
        print(f"❌ Erreur lors de l'inférence du modèle final : {e}")

if __name__ == "__main__":
    run_test()