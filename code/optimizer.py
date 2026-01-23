import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from mealpy import FloatVar
from mealpy.swarm_based import GWO, PSO, WOA, ABC, SMO, HHO
from mealpy.bio_based import SMA
from mealpy.evolutionary_based import GA, DE

import random as rd
import copy

from layer_classes import Conv2dCfg, DropoutCfg, FlattenCfg, LinearCfg, MaxPool2dCfg, GlobalAvgPoolCfg, BatchNorm1dCfg, BatchNorm2dCfg, ResBlockCfg

FEATURE_SIZE=15

class ResidualWrapper(nn.Module):
    def __init__(self, sub_layers_module, use_projection=False, in_channels=0, out_channels=0):
        super().__init__()
        self.net = sub_layers_module
        self.use_projection = use_projection
        self.projection = None
        

        if use_projection and in_channels != out_channels:

            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        elif in_channels != out_channels:

             pass

    def forward(self, x):
        identity = x
        out = self.net(x)
        
        if self.projection is not None:
            identity = self.projection(identity)

        if identity.shape != out.shape:
            return out 
            
        return out + identity

class DynamicNet(nn.Module):
    """
    A PyTorch Neural Network module that dynamically builds its architecture 
    based on a provided list of layer configurations.

    This class serves as a flexible wrapper to instantiate models with varying
    structures (Linear, Conv2d, Pooling, etc.) on the fly, which is essential
    for Neural Architecture Search (NAS).

    Args:
        layers_cfg (list): A list of configuration objects (e.g., LinearCfg, Conv2dCfg)
                           defining the sequence of layers.
    """
    def __init__(self,layers_cfg: list, input_shape: tuple = None):
        super().__init__()
        if input_shape is not None:
            self.layers_cfg = self._reconnect_layers(layers_cfg, input_shape)
        else:
            self.layers_cfg = layers_cfg

        self.net = self._build_sequential(self.layers_cfg)
        
    def _build_sequential(self, cfgs):
        layers = []
        for cfg in cfgs:
            if isinstance(cfg, LinearCfg):
                layers.append(nn.Linear(cfg.in_features, cfg.out_features))
                if cfg.activation:
                    layers.append(cfg.activation())
            elif isinstance(cfg, Conv2dCfg):
                layers.append(nn.Conv2d(cfg.in_channels, cfg.out_channels, 
                                        cfg.kernel_size, cfg.stride, cfg.padding))
                if cfg.activation:
                    layers.append(cfg.activation())
            elif isinstance(cfg, DropoutCfg):
                layers.append(nn.Dropout(p=cfg.p))
            elif isinstance(cfg, FlattenCfg):
                layers.append(nn.Flatten(start_dim=cfg.start_dim))
            elif isinstance(cfg, MaxPool2dCfg):
                layers.append(nn.MaxPool2d(kernel_size=cfg.kernel_size, stride=cfg.stride, padding=cfg.padding,  ceil_mode=cfg.ceil_mode) )
            elif isinstance(cfg, GlobalAvgPoolCfg):
                layers.append(nn.AdaptiveAvgPool2d((1, 1)))
                layers.append(nn.Flatten())
            elif isinstance(cfg, BatchNorm1dCfg):
                layers.append(nn.BatchNorm1d(cfg.num_features))
                
            elif isinstance(cfg, BatchNorm2dCfg):
                layers.append(nn.BatchNorm2d(cfg.num_features))

            elif isinstance(cfg, ResBlockCfg):
                inner_seq = self._build_sequential(cfg.sub_layers)
                

                in_ch = 0
                out_ch = 0
                if len(cfg.sub_layers) > 0:
                    first = cfg.sub_layers[0]
                    last = cfg.sub_layers[-1]
                    if hasattr(first, 'in_channels'): in_ch = first.in_channels
                    elif hasattr(first, 'in_features'): in_ch = first.in_features 
                    
                    if hasattr(last, 'out_channels'): out_ch = last.out_channels
                    elif hasattr(last, 'out_features'): out_ch = last.out_features

                wrapper = ResidualWrapper(inner_seq, cfg.use_projection, in_ch, out_ch)
                layers.append(wrapper)
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the sequential network.
        """
        return self.net(x)

    def count_parameters(self):
        """
        Counts the total number of trainable parameters in the network.

        Returns:
            int: The total number of elements in all parameters.
        """
        return sum(p.numel() for p in self.parameters())

    def flatten_weights(self, to_numpy=True, device=None):
        """
        Flattens all network parameters into a single vector.

        Args:
            to_numpy (bool, optional): If True, returns a NumPy array. If False,
                                       returns a PyTorch tensor. Defaults to True.
            device (torch.device, optional): The target device for the tensor if
                                             returning a tensor. Defaults to None.

        Returns:
            np.ndarray or torch.Tensor: A 1D array/tensor containing all model weights.
        """
        vec = parameters_to_vector(self.parameters())
        if to_numpy:
            return vec.detach().cpu().numpy()
        return vec.to(device) if device is not None else vec

    def load_flattened_weights(self, flat_weights):
        """
        Loads a flat vector of weights into the network's parameters.

        Args:
            flat_weights (np.ndarray or torch.Tensor): A 1D array or tensor representing
                                                       the weights to load.
        """
        if isinstance(flat_weights, np.ndarray):
            flat_weights = torch.as_tensor(flat_weights, dtype=torch.float32)
        
        device = next(self.parameters()).device
        flat_weights = flat_weights.to(device)
        
        try:
            vector_to_parameters(flat_weights, self.parameters())
        except RuntimeError:
            pass

    def evaluate_model(self, X, y, loss_fn=nn.MSELoss(), n_warmup=3, n_runs=20, verbose=False):
        """
        Evaluates the model on a given dataset, measuring loss and inference latency.

        Args:
            X (torch.Tensor): Input data.
            y (torch.Tensor): Target labels or values.
            loss_fn (nn.Module, optional): The loss function to use. Defaults to nn.MSELoss().
            n_warmup (int, optional): Number of warm-up runs for timing. Defaults to 3.
            n_runs (int, optional): Number of runs to calculate median inference time. Defaults to 20.
            verbose (bool, optional): If True, prints evaluation results. Defaults to False.

        Returns:
            tuple: A tuple containing (loss_value, inference_time_in_seconds).
        """
        model = self.net
        model.eval()

        use_cuda = torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        
        if next(model.parameters()).device.type != device:
            model = model.to(device)

        X = X.to(device)
        y = y.to(device)

        with torch.no_grad():
            pred = model(X)
            loss_value = loss_fn(pred, y).item()

        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(X)
            if use_cuda:
                torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = model(X)
                if use_cuda:
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

        inference_time = float(np.median(times))

        if verbose:
            print(
                f"Loss: {loss_value:.6f} | "
                f"Inference time (median): {inference_time*1000:.3f} ms | "
                f"Input: {tuple(X.shape)}"
            )

        return loss_value, inference_time
    
    def _reconnect_layers(self, layers, input_shape):
        
        dummy_input = torch.zeros(1, *input_shape)
        
        def process_recursive(cfg_list, current_tensor):
            processed = []
            x = current_tensor
            
            for original_cfg in cfg_list:

                import copy
                cfg = copy.deepcopy(original_cfg)
                
                try:
                    if isinstance(cfg, Conv2dCfg):
                        cfg.in_channels = x.shape[1]
                        layer = nn.Conv2d(cfg.in_channels, cfg.out_channels, cfg.kernel_size, cfg.stride, cfg.padding)
                        x = layer(x)
                        processed.append(cfg)

                    elif isinstance(cfg, BatchNorm2dCfg):
                        cfg.num_features = x.shape[1]
                        layer = nn.BatchNorm2d(cfg.num_features)
                        x = layer(x)
                        processed.append(cfg)

                    elif isinstance(cfg, LinearCfg):
                        if len(x.shape) > 2:
                            flat_cfg = FlattenCfg()
                            processed.append(flat_cfg)
                            x = torch.flatten(x, 1)
                        
                        cfg.in_features = x.shape[1]
                        layer = nn.Linear(cfg.in_features, cfg.out_features)
                        x = layer(x)
                        processed.append(cfg)

                    elif isinstance(cfg, ResBlockCfg):
 
                        inner_cfgs, inner_out = process_recursive(cfg.sub_layers, x)
                        cfg.sub_layers = inner_cfgs
                        
                        x = inner_out
                        processed.append(cfg)
                    
                    elif isinstance(cfg, GlobalAvgPoolCfg):
                        x = nn.AdaptiveAvgPool2d((1, 1))(x)
                        x = torch.flatten(x, 1)
                        processed.append(cfg)
                    
                    elif isinstance(cfg, FlattenCfg):
                        x = torch.flatten(x, cfg.start_dim)
                        processed.append(cfg)

                    else:
                        processed.append(cfg)
                        
                except Exception as e:
                    print(f"Warning:{cfg}: {e}")
                    pass
            
            return processed, x

        new_layers, _ = process_recursive(layers, dummy_input)
        return new_layers

    def _encode_config_to_vector(self, cfg):
        """
        Transforme une configuration de couche en un vecteur numérique (Features X).
        Vecteur de taille 15 :
        [0-7] : One-Hot Encoding du Type (Conv, Linear, Pool, BN, Drop, Flat, Res, Autre)
        [8]   : Kernel Size (normalisé /7)
        [9]   : Stride (normalisé /4)
        [10]  : Padding (normalisé /4)
        [11]  : In_Channels/Features (normalisé /1024)
        [12]  : Out_Channels/Features (normalisé /1024)
        [13]  : Dropout Probability
        [14]  : Flag 'Use Projection' (pour ResBlock)
        """
        vec = np.zeros(
        FEATURE_SIZE, dtype=np.float32)
        
        # --- PARTIE 1 : TYPES (Indices 0 à 7) ---
        if isinstance(cfg, Conv2dCfg):
            vec[0] = 1
            # Params spécifiques
            vec[8] = cfg.kernel_size / 7.0 if hasattr(cfg, 'kernel_size') else 0
            vec[9] = cfg.stride / 4.0 if hasattr(cfg, 'stride') else 0
            vec[10] = cfg.padding / 4.0 if hasattr(cfg, 'padding') else 0
            
        elif isinstance(cfg, LinearCfg):
            vec[1] = 1
            
        elif isinstance(cfg, (MaxPool2dCfg, GlobalAvgPoolCfg)):
            vec[2] = 1
            if hasattr(cfg, 'kernel_size'):
                vec[8] = cfg.kernel_size / 7.0
            if hasattr(cfg, 'stride'):
                vec[9] = cfg.stride / 4.0
                
        elif isinstance(cfg, (BatchNorm1dCfg, BatchNorm2dCfg)):
            vec[3] = 1
            if hasattr(cfg, 'num_features'):
                # BN conserve le nombre de features, on le met en In et Out
                vec[11] = cfg.num_features / 1024.0
                vec[12] = cfg.num_features / 1024.0

        elif isinstance(cfg, DropoutCfg):
            vec[4] = 1
            vec[13] = cfg.p
            
        elif isinstance(cfg, FlattenCfg):
            vec[5] = 1
            
        elif isinstance(cfg, ResBlockCfg):
            vec[6] = 1
            vec[14] = 1.0 if cfg.use_projection else 0.0
            # Note: Le contenu du ResBlock est géré par la récursion du graphe, 
            # ici on encode juste le "conteneur".
            
        else:
            vec[7] = 1 # Type inconnu/Autre

        # --- PARTIE 2 : PARAMETRES COMMUNS (Canaux) ---
        # On essaie de récupérer in_channels/features de manière générique
        if hasattr(cfg, 'in_channels'):
            vec[11] = cfg.in_channels / 1024.0
        elif hasattr(cfg, 'in_features'):
            vec[11] = cfg.in_features / 1024.0
            
        if hasattr(cfg, 'out_channels'):
            vec[12] = cfg.out_channels / 1024.0
        elif hasattr(cfg, 'out_features'):
            vec[12] = cfg.out_features / 1024.0
            
        return vec
    
    def get_graph_matrix(self):
        """
        Transforme la layers_cfg (arbre) en matrices A et X (graphe plat).
        """
        features_list = [] # Deviendra X
        edges_list = []    # Deviendra A (format COO: [src, dst])
        
        node_counter = 0
        
        def traverse(configs, previous_node_idx):
            nonlocal node_counter
            last_node_idx = previous_node_idx
            
            for cfg in configs:
                if isinstance(cfg, ResBlockCfg):
                    input_node_of_block = last_node_idx
                    
                    output_node_of_block = traverse(cfg.sub_layers, input_node_of_block)
                    

                    edges_list.append([input_node_of_block, output_node_of_block])
                    
                    last_node_idx = output_node_of_block
                    
                else:
                    node_counter += 1
                    current_node_idx = node_counter
                    
                    feat_vector = self._encode_config_to_vector(cfg)
                    features_list.append(feat_vector)
                    
                    edges_list.append([last_node_idx, current_node_idx])
                    
                    last_node_idx = current_node_idx
            
            return last_node_idx


        features_list.append([0]*FEATURE_SIZE) 
        traverse(self.layers_cfg, 0)
        
        return np.array(edges_list), np.array(features_list)

        


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
layers.append(LinearCfg(in_features=0, out_features=2, activation=None))

net=DynamicNet(layers)

print(net)

print(net.get_graph_matrix())