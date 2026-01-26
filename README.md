# Neural Architecture Search

## Abstract

algorithme qui cherche une architecture de réseaux de neurones optimisée de façon autonome

### 3 composantes

-Search Space : Quelles architectures sont possibles ? (ex: CNN, Transformers, GNN).

-Search Strategy : Comment explorer cet espace ? (ex: Algorithmes évolutionnaires, RL, Gradient).

-Performance Estimation Strategy : Comment évaluer si une architecture est bonne sans l'entraîner pendant des jours ?

## SOTA

### One-Shot NAS et Weight Sharing

entraine un grand modèle contenant plusieurs arch possibles


### D-NAS

utilise GD pour optimiser hyperparam. Peu robuste mais upgrade avec DrNAS & RobustDARTS

### Zero-Cost Proxies

Etimer la perf d'un réseaux sans training avec corrélation synaptique (SynFlow) ou matrice d'information de Fisher

### Hardware-Aware NAS

cherche meilleur réseau selon des critères

---

### Comparatif des Approches

| Approche | Vitesse de recherche | Précision | Complexité de mise en œuvre | Cas d'usage idéal |
| :--- | :--- | :--- | :--- | :--- |
| **Reinforcement Learning** (Old School) | Très lente | Très élevée | Moyenne | Recherche fondamentale, Google-scale compute |
| **Differentiable (DARTS)** | Rapide | Élevée (si stable) | Élevée | Vision par ordinateur standard |
| **One-Shot / Weight Sharing** | Très Rapide | Élevée | Très Élevée | Production, Edge AI, Mobile |
| **Zero-Cost Proxies** | Instantanée | Moyenne/Bonne | Faible | Filtrage initial massif |

---

## algorithmes

* **Once-for-All (OFA)** : Basé sur le One-Shot. On entraîne un grand réseau une seule fois, puis on extrait des sous-réseaux optimisés pour n'importe quel device.
* **Zero-Cost Proxies (SynFlow)** : Méthode de scoring instantané pour évaluer le potentiel d'un réseau sans entraînement.
* **Differentiable NAS (DARTS)** : Optimisation continue de l'architecture via gradient.
* **Algorithmes Évolutionnaires (AmoebaNet)** : Utilise la "Regularized Evolution", un algorithme génétique simple où l'on fait muter les meilleurs modèles.
* **NanoNAS** : Tendance 2024/2025. Spécialisé pour les microcontrôleurs (TinyML) avec de fortes contraintes de RAM.


## Idées

L'idée est que les hyperparamètres d'un NN, donc l'architecture et les caractéristiques des layers sont des variables, et on peut donc représenter les NN comme une fonction qui prend en argument l'architecture A, les poids W, et le dataset de test X pour dooner les prédictions Y : $f(A,W)(X)= Y$, la backpropargation ne modifie que W, en fonction de l'erreur, mais il doit etre possible de déterminer A grâce à des méthodes d'optimisation comme les métaheuristiques, de plus sous contrainte de Pareto pour le temps d'inférence principalement. 


on peut représenter NN par un graphe: un layer = un noeud

appliquer un GNN

encoder avec un GNN


### 1. Concept Fondamental
Contrairement aux approches classiques qui encodent un réseau de neurones sous forme de vecteur plat (perte d'information topologique), ce projet propose une **représentation basée sur les graphes**. 

L'objectif est d'utiliser un **Graph Neural Network (GNN)** comme "Prédicteur de Performance" (Predictor-Based NAS). Ce GNN apprendra à estimer la précision (`Accuracy`) d'une architecture candidate à partir de sa topologie, sans avoir à l'entraîner, accélérant exponentiellement la phase de recherche.

---

### 2. Formalisation Mathématique de l'Encodage

Une architecture neuronale est modélisée comme un graphe orienté acyclique (DAG) défini par le tuple $G = (A, X)$.

#### A. La Matrice d'Adjacence ($A$) - La Topologie
Elle représente les connexions entre les couches (flux de données). Pour un réseau de $N$ nœuds (couches), $A \in \{0,1\}^{N \times N}$.

$$
A_{i,j} = 
\begin{cases} 
1 & \text{si une connexion existe du nœud } i \text{ vers } j \\
0 & \text{sinon}
\end{cases}
$$

*Note : Pour garantir l'acyclicité (DAG), $A$ est généralement contrainte à être triangulaire supérieure ($i < j$).*


Dans notre cas, les réseaux de neurones sont assez peu connectés (généralement un  noeud vers juste un autre, ou deux), il est donc mieux de représenter sous fourmat de vecteur de couple $[(0,1),(1,2),(0,2),...]$ (meilleur que dictionnaire pour GNN)

#### B. La Matrice des Caractéristiques ($X$) - Les Opérations
Elle décrit la nature et les hyperparamètres de chaque couche. Pour $N$ nœuds et $F$ caractéristiques, $X \in \mathbb{R}^{N \times F}$.
Chaque ligne $X_i$ est un vecteur hybride combinant encodage One-Hot et valeurs continues normalisées :

$$
X_i = [\underbrace{t_1, t_2, ..., t_k}_{\text{Type (One-Hot)}}, \underbrace{p_1, p_2, ..., p_m}_{\text{Params (Kernel, Stride...)}}]
$$

*Exemple pour une couche $i$ de type "Conv3x3" :*
$X_i = [0, 1, 0, 0, \ 3, 1, 64]$
*(Où les types sont : Identity, Conv, Pool, Linear... et les params : Kernel=3, Stride=1, Filters=64).*

---

### 3. Pipeline du Projet (Méthodologie)

Le projet se divise en trois phases distinctes :

#### Phase 1 : Collecte de Données (Ground Truth)
* Génération aléatoire de $K$ architectures (graphes $(A, X)$).
* Entraînement réel (rapide) de ces architectures sur un dataset (ex: CIFAR-10) pour obtenir leur précision réelle $y$.
* Constitution du dataset d'entraînement du prédicteur : $\mathcal{D} = \{(G_i, y_i)\}_{i=1}^K$.

#### Phase 2 : Entraînement du Prédicteur GNN
* Architecture : Utilisation d'un GNN performant (ex: **GIN** - Graph Isomorphism Network) capable de capter les structures graphiques.
* Objectif : Minimiser l'erreur de prédiction (MSE) :
    $\mathcal{L} = \frac{1}{K} \sum_{i=1}^K (f_{GNN}(A_i, X_i) - y_i)^2$
* Le GNN apprend que certaines structures (ex: "Skip Connections") corrèlent avec une haute précision.

#### Phase 3 : Recherche par Métaheuristique (Inférence)
* Utilisation d'un algorithme évolutionnaire (Algorithme Génétique ou Recuit Simulé).
* **Fonction de Fitness :** Au lieu d'entraîner le réseau candidat, on passe son graphe $(A, X)$ dans le GNN entraîné.
    $\text{Score} = f_{GNN}(A_{candidat}, X_{candidat})$
* **Temps d'évaluation :** Quelques millisecondes vs heures d'entraînement.

---

### 4. Stack Technique & Faisabilité

* **Langage :** Python.
* **Deep Learning :** PyTorch.

??
* **Graph Library :** **PyTorch Geometric (PyG)** (Standard industriel pour gérer les inputs `Data(x=X, edge_index=A)`).
* **Avantage Académique :** Cette approche respecte l'invariance par permutation des graphes (si on mélange l'ordre des nœuds sans changer les liens, le GNN donne le même score, contrairement à un MLP sur vecteur plat).

### Résumé idée

NN=>encodage (A,X) => métaheur (A',X')=> (GNN =>) décodage => (entrainement =>) test


## Ce que j'ai fait

class Linearcfg, Convcfg,... des classes qui servent de conteneurs pour les paramètres des layers.

la classe DynamicNet qui peut traduire une liste des classes CFG en NN pytorch, ainsi que des méthodes qui renvoient les poids sous forme de vecteur 1D, et la réciproque charge un vecteur 1D dans les poids des neurones.


Etant donné qu'un NN =(A,X,W) avec A le graphe d'adjacence, X l'encodage de chaque layer et W les poids on peut enregistrer le modele sous format npz et le charger afin de pouvoir partager facilement 



## RDV

### RDV 26/01
Ok pour ce que j'ai fait
objectif : faire librairy avec quelques algo et les vérifier via benchmark
