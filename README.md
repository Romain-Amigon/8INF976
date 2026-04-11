

# 8INF976

**AMIGON Romain**

## Introduction

Le but de ce cours était de mener un projet sur les méthodes d'optimisation des architectures des réseaux de neurones (Neural Architecture Search - NAS).

L'idée de ce projet est de répondre à deux problématiques majeures du Deep Learning actuel :
1. Quand on pose la question "Pourquoi avoir choisi ces hyperparamètres ?", la réponse la plus courante reste : "Par expérience".
2. La tendance à concevoir des réseaux de neurones surdimensionnés (obèses) par rapport à la complexité réelle de leur tâche.

J'ai donc cherché à formaliser mathématiquement l'architecture des réseaux de neurones. Actuellement, la norme consiste à concevoir l'architecture à la main, puis à l'entraîner. Mathématiquement, cela revient à définir une fonction $g$ qui associe un espace de poids à un espace de fonctions $\mathcal{F}$, et dont on cherche à trouver l'optimum pour un jeu de données d'entraînement :

$$g : \mathbb{R}^n \rightarrow \mathcal{F}$$

Mon approche consiste à ajouter un niveau d'abstraction en introduisant une fonction $f$ qui prend en argument une architecture spécifique et renvoie la fonction $NN$ correspondante. 

On peut représenter une architecture par un graphe $A$ (topologie) et les paramètres de ses couches $X$. Pour simplifier, nous encodons ces deux dimensions dans une matrice unique $\Theta$ (cf. Annexe pour plus de détails sur l'encodage). Nous définissons ainsi l'espace des architectures possibles, et notre fonction devient :

$$f : \Theta \rightarrow \mathcal{F}$$
$$f(\theta) = g_{\theta}$$

L'objectif central de ce projet est donc de comparer différentes méthodes d'optimisation (Descente de Gradient, métaheuristiques, apprentissage par renforcement, etc.) pour déterminer l'architecture optimale $\theta^*$ qui maximise les performances de la fonction $f$.

> **[TODO : Insérer Schéma 1 - Vue globale]** > *Flux : Dataset $\rightarrow$ Optimizer $\rightarrow$ Train $\rightarrow$ Best NN*

> **[TODO : Insérer Schéma 2 - Boucle de l'optimiseur]**
> *Flux : Modèle de base $\rightarrow$ Evaluate $\rightarrow$ Nouvelle architecture (avec boucle de rétroaction)*

Pour mener à bien cette étude, j'ai décidé de comparer les méthodes suivantes :
* **Recuit simulé :** Une méthode d'optimisation stochastique (similaire à une descente de gradient avec exploration).
* **Algorithme génétique (GA) :** Métaheuristique basée sur l'évolution et la compétition.
* **Artificial Bee Colony (ABC) :** Métaheuristique en essaim basée sur la collaboration.
* **Réseau géniteur LSTM par RL :** L'approche classique de l'état de l'art pour la génération de séquences.
* **Réseau géniteur Transformer par RL :** Mon innovation pour ce projet.

Pour résumer, on formalise un forward dans un réseu de neurone comme ceci : $f(\theta)(W)(X)=y$, avec $\theta$ l'architecture du réseau, optimisée avec mes méthodes, W les poids, optimisé par entraînement, et X les données d'entrée, y la sortie.

---

## Annexe

### Encodage du Réseau de neurones

Comme expliqué précédemment, un réseau de neurones peut être représenté par une matrice $\Theta$. Dans cette matrice, les premières colonnes représentent un vecteur encodé en *One-Hot* définissant le type de la couche (Dense, CNN, ResBlock, Flatten, etc.), suivi des hyperparamètres spécifiques à cette couche (comme la taille du noyau ou le padding). 

J'ai implémenté cette fonctionnalité dans mon code, incluant des méthodes pour encoder et décoder des architectures complètes vers et depuis ce format matriciel, ainsi que la possibilité de les sauvegarder. *(Note : Bien que fonctionnel, ce formatage matriciel strict ne m'a finalement pas été utile pour l'implémentation finale des optimiseurs).*

