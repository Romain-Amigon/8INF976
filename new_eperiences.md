# Nouvelles expériences


## Commentaires du prof
I. Commentaires sur le fond 
1. Comparatif dans l’étude d’ablation.
Le Tableau II compare ABC seul (30 itérations, environ 3.5 h) à Transformer+ABC (20 + 15 = 35 itérations, environ 4.8h). L’approche hybride bénéficie d’environ 17% d’itérations supplémentaires et de 37 % de temps d’exécution supplémentaire. La prétendue résolution du problème de démarrage à froid est donc biaisée. Une partie, voire la totalité, du gain observé (79.76 % / 83.48 %) peut simplement provenir de cette différence additionnelle. Sur ce genre de point un reviewer pour une top conférence en ML (genre ICML) peut être vraiment tannant. Il serait préférable de réaliser une comparaison équitable, c’est-à-dire qui exigerait un nombre identique d’itérations et/ou un nombre identique d’heures de compute. De plus, avec des écarts-types de ±2.37 % et ±1.98 %, les intervalles se chevauchent presque entièrement. Il faudait aussi peut être ajouter un test statistique (est-ce que c’est significatif de ce point de vu et rapporter le nombre d’exécutions.
2. Travaux sur la recherche aléatoire (Random Search)
La littérature sur le NAS (https://arxiv.org/abs/1902.07638 ; https://arxiv.org/abs/2001.00326) a montré à plusieurs reprises que la recherche aléatoire constitue une référence étonnamment robuste dans les espaces de recherche de petite taille. Avec seulement 15 tokens et une profondeur limitée, tu devrais probablement ajouter une comparaison de ton travail à ces résultats.
3. Apport par vis-à-vis de ResNet.
ResNet-20 (approx. 0.27M de paramètres) atteint environ 91.25% sur CIFAR-10, soit moins de paramètres et environ 7 points de précision supplémentaires par rapport au meilleur résultat présenté (Seed-42 : 432k paramètres, 84.29%). Ton argumentaire dans le papier est trop faible par rapport à ce point-là. Il faut arriver à mieux convaincre un reviewer un peut tannant de te détruire avec une simple question genre « Ok, c’est cool, mais en fait je peux juste utiliser ResNet et ton truc sert à rien… » (oui ça arrive souvent…).
4. Manque ablation LSTM warm-start
La contribution principale affirme que le Transformer corrige le biais séquentiel des RNN. Pourtant, le Tableau I montre que le LSTM surpasse le Transformer sur Breast Cancer (99.56% contre 98.90%) et n’est que marginalement en dessous sur la tâche de régression (-0.39 contre -0.36, alors que l’écart-type du Transformer, ±0.04, couvre une grande partie de cette différence). Afin d’isoler la contribution réelle du Transformer, la chaîne mémétique devrait être évaluée avec une initialisation basée sur un LSTM.
5.  La comparaison avec l’état de l’art (Tableau III) n’est pas équitable et l’affirmation concernant Pareto est un peu sur-vendue.
Ton article reconnaît lui-même l’existence de facteurs de confusion (espaces de recherche différents, 100 contre 600 époques d’entraînement final), tout en affirmant malgré tout « déplacer drastiquement la frontière de Pareto ». DARTS atteint 97% avec 1.5 jour-GPU, tandis que nas-torch atteint entre 80.75% et 85.39% avec 0.2 jour-GPU. Une réduction de coût de 7.5 pour une perte d’environ 12 points de précision ne constitue pas nécessairement une amélioration de Pareto. Cette conclusion est d’autant plus discutable que les jours-GPU ne sont pas normalisés par rapport au matériel utilisé (une RTX 3060 en 2026 contre des GPU K40/P100 dans les références [1] et [2]).
6. Des hyperparamètres essentiels ne sont pas rapportés, ce qui nuit à la reproductibilité de ton travail.
Le coefficient de pénalisation de profondeur λ n’est pas précisé. Il en va de même pour :
	le coefficient d’entropie β ;
	sa stratégie d’augmentation ;
	la fenêtre de stagnation N ;
	la taille de la colonie ABC ;
	le paramètre limit ;
	le nombre d’époques utilisé pour les évaluations proxy ;
	le nombre de graines aléatoires utilisées pour chaque résultat.
Pour un article dont la contribution est essentiellement empirique, cette omission est rédhibitoire en l’état.
7. La fidélité du proxy n’est jamais validée.
Les architectures sont classées après seulement 5 à 10 époques d’entraînement, mais aucune mesure de corrélation entre ce classement proxy et les performances après entraînement complet n’est fournie. Or, il est bien connu que les proxies à faible fidélité peuvent produire des classements erronés (cf. [4]). Sans analyse de corrélation de type Kendall-τ ou Spearman, la qualité réelle du signal de recherche demeure inconnue.
8. Certaines affirmations comparatives ne sont pas étayées.
L’article affirme rivaliser avec des méthodes d’ensemble ajustées manuellement telles que le Gradient Boosting et des modèles d’ensemble complexes tels que Random Forest ou XGBoost. Cependant, aucune de ces références n’a été exécutée dans les expériences. Sur l’ensemble de données standard de fraude par carte de crédit disponible sur Kaggle, un modèle XGBoost correctement ajusté dépasse généralement sans difficulté une F1-score de 0.77. De ce fait, tu devrais soit réaliser ces comparaisons, soit retirer ces affirmations.
II. Commentaires sur la forme
	Dans la section III-B, tu as deux fois le paragraphe qui commence par : « We frame the generation process as a Natural Language Processing (NLP) task… »

	Le Tableau I présente des valeurs telles que « −0,40 (MSE) ». Or, la MSE est par définition toujours positive ou nulle.

	Dans ton texte, tu indiques que le Transformer atteint une MSE maximale de −0.31, alors que le tableau rapporte −0.36 ± 0.04. La distinction entre valeur maximale et moyenne devrait être explicitement indiquée. Le fait de mettre en avant uniquement la meilleure valeur dans le texte est trompeur.

	L’équation 6 omet le terme de base, alors que la fonction de perte présentée dans la section III-C repose sur la notion d’avantage. L’estimateur de l’avantage ou de la baseline n’est jamais défini. Ainsi rédigée, l’équation correspond à REINFORCE appliqué à la récompense brute, ce qui est incohérent avec la perte qui est implémentée.

	Dans l’équation 6 la notation : a_((t-1):1) n’est pas standard et devrait probablement être a_(1:(t-1) ).

	Dans l’équation 1, ta formulation est imprécise. La fonction g:R^n→F associe un vecteur de poids à une fonction, puis le texte indique que l’objectif est de « trouver les poids optimaux ». La formulation mélange donc la fonction de paramétrisation et le problème d’optimisation lui-même. Le problème est mineur, mais puisque ton article apporte quand même une certaine rigueur mathématique tu devrais clarifier ce point.
III. Fautes et remarques en vrac
	« can be foud here ».
	« expensive.. ».
	Les figures 1 à 3 pourrait être fusionnées en une seule figure (avec potentiellement des sous-figures).
	Les résultats CIFAR-100 (52.23%) sont mentionnés dans le texte sans tableau récapitulatif, sans écart-type et sans seed.
	Vérifier la référence arXiv:2605.04057 qui me semble étrange.
	« the school-project repositories can be foud here ».
	« school-project » est non nécessaire pour un article scientifique. Conserve uniquement le lien clean.
	« is computationally expensive.. ».
	« optimizers based on metaheuristics : Simulated Annealing ». Pas d’espace avant « : »| en anglais.
	Le titre des sectitons III. Et IV ne sont vraiment pas terrible. Essaye de trouver quelque chose de plus explicite, plus scientifique.
	Ton utilisation récurrente de dangling modifiers, est parfois ambiguë.
	Remplacer « for having followed this project » par « for supervising this project » ou « for his guidance throughout this project ».
	Attention à l’usage excessif d’adjectifs trop promotionnels (« ultra-lightweight », « drastically shifts », « exceptionally lightweight », etc.).
	Revoir les majuscules ou ce n’est pas toujours nécessaire (« Cold-Start », « Warm-Start », « Accuracy », etc.).
	Il y a certaines redondances entre l’abstract, l’introduction et l’état de l’art, tu pourrais simplement reformuler certaines phrases en paraphrasant ce qui est redondant.
	Revoir certaines phrases qui sont assez longues.
	Tu devrais revoir les légendes de tes tableaux et de tes figures. En fait elles devraient être autonome et expliquer clairement ce que le lecteur doit retenir de la figure sans avoir lu le reste du texte autour (on commence souvent par se faire une première tête rapide d’un article en lisant titre, abstract et captions des figures/tableaux).



## Vérfication proxy 10 épochs -> 100 epochs

architectures générée par ABCoptimizer puis 15 choisies aléatoirement dedans

--- Modèle 1/15 ---
  -> Score Proxy (10 epochs) déjà calculé : 53.79%
  -> Entraînement Full (100 epochs)...
  => Proxy (10ep): 53.79% | Full (100ep): 66.56%

--- Modèle 2/15 ---
  -> Score Proxy (10 epochs) déjà calculé : 47.68%
  -> Entraînement Full (100 epochs)...
  => Proxy (10ep): 47.68% | Full (100ep): 53.23%

--- Modèle 3/15 ---
  -> Score Proxy (10 epochs) déjà calculé : 52.99%
  -> Entraînement Full (100 epochs)...
  => Proxy (10ep): 52.99% | Full (100ep): 66.69%

--- Modèle 4/15 ---
  -> Score Proxy (10 epochs) déjà calculé : 44.71%
  -> Entraînement Full (100 epochs)...
  => Proxy (10ep): 44.71% | Full (100ep): 54.27%

--- Modèle 5/15 ---
  -> Score Proxy (10 epochs) déjà calculé : 54.95%
  -> Entraînement Full (100 epochs)...
  => Proxy (10ep): 54.95% | Full (100ep): 73.17%

--- Modèle 6/15 ---
  -> Score Proxy (10 epochs) déjà calculé : 55.71%
  -> Entraînement Full (100 epochs)...
  => Proxy (10ep): 55.71% | Full (100ep): 65.44%

--- Modèle 7/15 ---
  -> Score Proxy (10 epochs) déjà calculé : 53.29%
  -> Entraînement Full (100 epochs)...
  => Proxy (10ep): 53.29% | Full (100ep): 64.59%

--- Modèle 8/15 ---
  -> Score Proxy (10 epochs) déjà calculé : 47.18%
  -> Entraînement Full (100 epochs)...
  => Proxy (10ep): 47.18% | Full (100ep): 57.28%

--- Modèle 9/15 ---
  -> Score Proxy (10 epochs) déjà calculé : 47.91%
  -> Entraînement Full (100 epochs)...
  => Proxy (10ep): 47.91% | Full (100ep): 51.92%

--- Modèle 10/15 ---
  -> Score Proxy (10 epochs) déjà calculé : 53.51%
  -> Entraînement Full (100 epochs)...
  => Proxy (10ep): 53.51% | Full (100ep): 65.56%

--- Modèle 11/15 ---
  -> Score Proxy (10 epochs) déjà calculé : 55.57%
  -> Entraînement Full (100 epochs)...
  => Proxy (10ep): 55.57% | Full (100ep): 73.48%

--- Modèle 12/15 ---
  -> Score Proxy (10 epochs) déjà calculé : 52.66%
  -> Entraînement Full (100 epochs)...
  => Proxy (10ep): 52.66% | Full (100ep): 64.45%

--- Modèle 13/15 ---
  -> Score Proxy (10 epochs) déjà calculé : 52.71%
  -> Entraînement Full (100 epochs)...
  => Proxy (10ep): 52.71% | Full (100ep): 64.38%

--- Modèle 14/15 ---
  -> Score Proxy (10 epochs) déjà calculé : 48.32%
  -> Entraînement Full (100 epochs)...
  => Proxy (10ep): 48.32% | Full (100ep): 71.20%

--- Modèle 15/15 ---
  -> Score Proxy (10 epochs) déjà calculé : 55.35%
  -> Entraînement Full (100 epochs)...
  => Proxy (10ep): 55.35% | Full (100ep): 67.85%

==================================================
RÉSULTAT DE LA FIDÉLITÉ DU PROXY
==================================================
Corrélation de Spearman (rho) : 0.721
P-value : 0.0024



# Fraude bancaire

========================================

RUNNING: C:/Users/ramigon/Downloads/8INF976/code/test_bank_eva.py

========================================



Utilisation du device : cuda

Chargement et transfert des données en VRAM...

Train: 182276 | Val: 45569 | Test: 56962



==================================================

FRAUD - SEED: 42

==================================================

Phase Transformer (recherche macro)...

C:\Users\ramigon\.conda\envs\pytorch_gpu\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)

  warnings.warn(

Transformer Iter 0: New Best Score 72.16 (Depth: 3)

Transformer Iter 1: New Best Score 73.75 (Depth: 13)

Transformer Iter 4: New Best Score 77.92 (Depth: 5)

Phase ABC (exploitation locale)...

ABC Iter 0: Best Score 75.16

ABC Iter 1: Best Score 77.98

ABC Iter 2: Best Score 77.98

ABC Iter 3: Best Score 78.71

ABC Iter 4: Best Score 78.71

ABC Iter 5: Best Score 78.71

ABC Iter 6: Best Score 78.71

ABC Iter 7: Best Score 78.71

ABC Iter 8: Best Score 78.71

Early stopping déclenché à l'itération 8 : Aucun gain depuis 5 itérations.

Bilan recherche : 572 évaluations proxy en 23.4 min

Entraînement final (100 epochs)...

Seuil optimal sur validation : 0.99 (F1 val = 0.7089)

--> Seed 42 | F1=0.7013 | Precision=0.6090 | Recall=0.8265 | AUPRC=0.8068 | 3,137 paramètres

              precision    recall  f1-score   support



      Normal       1.00      1.00      1.00     56864

       Fraud       0.61      0.83      0.70        98



    accuracy                           1.00     56962

   macro avg       0.80      0.91      0.85     56962

weighted avg       1.00      1.00      1.00     56962





==================================================

FRAUD - SEED: 43

==================================================

Phase Transformer (recherche macro)...

C:\Users\ramigon\.conda\envs\pytorch_gpu\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)

  warnings.warn(

Transformer Iter 0: New Best Score 68.33 (Depth: 3)

Transformer Iter 0: New Best Score 68.71 (Depth: 16)

Transformer Iter 1: New Best Score 70.56 (Depth: 31)

Transformer Iter 1: New Best Score 71.82 (Depth: 10)

Transformer Iter 2: New Best Score 72.11 (Depth: 47)

Transformer Iter 4: New Best Score 74.12 (Depth: 48)

Phase ABC (exploitation locale)...

ABC Iter 0: Best Score 77.18

ABC Iter 1: Best Score 77.18

ABC Iter 2: Best Score 79.34

ABC Iter 3: Best Score 79.34

ABC Iter 4: Best Score 79.34

ABC Iter 5: Best Score 79.34

ABC Iter 6: Best Score 79.34

ABC Iter 7: Best Score 79.34

Early stopping déclenché à l'itération 7 : Aucun gain depuis 5 itérations.

Bilan recherche : 525 évaluations proxy en 36.2 min

Entraînement final (100 epochs)...

Seuil optimal sur validation : 0.99 (F1 val = 0.7273)

--> Seed 43 | F1=0.7022 | Precision=0.6220 | Recall=0.8061 | AUPRC=0.7416 | 7,441 paramètres

              precision    recall  f1-score   support



      Normal       1.00      1.00      1.00     56864

       Fraud       0.62      0.81      0.70        98



    accuracy                           1.00     56962

   macro avg       0.81      0.90      0.85     56962

weighted avg       1.00      1.00      1.00     56962





==================================================

FRAUD - SEED: 44

==================================================

Phase Transformer (recherche macro)...

C:\Users\ramigon\.conda\envs\pytorch_gpu\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)

  warnings.warn(

Transformer Iter 0: New Best Score 69.37 (Depth: 10)

Transformer Iter 0: New Best Score 71.48 (Depth: 20)

Transformer Iter 0: New Best Score 72.12 (Depth: 6)

Transformer Iter 0: New Best Score 72.52 (Depth: 16)

Transformer Iter 1: New Best Score 72.97 (Depth: 28)

Transformer Iter 1: New Best Score 73.13 (Depth: 13)

Transformer Iter 2: New Best Score 73.27 (Depth: 7)

Transformer Iter 6: New Best Score 73.89 (Depth: 35)

Phase ABC (exploitation locale)...

ABC Iter 0: Best Score 73.67

ABC Iter 1: Best Score 73.67

ABC Iter 2: Best Score 73.96

ABC Iter 3: Best Score 74.89

ABC Iter 4: Best Score 77.02

ABC Iter 5: Best Score 77.02

ABC Iter 6: Best Score 77.02

ABC Iter 7: Best Score 79.86

ABC Iter 8: Best Score 79.86

ABC Iter 9: Best Score 79.86

ABC Iter 10: Best Score 79.86

ABC Iter 11: Best Score 79.86

ABC Iter 12: Best Score 79.86

Early stopping déclenché à l'itération 12 : Aucun gain depuis 5 itérations.

Bilan recherche : 747 évaluations proxy en 30.5 min

Entraînement final (100 epochs)...

Seuil optimal sur validation : 0.99 (F1 val = 0.7333)

--> Seed 44 | F1=0.7500 | Precision=0.7091 | Recall=0.7959 | AUPRC=0.7609 | 3,265 paramètres

              precision    recall  f1-score   support



      Normal       1.00      1.00      1.00     56864

       Fraud       0.71      0.80      0.75        98



    accuracy                           1.00     56962

   macro avg       0.85      0.90      0.87     56962

weighted avg       1.00      1.00      1.00     56962





==================================================

BILAN GLOBAL FRAUD (3 seeds)

==================================================

F1-Score  : 0.7178 ± 0.0279

Precision : 0.6467 ± 0.0544

Recall    : 0.8095 ± 0.0156

AUPRC     : 0.7698 ± 0.0335

Params    : 4614 ± 2449

Évaluations: 615 ± 117

Temps NAS : 30.0 ± 6.4 min

# Le reste 

Proxy : 10 epochs

cifar hybride : 20 + 15 run
cifar sa : temperature décroissant = 0.99, 800 run
cfar abc : pop_size=20, 30 run
cifar random : max 940 architectures.

==================================================
STARTING ALL EXPERIMENTS AUTOMATICALLY
==================================================

========================================
RUNNING: C:/Users/ramigon/Downloads/8INF976/code/Cifar_hybrid.py
========================================

Utilisation du device : cuda
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified

==================================================
RECHERCHE HYBRIDE - SEED: 42
==================================================
Préchargement du proxy sur GPU...
  Fait en 4.2s

Début Transformer...
C:\Users\ramigon\.conda\envs\pytorch_gpu\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Transformer Iter 0: New Best Score 55.66 (Depth: 5)
Transformer Iter 0: New Best Score 56.60 (Depth: 5)
Transformer Iter 0: New Best Score 61.04 (Depth: 12)
Transformer Iter 1: New Best Score 63.56 (Depth: 10)
Transformer Iter 4: New Best Score 65.24 (Depth: 8)

Début ABC...
C:\Users\ramigon\.conda\envs\pytorch_gpu\Lib\site-packages\torch\nn\init.py:511: UserWarning: Initializing zero-element tensors is a no-op
  warnings.warn("Initializing zero-element tensors is a no-op")
ABC Iter 0: Best Score 67.60
ABC Iter 1: Best Score 67.60
ABC Iter 2: Best Score 68.28
ABC Iter 3: Best Score 68.28
ABC Iter 4: Best Score 68.28
ABC Iter 5: Best Score 68.28
ABC Iter 6: Best Score 68.28
ABC Iter 7: Best Score 68.28
Early stopping déclenché à l'itération 7 : Aucun gain depuis 5 itérations.
Bilan recherche : 689 évaluations proxy réalisées.
Libération de la VRAM Proxy...

Entraînement Final (100 epochs)...
--> Accuracy Seed 42 : 81.47% | 50,650 paramètres

==================================================
RECHERCHE HYBRIDE - SEED: 43
==================================================
Préchargement du proxy sur GPU...
  Fait en 2.2s

Début Transformer...
Transformer Iter 0: New Best Score 46.50 (Depth: 5)
Transformer Iter 0: New Best Score 47.80 (Depth: 13)
Transformer Iter 0: New Best Score 54.30 (Depth: 7)
Transformer Iter 0: New Best Score 60.50 (Depth: 6)
Transformer Iter 0: New Best Score 61.52 (Depth: 8)
Transformer Iter 5: New Best Score 67.36 (Depth: 12)

Début ABC...
ABC Iter 0: Best Score 68.50
ABC Iter 1: Best Score 70.42
ABC Iter 2: Best Score 70.56
ABC Iter 3: Best Score 70.56
ABC Iter 4: Best Score 70.56
ABC Iter 5: Best Score 70.56
ABC Iter 6: Best Score 70.56
ABC Iter 7: Best Score 70.56
Early stopping déclenché à l'itération 7 : Aucun gain depuis 5 itérations.
Bilan recherche : 685 évaluations proxy réalisées.
Libération de la VRAM Proxy...

Entraînement Final (100 epochs)...
--> Accuracy Seed 43 : 83.82% | 77,002 paramètres

==================================================
RECHERCHE HYBRIDE - SEED: 44
==================================================
Préchargement du proxy sur GPU...
  Fait en 2.0s

Début Transformer...
Transformer Iter 0: New Best Score 36.48 (Depth: 14)
Transformer Iter 0: New Best Score 38.66 (Depth: 15)
Transformer Iter 0: New Best Score 46.54 (Depth: 12)
Transformer Iter 0: New Best Score 57.62 (Depth: 8)
Transformer Iter 0: New Best Score 61.60 (Depth: 6)
Transformer Iter 1: New Best Score 64.50 (Depth: 7)

Début ABC...
ABC Iter 0: Best Score 68.02
ABC Iter 1: Best Score 68.02
ABC Iter 2: Best Score 68.02
ABC Iter 3: Best Score 68.02
ABC Iter 4: Best Score 68.02
ABC Iter 5: Best Score 68.18
ABC Iter 6: Best Score 68.26
ABC Iter 7: Best Score 68.26
ABC Iter 8: Best Score 68.66
ABC Iter 9: Best Score 68.66
ABC Iter 10: Best Score 68.66
ABC Iter 11: Best Score 68.66
ABC Iter 12: Best Score 68.66
ABC Iter 13: Best Score 68.66
Early stopping déclenché à l'itération 13 : Aucun gain depuis 5 itérations.
Bilan recherche : 946 évaluations proxy réalisées.
Libération de la VRAM Proxy...

Entraînement Final (100 epochs)...
--> Accuracy Seed 44 : 84.88% | 165,209 paramètres

Moyenne: 83.39% ± 1.75%

[SUCCESS] C:/Users/ramigon/Downloads/8INF976/code/Cifar_hybrid.py finished in 279.67 minutes.

========================================
RUNNING: C:/Users/ramigon/Downloads/8INF976/code/Cifar _SA.py
========================================

Utilisation du device : cuda
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified

==================================================
LANCEMENT SA SEUL - SEED: 42
==================================================
Préchargement du proxy sur GPU...
  Fait en 2.1s

Début de la recherche Recuit Simulé sur proxy CIFAR-10...
iter 1: New Best! Score 56.72
iter 2: New Best! Score 58.08
iter 10: New Best! Score 58.32
iter 14: New Best! Score 60.32
iter 31: New Best! Score 60.90
iter 32: New Best! Score 61.26
iter 34: New Best! Score 62.34
iter 63: New Best! Score 62.44
iter 247: New Best! Score 64.20
iter 248: New Best! Score 65.36
iter 249: New Best! Score 66.92
iter 433: New Best! Score 67.82
iter 438: New Best! Score 68.76
iter 451: New Best! Score 70.02
iter 522: New Best! Score 71.22
iter 532: New Best! Score 71.50
iter 536: New Best! Score 71.72
iter 537: New Best! Score 73.36
iter 596: New Best! Score 73.74
iter 600: New Best! Score 74.04
Bilan recherche : 801 évaluations proxy réalisées.
Libération de la VRAM Proxy...

Entraînement Final (100 Epochs)...
--> Accuracy finale Seed 42 : 85.82% | 288,047 paramètres

==================================================
LANCEMENT SA SEUL - SEED: 43
==================================================
Préchargement du proxy sur GPU...
  Fait en 2.0s

Début de la recherche Recuit Simulé sur proxy CIFAR-10...
iter 45: New Best! Score 56.74
iter 50: New Best! Score 59.28
iter 51: New Best! Score 59.38
iter 55: New Best! Score 62.16
iter 56: New Best! Score 63.94
iter 421: New Best! Score 68.74
iter 432: New Best! Score 70.70
iter 433: New Best! Score 72.20
iter 480: New Best! Score 72.34
iter 487: New Best! Score 72.50
iter 489: New Best! Score 73.86
iter 512: New Best! Score 74.82
iter 523: New Best! Score 75.34
iter 542: New Best! Score 75.46
iter 547: New Best! Score 75.58
iter 555: New Best! Score 75.62
Bilan recherche : 801 évaluations proxy réalisées.
Libération de la VRAM Proxy...

Entraînement Final (100 Epochs)...
--> Accuracy finale Seed 43 : 88.98% | 359,629 paramètres

==================================================
LANCEMENT SA SEUL - SEED: 44
==================================================
Préchargement du proxy sur GPU...
  Fait en 2.0s

Début de la recherche Recuit Simulé sur proxy CIFAR-10...
iter 0: New Best! Score 56.12
iter 5: New Best! Score 58.92
iter 7: New Best! Score 59.08
iter 16: New Best! Score 60.10
iter 60: New Best! Score 70.28
iter 525: New Best! Score 71.80
iter 529: New Best! Score 72.12
iter 577: New Best! Score 72.16
iter 584: New Best! Score 72.52
iter 626: New Best! Score 72.84
iter 642: New Best! Score 73.22
iter 646: New Best! Score 73.28
iter 660: New Best! Score 73.36
iter 674: New Best! Score 73.88
iter 693: New Best! Score 74.34
iter 781: New Best! Score 74.52
Bilan recherche : 801 évaluations proxy réalisées.
Libération de la VRAM Proxy...

Entraînement Final (100 Epochs)...
--> Accuracy finale Seed 44 : 86.83% | 236,607 paramètres

Moyenne: 87.21% ± 1.61%

[SUCCESS] C:/Users/ramigon/Downloads/8INF976/code/Cifar _SA.py finished in 362.01 minutes.

========================================
RUNNING: C:/Users/ramigon/Downloads/8INF976/code/Cifar _abc.py
========================================

Utilisation du device : cuda
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified

==================================================
LANCEMENT ABC SEUL - SEED: 42
==================================================
Préchargement du proxy sur GPU...
  Fait en 2.3s

Début de la recherche ABC sur proxy CIFAR-10...
ABC Iter 0: Best Score 60.94
ABC Iter 1: Best Score 62.80
ABC Iter 2: Best Score 64.88
ABC Iter 3: Best Score 64.88
ABC Iter 4: Best Score 64.88
ABC Iter 5: Best Score 64.88
ABC Iter 6: Best Score 64.88
ABC Iter 7: Best Score 64.88
Early stopping déclenché à l'itération 7 : Aucun gain depuis 5 itérations.
Bilan recherche : 357 évaluations proxy réalisées.
Libération de la VRAM Proxy...

Entraînement Final (100 epochs)...
--> Accuracy finale Seed 42 : 77.70% | 184,040 paramètres

==================================================
LANCEMENT ABC SEUL - SEED: 43
==================================================
Préchargement du proxy sur GPU...
  Fait en 2.0s

Début de la recherche ABC sur proxy CIFAR-10...
ABC Iter 0: Best Score 60.08
ABC Iter 1: Best Score 60.08
ABC Iter 2: Best Score 61.00
ABC Iter 3: Best Score 62.34
ABC Iter 4: Best Score 62.94
ABC Iter 5: Best Score 67.00
ABC Iter 6: Best Score 67.00
ABC Iter 7: Best Score 67.48
ABC Iter 8: Best Score 67.48
ABC Iter 9: Best Score 67.48
ABC Iter 10: Best Score 67.48
ABC Iter 11: Best Score 67.48
ABC Iter 12: Best Score 67.48
Early stopping déclenché à l'itération 12 : Aucun gain depuis 5 itérations.
Bilan recherche : 573 évaluations proxy réalisées.
Libération de la VRAM Proxy...

Entraînement Final (100 epochs)...
--> Accuracy finale Seed 43 : 76.27% | 421,335 paramètres

==================================================
LANCEMENT ABC SEUL - SEED: 44
==================================================
Préchargement du proxy sur GPU...
  Fait en 2.0s

Début de la recherche ABC sur proxy CIFAR-10...
ABC Iter 0: Best Score 62.84
ABC Iter 1: Best Score 63.68
ABC Iter 2: Best Score 63.68
ABC Iter 3: Best Score 63.68
ABC Iter 4: Best Score 63.68
ABC Iter 5: Best Score 63.68
ABC Iter 6: Best Score 63.68
Early stopping déclenché à l'itération 6 : Aucun gain depuis 5 itérations.
Bilan recherche : 315 évaluations proxy réalisées.
Libération de la VRAM Proxy...

Entraînement Final (100 epochs)...
--> Accuracy finale Seed 44 : 74.61% | 138,413 paramètres

Moyenne: 76.19% ± 1.55%

[SUCCESS] C:/Users/ramigon/Downloads/8INF976/code/Cifar _abc.py finished in 70.87 minutes.

========================================
RUNNING: C:/Users/ramigon/Downloads/8INF976/code/benchmark_technique.py
========================================


==================================================================================================================================
REAL DATASETS BENCHMARK (Runs: 5 | Iterations: 40) | DEVICE: cuda
==================================================================================================================================

>>> TASK: CALIFORNIA_HOUSING
  > Running Simulated Annealing...iter 2: New Best! Score -0.45
iter 4: New Best! Score -0.44
iter 7: New Best! Score -0.43
iter 9: New Best! Score -0.41
iter 0: New Best! Score -0.40
iter 3: New Best! Score -0.39
iter 0: New Best! Score -0.43
iter 1: New Best! Score -0.42
iter 2: New Best! Score -0.38
iter 3: New Best! Score -0.33
iter 0: New Best! Score -0.44
iter 3: New Best! Score -0.44
iter 4: New Best! Score -0.44
iter 5: New Best! Score -0.44
iter 6: New Best! Score -0.40
iter 8: New Best! Score -0.39
iter 9: New Best! Score -0.38
iter 0: New Best! Score -0.41
iter 2: New Best! Score -0.41
iter 4: New Best! Score -0.34
 Done in 166.1s total.
  > Running ABC Algorithm...ABC Iter 0: Best Score -0.37
ABC Iter 1: Best Score -0.35
ABC Iter 2: Best Score -0.35
ABC Iter 3: Best Score -0.35
ABC Iter 4: Best Score -0.35
ABC Iter 5: Best Score -0.35
ABC Iter 6: Best Score -0.33
ABC Iter 7: Best Score -0.33
ABC Iter 8: Best Score -0.33
ABC Iter 9: Best Score -0.33
ABC Iter 10: Best Score -0.31
ABC Iter 11: Best Score -0.31
ABC Iter 12: Best Score -0.31
ABC Iter 13: Best Score -0.31
ABC Iter 14: Best Score -0.31
ABC Iter 15: Best Score -0.31
ABC Iter 16: Best Score -0.31
ABC Iter 17: Best Score -0.31
ABC Iter 18: Best Score -0.31
ABC Iter 19: Best Score -0.31
ABC Iter 20: Best Score -0.31
ABC Iter 21: Best Score -0.31
ABC Iter 22: Best Score -0.31
ABC Iter 23: Best Score -0.31
ABC Iter 24: Best Score -0.31
ABC Iter 25: Best Score -0.31
ABC Iter 26: Best Score -0.31
ABC Iter 27: Best Score -0.31
ABC Iter 28: Best Score -0.31
ABC Iter 29: Best Score -0.31
ABC Iter 30: Best Score -0.31
ABC Iter 31: Best Score -0.31
ABC Iter 32: Best Score -0.31
ABC Iter 33: Best Score -0.30
ABC Iter 34: Best Score -0.30
ABC Iter 35: Best Score -0.30
ABC Iter 36: Best Score -0.30
ABC Iter 37: Best Score -0.30
ABC Iter 38: Best Score -0.30
ABC Iter 39: Best Score -0.30
ABC Iter 0: Best Score -0.35
ABC Iter 1: Best Score -0.35
ABC Iter 2: Best Score -0.35
ABC Iter 3: Best Score -0.35
ABC Iter 4: Best Score -0.35
ABC Iter 5: Best Score -0.35
ABC Iter 6: Best Score -0.35
ABC Iter 7: Best Score -0.35
ABC Iter 8: Best Score -0.34
ABC Iter 9: Best Score -0.34
ABC Iter 10: Best Score -0.34
ABC Iter 11: Best Score -0.34
ABC Iter 12: Best Score -0.32
ABC Iter 13: Best Score -0.32
ABC Iter 14: Best Score -0.32
ABC Iter 15: Best Score -0.32
ABC Iter 16: Best Score -0.32
ABC Iter 17: Best Score -0.32
ABC Iter 18: Best Score -0.32
ABC Iter 19: Best Score -0.32
ABC Iter 20: Best Score -0.32
ABC Iter 21: Best Score -0.32
ABC Iter 22: Best Score -0.32
ABC Iter 23: Best Score -0.32
ABC Iter 24: Best Score -0.32
ABC Iter 25: Best Score -0.32
ABC Iter 26: Best Score -0.32
ABC Iter 27: Best Score -0.32
ABC Iter 28: Best Score -0.32
ABC Iter 29: Best Score -0.32
ABC Iter 30: Best Score -0.32
ABC Iter 31: Best Score -0.32
ABC Iter 32: Best Score -0.32
ABC Iter 33: Best Score -0.32
ABC Iter 34: Best Score -0.32
ABC Iter 35: Best Score -0.32
ABC Iter 36: Best Score -0.32
ABC Iter 37: Best Score -0.32
ABC Iter 38: Best Score -0.32
ABC Iter 39: Best Score -0.32
ABC Iter 0: Best Score -0.38
ABC Iter 1: Best Score -0.37
ABC Iter 2: Best Score -0.35
ABC Iter 3: Best Score -0.35
ABC Iter 4: Best Score -0.34
ABC Iter 5: Best Score -0.34
ABC Iter 6: Best Score -0.34
ABC Iter 7: Best Score -0.34
ABC Iter 8: Best Score -0.34
ABC Iter 9: Best Score -0.34
ABC Iter 10: Best Score -0.34
ABC Iter 11: Best Score -0.34
ABC Iter 12: Best Score -0.34
ABC Iter 13: Best Score -0.34
ABC Iter 14: Best Score -0.34
ABC Iter 15: Best Score -0.33
ABC Iter 16: Best Score -0.33
ABC Iter 17: Best Score -0.33
ABC Iter 18: Best Score -0.33
ABC Iter 19: Best Score -0.33
ABC Iter 20: Best Score -0.33
ABC Iter 21: Best Score -0.33
ABC Iter 22: Best Score -0.33
ABC Iter 23: Best Score -0.33
ABC Iter 24: Best Score -0.33
ABC Iter 25: Best Score -0.33
ABC Iter 26: Best Score -0.33
ABC Iter 27: Best Score -0.33
ABC Iter 28: Best Score -0.33
ABC Iter 29: Best Score -0.33
ABC Iter 30: Best Score -0.33
ABC Iter 31: Best Score -0.33
ABC Iter 32: Best Score -0.33
ABC Iter 33: Best Score -0.33
ABC Iter 34: Best Score -0.33
ABC Iter 35: Best Score -0.33
ABC Iter 36: Best Score -0.33
ABC Iter 37: Best Score -0.33
ABC Iter 38: Best Score -0.33
ABC Iter 39: Best Score -0.33
ABC Iter 0: Best Score -0.42
ABC Iter 1: Best Score -0.37
ABC Iter 2: Best Score -0.35
ABC Iter 3: Best Score -0.34
ABC Iter 4: Best Score -0.34
ABC Iter 5: Best Score -0.34
ABC Iter 6: Best Score -0.34
ABC Iter 7: Best Score -0.33
ABC Iter 8: Best Score -0.33
ABC Iter 9: Best Score -0.33
ABC Iter 10: Best Score -0.32
ABC Iter 11: Best Score -0.32
ABC Iter 12: Best Score -0.32
ABC Iter 13: Best Score -0.32
ABC Iter 14: Best Score -0.32
ABC Iter 15: Best Score -0.32
ABC Iter 16: Best Score -0.31
ABC Iter 17: Best Score -0.31
ABC Iter 18: Best Score -0.31
ABC Iter 19: Best Score -0.31
ABC Iter 20: Best Score -0.31
ABC Iter 21: Best Score -0.31
ABC Iter 22: Best Score -0.31
ABC Iter 23: Best Score -0.31
ABC Iter 24: Best Score -0.31
ABC Iter 25: Best Score -0.31
ABC Iter 26: Best Score -0.31
ABC Iter 27: Best Score -0.31
ABC Iter 28: Best Score -0.31
ABC Iter 29: Best Score -0.31
ABC Iter 30: Best Score -0.31
ABC Iter 31: Best Score -0.31
ABC Iter 32: Best Score -0.31
ABC Iter 33: Best Score -0.31
ABC Iter 34: Best Score -0.31
ABC Iter 35: Best Score -0.31
ABC Iter 36: Best Score -0.31
ABC Iter 37: Best Score -0.31
ABC Iter 38: Best Score -0.31
ABC Iter 39: Best Score -0.31
ABC Iter 0: Best Score -0.42
ABC Iter 1: Best Score -0.40
ABC Iter 2: Best Score -0.36
ABC Iter 3: Best Score -0.36
ABC Iter 4: Best Score -0.34
ABC Iter 5: Best Score -0.34
ABC Iter 6: Best Score -0.34
ABC Iter 7: Best Score -0.32
ABC Iter 8: Best Score -0.31
ABC Iter 9: Best Score -0.31
ABC Iter 10: Best Score -0.31
ABC Iter 11: Best Score -0.31
ABC Iter 12: Best Score -0.31
ABC Iter 13: Best Score -0.31
ABC Iter 14: Best Score -0.31
ABC Iter 15: Best Score -0.31
ABC Iter 16: Best Score -0.31
ABC Iter 17: Best Score -0.31
ABC Iter 18: Best Score -0.31
ABC Iter 19: Best Score -0.31
ABC Iter 20: Best Score -0.31
ABC Iter 21: Best Score -0.31
ABC Iter 22: Best Score -0.31
ABC Iter 23: Best Score -0.31
ABC Iter 24: Best Score -0.31
ABC Iter 25: Best Score -0.31
ABC Iter 26: Best Score -0.31
ABC Iter 27: Best Score -0.31
ABC Iter 28: Best Score -0.31
ABC Iter 29: Best Score -0.31
ABC Iter 30: Best Score -0.31
ABC Iter 31: Best Score -0.31
ABC Iter 32: Best Score -0.31
ABC Iter 33: Best Score -0.31
ABC Iter 34: Best Score -0.31
ABC Iter 35: Best Score -0.31
ABC Iter 36: Best Score -0.31
ABC Iter 37: Best Score -0.31
ABC Iter 38: Best Score -0.31
ABC Iter 39: Best Score -0.31
 Done in 7670.2s total.
  > Running Transformer...C:\Users\ramigon\.conda\envs\pytorch_gpu\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Transformer Iter 0: New Best Score -0.57 (Depth: 9)
Transformer Iter 0: New Best Score -0.44 (Depth: 18)
Transformer Iter 1: New Best Score -0.43 (Depth: 4)
Transformer Iter 1: New Best Score -0.36 (Depth: 4)
Transformer Iter 2: New Best Score -0.34 (Depth: 6)
Transformer Iter 2: New Best Score -0.34 (Depth: 4)
Transformer Iter 3: New Best Score -0.33 (Depth: 11)
Transformer Iter 4: New Best Score -0.31 (Depth: 7)
Transformer Iter 4: New Best Score -0.30 (Depth: 13)
Transformer Iter 26: New Best Score -0.30 (Depth: 6)
C:\Users\ramigon\.conda\envs\pytorch_gpu\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Transformer Iter 0: New Best Score -0.94 (Depth: 13)
Transformer Iter 0: New Best Score -0.46 (Depth: 6)
Transformer Iter 0: New Best Score -0.38 (Depth: 4)
Transformer Iter 1: New Best Score -0.33 (Depth: 9)
Transformer Iter 10: New Best Score -0.33 (Depth: 26)
C:\Users\ramigon\.conda\envs\pytorch_gpu\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Transformer Iter 0: New Best Score -1.95 (Depth: 2)
Transformer Iter 0: New Best Score -1.73 (Depth: 3)
Transformer Iter 0: New Best Score -0.36 (Depth: 4)
Transformer Iter 9: New Best Score -0.36 (Depth: 15)
C:\Users\ramigon\.conda\envs\pytorch_gpu\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Transformer Iter 0: New Best Score -0.39 (Depth: 7)
C:\Users\ramigon\.conda\envs\pytorch_gpu\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Transformer Iter 0: New Best Score -2.38 (Depth: 3)
Transformer Iter 0: New Best Score -0.38 (Depth: 3)
Transformer Iter 2: New Best Score -0.35 (Depth: 20)
Transformer Iter 5: New Best Score -0.32 (Depth: 43)
Transformer Iter 14: New Best Score -0.32 (Depth: 13)
Transformer Iter 18: New Best Score -0.31 (Depth: 9)
 Done in 6911.9s total.

>>> TASK: BREAST_CANCER
  > Running Simulated Annealing...iter 0: New Best! Score 93.41
iter 2: New Best! Score 95.60
iter 25: New Best! Score 96.70
iter 25: New Best! Score 98.90
iter 13: New Best! Score 92.31
iter 14: New Best! Score 93.41
iter 18: New Best! Score 96.70
iter 29: New Best! Score 97.80
iter 0: New Best! Score 92.31
iter 2: New Best! Score 93.41
iter 7: New Best! Score 96.70
iter 30: New Best! Score 97.80
iter 0: New Best! Score 93.41
iter 1: New Best! Score 97.80
 Done in 5.9s total.
  > Running ABC Algorithm...ABC Iter 0: Best Score 97.80
ABC Iter 1: Best Score 97.80
ABC Iter 2: Best Score 97.80
ABC Iter 3: Best Score 97.80
ABC Iter 4: Best Score 97.80
ABC Iter 5: Best Score 97.80
ABC Iter 6: Best Score 97.80
ABC Iter 7: Best Score 97.80
ABC Iter 8: Best Score 98.90
ABC Iter 9: Best Score 98.90
ABC Iter 10: Best Score 98.90
ABC Iter 11: Best Score 98.90
ABC Iter 12: Best Score 98.90
ABC Iter 13: Best Score 98.90
ABC Iter 14: Best Score 98.90
ABC Iter 15: Best Score 98.90
ABC Iter 16: Best Score 98.90
ABC Iter 17: Best Score 98.90
ABC Iter 18: Best Score 98.90
ABC Iter 19: Best Score 98.90
ABC Iter 20: Best Score 98.90
ABC Iter 21: Best Score 98.90
ABC Iter 22: Best Score 98.90
ABC Iter 23: Best Score 98.90
ABC Iter 24: Best Score 98.90
ABC Iter 25: Best Score 98.90
ABC Iter 26: Best Score 98.90
ABC Iter 27: Best Score 98.90
ABC Iter 28: Best Score 98.90
ABC Iter 29: Best Score 98.90
ABC Iter 30: Best Score 98.90
ABC Iter 31: Best Score 98.90
ABC Iter 32: Best Score 98.90
ABC Iter 33: Best Score 98.90
ABC Iter 34: Best Score 98.90
ABC Iter 35: Best Score 98.90
ABC Iter 36: Best Score 98.90
ABC Iter 37: Best Score 98.90
ABC Iter 38: Best Score 98.90
ABC Iter 39: Best Score 98.90
ABC Iter 0: Best Score 97.80
ABC Iter 1: Best Score 97.80
ABC Iter 2: Best Score 97.80
ABC Iter 3: Best Score 98.90
ABC Iter 4: Best Score 98.90
ABC Iter 5: Best Score 98.90
ABC Iter 6: Best Score 98.90
ABC Iter 7: Best Score 98.90
ABC Iter 8: Best Score 98.90
ABC Iter 9: Best Score 98.90
ABC Iter 10: Best Score 98.90
ABC Iter 11: Best Score 98.90
ABC Iter 12: Best Score 98.90
ABC Iter 13: Best Score 98.90
ABC Iter 14: Best Score 98.90
ABC Iter 15: Best Score 98.90
ABC Iter 16: Best Score 98.90
ABC Iter 17: Best Score 98.90
ABC Iter 18: Best Score 98.90
ABC Iter 19: Best Score 98.90
ABC Iter 20: Best Score 98.90
ABC Iter 21: Best Score 98.90
ABC Iter 22: Best Score 98.90
ABC Iter 23: Best Score 98.90
ABC Iter 24: Best Score 98.90
ABC Iter 25: Best Score 98.90
ABC Iter 26: Best Score 98.90
ABC Iter 27: Best Score 98.90
ABC Iter 28: Best Score 98.90
ABC Iter 29: Best Score 98.90
ABC Iter 30: Best Score 98.90
ABC Iter 31: Best Score 98.90
ABC Iter 32: Best Score 98.90
ABC Iter 33: Best Score 98.90
ABC Iter 34: Best Score 98.90
ABC Iter 35: Best Score 98.90
ABC Iter 36: Best Score 98.90
ABC Iter 37: Best Score 98.90
ABC Iter 38: Best Score 98.90
ABC Iter 39: Best Score 98.90
ABC Iter 0: Best Score 98.90
ABC Iter 1: Best Score 98.90
ABC Iter 2: Best Score 100.00
ABC Iter 3: Best Score 100.00
ABC Iter 4: Best Score 100.00
ABC Iter 5: Best Score 100.00
ABC Iter 6: Best Score 100.00
ABC Iter 7: Best Score 100.00
ABC Iter 8: Best Score 100.00
ABC Iter 9: Best Score 100.00
ABC Iter 10: Best Score 100.00
ABC Iter 11: Best Score 100.00
ABC Iter 12: Best Score 100.00
ABC Iter 13: Best Score 100.00
ABC Iter 14: Best Score 100.00
ABC Iter 15: Best Score 100.00
ABC Iter 16: Best Score 100.00
ABC Iter 17: Best Score 100.00
ABC Iter 18: Best Score 100.00
ABC Iter 19: Best Score 100.00
ABC Iter 20: Best Score 100.00
ABC Iter 21: Best Score 100.00
ABC Iter 22: Best Score 100.00
ABC Iter 23: Best Score 100.00
ABC Iter 24: Best Score 100.00
ABC Iter 25: Best Score 100.00
ABC Iter 26: Best Score 100.00
ABC Iter 27: Best Score 100.00
ABC Iter 28: Best Score 100.00
ABC Iter 29: Best Score 100.00
ABC Iter 30: Best Score 100.00
ABC Iter 31: Best Score 100.00
ABC Iter 32: Best Score 100.00
ABC Iter 33: Best Score 100.00
ABC Iter 34: Best Score 100.00
ABC Iter 35: Best Score 100.00
ABC Iter 36: Best Score 100.00
ABC Iter 37: Best Score 100.00
ABC Iter 38: Best Score 100.00
ABC Iter 39: Best Score 100.00
ABC Iter 0: Best Score 97.80
ABC Iter 1: Best Score 97.80
ABC Iter 2: Best Score 97.80
ABC Iter 3: Best Score 97.80
ABC Iter 4: Best Score 97.80
ABC Iter 5: Best Score 97.80
ABC Iter 6: Best Score 97.80
ABC Iter 7: Best Score 97.80
ABC Iter 8: Best Score 97.80
ABC Iter 9: Best Score 97.80
ABC Iter 10: Best Score 97.80
ABC Iter 11: Best Score 97.80
ABC Iter 12: Best Score 97.80
ABC Iter 13: Best Score 97.80
ABC Iter 14: Best Score 97.80
ABC Iter 15: Best Score 97.80
ABC Iter 16: Best Score 97.80
ABC Iter 17: Best Score 97.80
ABC Iter 18: Best Score 97.80
ABC Iter 19: Best Score 97.80
ABC Iter 20: Best Score 97.80
ABC Iter 21: Best Score 97.80
ABC Iter 22: Best Score 97.80
ABC Iter 23: Best Score 97.80
ABC Iter 24: Best Score 97.80
ABC Iter 25: Best Score 97.80
ABC Iter 26: Best Score 97.80
ABC Iter 27: Best Score 97.80
ABC Iter 28: Best Score 97.80
ABC Iter 29: Best Score 97.80
ABC Iter 30: Best Score 97.80
ABC Iter 31: Best Score 97.80
ABC Iter 32: Best Score 97.80
ABC Iter 33: Best Score 97.80
ABC Iter 34: Best Score 97.80
ABC Iter 35: Best Score 97.80
ABC Iter 36: Best Score 97.80
ABC Iter 37: Best Score 97.80
ABC Iter 38: Best Score 97.80
ABC Iter 39: Best Score 97.80
ABC Iter 0: Best Score 96.70
ABC Iter 1: Best Score 96.70
ABC Iter 2: Best Score 96.70
ABC Iter 3: Best Score 96.70
ABC Iter 4: Best Score 96.70
ABC Iter 5: Best Score 96.70
ABC Iter 6: Best Score 96.70
ABC Iter 7: Best Score 96.70
ABC Iter 8: Best Score 96.70
ABC Iter 9: Best Score 98.90
ABC Iter 10: Best Score 98.90
ABC Iter 11: Best Score 98.90
ABC Iter 12: Best Score 98.90
ABC Iter 13: Best Score 98.90
ABC Iter 14: Best Score 98.90
ABC Iter 15: Best Score 98.90
ABC Iter 16: Best Score 98.90
ABC Iter 17: Best Score 98.90
ABC Iter 18: Best Score 98.90
ABC Iter 19: Best Score 98.90
ABC Iter 20: Best Score 98.90
ABC Iter 21: Best Score 98.90
ABC Iter 22: Best Score 98.90
ABC Iter 23: Best Score 98.90
ABC Iter 24: Best Score 98.90
ABC Iter 25: Best Score 98.90
ABC Iter 26: Best Score 98.90
ABC Iter 27: Best Score 98.90
ABC Iter 28: Best Score 98.90
ABC Iter 29: Best Score 98.90
ABC Iter 30: Best Score 98.90
ABC Iter 31: Best Score 98.90
ABC Iter 32: Best Score 98.90
ABC Iter 33: Best Score 98.90
ABC Iter 34: Best Score 98.90
ABC Iter 35: Best Score 98.90
ABC Iter 36: Best Score 98.90
ABC Iter 37: Best Score 98.90
ABC Iter 38: Best Score 98.90
ABC Iter 39: Best Score 98.90
 Done in 233.7s total.
  > Running Transformer...C:\Users\ramigon\.conda\envs\pytorch_gpu\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Transformer Iter 0: New Best Score 91.21 (Depth: 3)
Transformer Iter 0: New Best Score 94.51 (Depth: 15)
Transformer Iter 1: New Best Score 95.60 (Depth: 5)
Transformer Iter 3: New Best Score 97.80 (Depth: 13)
Transformer Iter 4: New Best Score 98.90 (Depth: 4)
Transformer Iter 7: New Best Score 100.00 (Depth: 15)
C:\Users\ramigon\.conda\envs\pytorch_gpu\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Transformer Iter 0: New Best Score 63.74 (Depth: 23)
Transformer Iter 0: New Best Score 92.31 (Depth: 2)
Transformer Iter 3: New Best Score 93.41 (Depth: 3)
Transformer Iter 5: New Best Score 94.51 (Depth: 3)
Transformer Iter 8: New Best Score 95.60 (Depth: 2)
Transformer Iter 27: New Best Score 96.70 (Depth: 6)
Transformer Iter 29: New Best Score 97.80 (Depth: 3)
C:\Users\ramigon\.conda\envs\pytorch_gpu\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Transformer Iter 0: New Best Score 94.51 (Depth: 6)
Transformer Iter 0: New Best Score 96.70 (Depth: 3)
Transformer Iter 7: New Best Score 97.80 (Depth: 8)
Transformer Iter 9: New Best Score 98.90 (Depth: 7)
C:\Users\ramigon\.conda\envs\pytorch_gpu\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Transformer Iter 0: New Best Score 84.62 (Depth: 20)
Transformer Iter 0: New Best Score 97.80 (Depth: 8)
Transformer Iter 2: New Best Score 98.90 (Depth: 11)
Transformer Iter 2: New Best Score 100.00 (Depth: 9)
C:\Users\ramigon\.conda\envs\pytorch_gpu\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Transformer Iter 0: New Best Score 98.90 (Depth: 8)
 Done in 424.0s total.

>>> TASK: FASHION_MNIST_SIMPLE
  > Running Simulated Annealing...Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ./data\FashionMNIST\raw\train-images-idx3-ubyte.gz
100.0%
Extracting ./data\FashionMNIST\raw\train-images-idx3-ubyte.gz to ./data\FashionMNIST\raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ./data\FashionMNIST\raw\train-labels-idx1-ubyte.gz
100.0%
Extracting ./data\FashionMNIST\raw\train-labels-idx1-ubyte.gz to ./data\FashionMNIST\raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ./data\FashionMNIST\raw\t10k-images-idx3-ubyte.gz
100.0%
Extracting ./data\FashionMNIST\raw\t10k-images-idx3-ubyte.gz to ./data\FashionMNIST\raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ./data\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz
100.0%
Extracting ./data\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz to ./data\FashionMNIST\raw

iter 0: New Best! Score 83.75
iter 0: New Best! Score 83.75
iter 0: New Best! Score 82.50
iter 17: New Best! Score 83.00
iter 23: New Best! Score 83.25
iter 39: New Best! Score 83.50
iter 0: New Best! Score 84.00
iter 6: New Best! Score 85.25
iter 4: New Best! Score 83.50
 Done in 36.6s total.
  > Running ABC Algorithm...ABC Iter 0: Best Score 86.75
ABC Iter 1: Best Score 86.75
ABC Iter 2: Best Score 86.75
ABC Iter 3: Best Score 86.75
ABC Iter 4: Best Score 86.75
ABC Iter 5: Best Score 86.75
ABC Iter 6: Best Score 86.75
ABC Iter 7: Best Score 86.75
ABC Iter 8: Best Score 86.75
ABC Iter 9: Best Score 86.75
ABC Iter 10: Best Score 86.75
ABC Iter 11: Best Score 86.75
ABC Iter 12: Best Score 86.75
ABC Iter 13: Best Score 86.75
ABC Iter 14: Best Score 86.75
ABC Iter 15: Best Score 87.25
ABC Iter 16: Best Score 87.25
ABC Iter 17: Best Score 87.25
ABC Iter 18: Best Score 87.25
ABC Iter 19: Best Score 87.25
ABC Iter 20: Best Score 87.25
ABC Iter 21: Best Score 87.25
ABC Iter 22: Best Score 87.25
ABC Iter 23: Best Score 87.25
ABC Iter 24: Best Score 87.25
ABC Iter 25: Best Score 87.25
ABC Iter 26: Best Score 87.25
ABC Iter 27: Best Score 87.25
ABC Iter 28: Best Score 87.25
ABC Iter 29: Best Score 87.25
ABC Iter 30: Best Score 87.25
ABC Iter 31: Best Score 87.25
ABC Iter 32: Best Score 87.25
ABC Iter 33: Best Score 87.25
ABC Iter 34: Best Score 87.25
ABC Iter 35: Best Score 87.25
ABC Iter 36: Best Score 87.25
ABC Iter 37: Best Score 87.25
ABC Iter 38: Best Score 87.25
ABC Iter 39: Best Score 87.25
ABC Iter 0: Best Score 86.50
ABC Iter 1: Best Score 86.50
ABC Iter 2: Best Score 86.50
ABC Iter 3: Best Score 86.75
ABC Iter 4: Best Score 86.75
ABC Iter 5: Best Score 86.75
ABC Iter 6: Best Score 87.00
ABC Iter 7: Best Score 87.00
ABC Iter 8: Best Score 87.00
ABC Iter 9: Best Score 87.00
ABC Iter 10: Best Score 87.00
ABC Iter 11: Best Score 87.00
ABC Iter 12: Best Score 87.00
ABC Iter 13: Best Score 87.00
ABC Iter 14: Best Score 87.00
ABC Iter 15: Best Score 87.00
ABC Iter 16: Best Score 87.00
ABC Iter 17: Best Score 87.00
ABC Iter 18: Best Score 87.00
ABC Iter 19: Best Score 87.00
ABC Iter 20: Best Score 87.25
ABC Iter 21: Best Score 87.50
ABC Iter 22: Best Score 87.50
ABC Iter 23: Best Score 87.50
ABC Iter 24: Best Score 87.50
ABC Iter 25: Best Score 87.50
ABC Iter 26: Best Score 87.50
ABC Iter 27: Best Score 87.50
ABC Iter 28: Best Score 87.50
ABC Iter 29: Best Score 87.50
ABC Iter 30: Best Score 87.50
ABC Iter 31: Best Score 87.50
ABC Iter 32: Best Score 87.50
ABC Iter 33: Best Score 87.50
ABC Iter 34: Best Score 88.75
ABC Iter 35: Best Score 88.75
ABC Iter 36: Best Score 88.75
ABC Iter 37: Best Score 88.75
ABC Iter 38: Best Score 88.75
ABC Iter 39: Best Score 88.75
ABC Iter 0: Best Score 85.75
ABC Iter 1: Best Score 86.00
ABC Iter 2: Best Score 86.25
ABC Iter 3: Best Score 86.75
ABC Iter 4: Best Score 86.75
ABC Iter 5: Best Score 86.75
ABC Iter 6: Best Score 86.75
ABC Iter 7: Best Score 86.75
ABC Iter 8: Best Score 86.75
ABC Iter 9: Best Score 87.00
ABC Iter 10: Best Score 87.00
ABC Iter 11: Best Score 87.00
ABC Iter 12: Best Score 87.00
ABC Iter 13: Best Score 87.00
ABC Iter 14: Best Score 87.00
ABC Iter 15: Best Score 87.00
ABC Iter 16: Best Score 87.00
ABC Iter 17: Best Score 87.00
ABC Iter 18: Best Score 87.00
ABC Iter 19: Best Score 87.00
ABC Iter 20: Best Score 87.00
ABC Iter 21: Best Score 87.00
ABC Iter 22: Best Score 87.00
ABC Iter 23: Best Score 87.00
ABC Iter 24: Best Score 87.00
ABC Iter 25: Best Score 87.25
ABC Iter 26: Best Score 87.25
ABC Iter 27: Best Score 87.25
ABC Iter 28: Best Score 87.25
ABC Iter 29: Best Score 87.25
ABC Iter 30: Best Score 87.25
ABC Iter 31: Best Score 87.25
ABC Iter 32: Best Score 87.25
ABC Iter 33: Best Score 87.25
ABC Iter 34: Best Score 87.25
ABC Iter 35: Best Score 88.00
ABC Iter 36: Best Score 88.00
ABC Iter 37: Best Score 88.00
ABC Iter 38: Best Score 88.00
ABC Iter 39: Best Score 88.00
ABC Iter 0: Best Score 86.00
ABC Iter 1: Best Score 86.00
ABC Iter 2: Best Score 86.75
ABC Iter 3: Best Score 86.75
ABC Iter 4: Best Score 86.75
ABC Iter 5: Best Score 86.75
ABC Iter 6: Best Score 87.25
ABC Iter 7: Best Score 87.25
ABC Iter 8: Best Score 87.25
ABC Iter 9: Best Score 87.25
ABC Iter 10: Best Score 87.25
ABC Iter 11: Best Score 87.25
ABC Iter 12: Best Score 87.25
ABC Iter 13: Best Score 87.25
ABC Iter 14: Best Score 87.25
ABC Iter 15: Best Score 87.25
ABC Iter 16: Best Score 87.25
ABC Iter 17: Best Score 87.25
ABC Iter 18: Best Score 87.25
ABC Iter 19: Best Score 87.25
ABC Iter 20: Best Score 87.25
ABC Iter 21: Best Score 87.25
ABC Iter 22: Best Score 87.25
ABC Iter 23: Best Score 87.25
ABC Iter 24: Best Score 87.25
ABC Iter 25: Best Score 87.25
ABC Iter 26: Best Score 87.25
ABC Iter 27: Best Score 87.25
ABC Iter 28: Best Score 87.25
ABC Iter 29: Best Score 87.25
ABC Iter 30: Best Score 87.25
ABC Iter 31: Best Score 87.25
ABC Iter 32: Best Score 87.25
ABC Iter 33: Best Score 87.25
ABC Iter 34: Best Score 87.25
ABC Iter 35: Best Score 87.25
ABC Iter 36: Best Score 87.25
ABC Iter 37: Best Score 87.25
ABC Iter 38: Best Score 87.25
ABC Iter 39: Best Score 87.25
ABC Iter 0: Best Score 83.00
ABC Iter 1: Best Score 84.00
ABC Iter 2: Best Score 84.00
ABC Iter 3: Best Score 84.00
ABC Iter 4: Best Score 84.00
ABC Iter 5: Best Score 84.00
ABC Iter 6: Best Score 84.00
ABC Iter 7: Best Score 84.00
ABC Iter 8: Best Score 84.00
ABC Iter 9: Best Score 84.00
ABC Iter 10: Best Score 84.00
ABC Iter 11: Best Score 84.00
ABC Iter 12: Best Score 84.00
ABC Iter 13: Best Score 84.25
ABC Iter 14: Best Score 84.25
ABC Iter 15: Best Score 84.25
ABC Iter 16: Best Score 84.25
ABC Iter 17: Best Score 84.25
ABC Iter 18: Best Score 84.25
ABC Iter 19: Best Score 84.25
ABC Iter 20: Best Score 84.25
ABC Iter 21: Best Score 84.25
ABC Iter 22: Best Score 84.25
ABC Iter 23: Best Score 84.25
ABC Iter 24: Best Score 84.25
ABC Iter 25: Best Score 85.00
ABC Iter 26: Best Score 85.00
ABC Iter 27: Best Score 85.00
ABC Iter 28: Best Score 85.00
ABC Iter 29: Best Score 85.00
ABC Iter 30: Best Score 85.00
ABC Iter 31: Best Score 85.00
ABC Iter 32: Best Score 85.00
ABC Iter 33: Best Score 85.00
ABC Iter 34: Best Score 85.00
ABC Iter 35: Best Score 85.00
ABC Iter 36: Best Score 85.00
ABC Iter 37: Best Score 85.00
ABC Iter 38: Best Score 85.00
ABC Iter 39: Best Score 85.00
 Done in 1393.4s total.
  > Running Transformer...Transformer Iter 0: New Best Score 68.25 (Depth: 13)
Transformer Iter 0: New Best Score 76.25 (Depth: 20)
Transformer Iter 0: New Best Score 77.50 (Depth: 4)
Transformer Iter 0: New Best Score 81.00 (Depth: 4)
Transformer Iter 0: New Best Score 81.25 (Depth: 7)
Transformer Iter 2: New Best Score 81.50 (Depth: 8)
Transformer Iter 3: New Best Score 81.75 (Depth: 4)
Transformer Iter 3: New Best Score 83.75 (Depth: 7)
Transformer Iter 4: New Best Score 84.75 (Depth: 7)
Transformer Iter 15: New Best Score 85.50 (Depth: 6)
Transformer Iter 0: New Best Score 80.00 (Depth: 2)
Transformer Iter 0: New Best Score 80.25 (Depth: 6)
Transformer Iter 0: New Best Score 82.00 (Depth: 4)
Transformer Iter 0: New Best Score 82.50 (Depth: 4)
Transformer Iter 1: New Best Score 82.75 (Depth: 11)
Transformer Iter 1: New Best Score 83.75 (Depth: 3)
Transformer Iter 3: New Best Score 84.00 (Depth: 4)
Transformer Iter 6: New Best Score 84.75 (Depth: 8)
Transformer Iter 8: New Best Score 85.50 (Depth: 5)
Transformer Iter 39: New Best Score 86.50 (Depth: 7)
Transformer Iter 0: New Best Score 52.50 (Depth: 12)
Transformer Iter 0: New Best Score 71.75 (Depth: 6)
Transformer Iter 0: New Best Score 72.50 (Depth: 16)
Transformer Iter 0: New Best Score 79.75 (Depth: 3)
Transformer Iter 0: New Best Score 80.75 (Depth: 6)
Transformer Iter 1: New Best Score 82.50 (Depth: 6)
Transformer Iter 4: New Best Score 84.25 (Depth: 6)
Transformer Iter 0: New Best Score 58.50 (Depth: 9)
Transformer Iter 0: New Best Score 78.75 (Depth: 4)
Transformer Iter 0: New Best Score 83.50 (Depth: 3)
Transformer Iter 2: New Best Score 84.50 (Depth: 3)
Transformer Iter 2: New Best Score 84.75 (Depth: 5)
Transformer Iter 3: New Best Score 85.00 (Depth: 5)
Transformer Iter 10: New Best Score 85.50 (Depth: 3)
Transformer Iter 10: New Best Score 85.75 (Depth: 3)
Transformer Iter 11: New Best Score 86.00 (Depth: 3)
Transformer Iter 14: New Best Score 86.25 (Depth: 3)
Transformer Iter 0: New Best Score 61.25 (Depth: 12)
Transformer Iter 0: New Best Score 78.25 (Depth: 7)
Transformer Iter 0: New Best Score 80.25 (Depth: 4)
Transformer Iter 0: New Best Score 81.00 (Depth: 3)
Transformer Iter 1: New Best Score 81.50 (Depth: 8)
Transformer Iter 2: New Best Score 82.00 (Depth: 4)
Transformer Iter 2: New Best Score 82.50 (Depth: 3)
Transformer Iter 3: New Best Score 82.75 (Depth: 3)
Transformer Iter 5: New Best Score 83.50 (Depth: 4)
Transformer Iter 9: New Best Score 84.25 (Depth: 3)
Transformer Iter 12: New Best Score 84.50 (Depth: 3)
Transformer Iter 25: New Best Score 85.25 (Depth: 11)
 Done in 809.1s total.

==================================================================================================================================
TASK                      | ALGORITHM              | PROXY SCORE (Avg±Std) | EVALS  | GAIN     | Δ DEPTH  | TIME(s)
----------------------------------------------------------------------------------------------------------------------------------
california_housing        | Simulated Annealing    | -0.37 ± 0.03         | 41     | 0.08     | +1.0     | 33.22
california_housing        | ABC Algorithm          | -0.31 ± 0.01         | 1736   | 0.14     | +2.6     | 1534.04
california_housing        | Transformer            | -0.34 ± 0.04         | 641    | 0.11     | +9.6     | 1382.39
breast_cancer             | Simulated Annealing    | 97.80 ± 0.78         | 41     | 5.71     | +0.2     | 1.18
breast_cancer             | ABC Algorithm          | 98.90 ± 0.78         | 1788   | 5.49     | +1.0     | 46.73
breast_cancer             | Transformer            | 99.12 ± 0.92         | 641    | 7.69     | +5.4     | 84.81
fashion_mnist_simple      | Simulated Annealing    | 83.95 ± 0.74         | 41     | 2.25     | +0.2     | 7.31
fashion_mnist_simple      | ABC Algorithm          | 87.25 ± 1.40         | 1777   | 3.70     | +1.0     | 278.68
fashion_mnist_simple      | Transformer            | 85.55 ± 0.89         | 641    | 2.75     | +1.6     | 161.83

--> Benchmark sauvegardé dans 'results/academic_benchmark_technique.json' et 'results/academic_benchmark_summary.txt'

[SUCCESS] C:/Users/ramigon/Downloads/8INF976/code/benchmark_technique.py finished in 294.56 minutes.

========================================
RUNNING: C:/Users/ramigon/Downloads/8INF976/code/random_search.py
========================================

cuda
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified

==================================================
RANDOM SEARCH - SEED: 42
==================================================
Préchargement du proxy sur GPU...
  fait en 2.2s | train (20000, 3, 32, 32) | val (5000, 3, 32, 32)
Début Random Search...
[seed 42] budget lu depuis results/academic_results_hybride_42.json : 689 évaluations
C:\Users\ramigon\.conda\envs\pytorch_gpu\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
RS eval 1: nouveau best 53.36 (profondeur 11, 0.4 min)
RS eval 4: nouveau best 62.42 (profondeur 8, 0.4 min)
RS eval 40: nouveau best 65.88 (profondeur 9, 1.5 min)
RS eval 408: nouveau best 69.24 (profondeur 10, 15.5 min)
Bilan recherche : 689 évaluations (27 invalides) en 0.40 h | best proxy = 69.24
Entraînement final (100 Epochs)...
--> Accuracy finale Seed 42 : 84.81% | 68,010 paramètres

==================================================
RANDOM SEARCH - SEED: 43
==================================================
Préchargement du proxy sur GPU...
  fait en 2.1s | train (20000, 3, 32, 32) | val (5000, 3, 32, 32)
Début Random Search...
[seed 43] budget lu depuis results/academic_results_hybride_43.json : 685 évaluations
RS eval 1: nouveau best 49.20 (profondeur 14, 0.1 min)
RS eval 5: nouveau best 57.32 (profondeur 7, 0.4 min)
RS eval 22: nouveau best 58.34 (profondeur 8, 0.8 min)
RS eval 23: nouveau best 58.76 (profondeur 8, 0.9 min)
RS eval 29: nouveau best 61.98 (profondeur 15, 1.1 min)
RS eval 74: nouveau best 62.08 (profondeur 5, 2.8 min)
RS eval 144: nouveau best 63.00 (profondeur 15, 5.6 min)
RS eval 146: nouveau best 66.34 (profondeur 10, 5.7 min)
RS eval 246: nouveau best 66.44 (profondeur 12, 9.1 min)
Bilan recherche : 685 évaluations (23 invalides) en 0.43 h | best proxy = 66.44
Entraînement final (100 Epochs)...
--> Accuracy finale Seed 43 : 82.18% | 171,466 paramètres

==================================================
RANDOM SEARCH - SEED: 44
==================================================
Préchargement du proxy sur GPU...
  fait en 2.0s | train (20000, 3, 32, 32) | val (5000, 3, 32, 32)
Début Random Search...
[seed 44] budget lu depuis results/academic_results_hybride_44.json : 946 évaluations
RS eval 1: nouveau best 39.52 (profondeur 10, 0.0 min)
RS eval 2: nouveau best 52.62 (profondeur 10, 0.1 min)
RS eval 4: nouveau best 60.78 (profondeur 13, 0.4 min)
RS eval 10: nouveau best 60.94 (profondeur 10, 0.6 min)
RS eval 40: nouveau best 64.02 (profondeur 13, 1.2 min)
RS eval 124: nouveau best 66.60 (profondeur 12, 5.9 min)
Bilan recherche : 946 évaluations (38 invalides) en 0.63 h | best proxy = 66.60
Entraînement final (100 Epochs)...
--> Accuracy finale Seed 44 : 83.53% | 161,226 paramètres

Moyenne: 83.51% ± 1.32%

[SUCCESS] C:/Users/ramigon/Downloads/8INF976/code/random_search.py finished in 124.03 minutes.

==================================================
ALL EXPERIMENTS COMPLETED IN 18.86 HOURS
==================================================




## Cifar_hybride avec malus de profondeur et sans early exit

voici les resultats pour cifar hybride sans patience avec malus de profondeur :
Seed 42 : {"accuracy": 0.8328, "evals": 1001, "time": 5715.473071336746, "n_parameters": 176350}
sedd 43 : {"accuracy": 0.8817, "evals": 1003, "time": 15168.46367430687, "n_parameters": 229578}
seed 44 : {"accuracy": 0.8309, "evals": 1002, "time": 13678.43964958191, "n_parameters": 117610}


## Cifar hybride sans malus de profondeur et sans early_exit

Voici les résultats pour cifar hybride sans patience sans malus de profondeur

==================================================

RECHERCHE HYBRIDE - SEED: 42

==================================================

Préchargement du proxy sur GPU...

  Fait en 6.2s



Début Transformer...

C:\APPS\Anaconda3\envs\torchgpu\lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)

  warnings.warn(

Transformer Iter 0: New Best Score 55.52 (Depth: 5)

Transformer Iter 0: New Best Score 56.56 (Depth: 7)

Transformer Iter 0: New Best Score 56.76 (Depth: 14)

Transformer Iter 0: New Best Score 57.32 (Depth: 12)

Transformer Iter 0: New Best Score 58.14 (Depth: 6)

Transformer Iter 2: New Best Score 59.74 (Depth: 9)

Transformer Iter 2: New Best Score 62.00 (Depth: 12)

Transformer Iter 2: New Best Score 62.48 (Depth: 9)

Transformer Iter 2: New Best Score 64.46 (Depth: 10)

Transformer Iter 7: New Best Score 64.84 (Depth: 14)

Transformer Iter 7: New Best Score 69.26 (Depth: 17)



Début ABC...

C:\APPS\Anaconda3\envs\torchgpu\lib\site-packages\torch\nn\init.py:511: UserWarning: Initializing zero-element tensors is a no-op

  warnings.warn("Initializing zero-element tensors is a no-op")

ABC Iter 0: Best Score 70.64

ABC Iter 1: Best Score 70.64

ABC Iter 2: Best Score 70.64

ABC Iter 3: Best Score 70.64

ABC Iter 4: Best Score 70.88

ABC Iter 5: Best Score 70.88

ABC Iter 6: Best Score 71.28

ABC Iter 7: Best Score 71.28

ABC Iter 8: Best Score 71.28

ABC Iter 9: Best Score 71.28

ABC Iter 10: Best Score 71.28

ABC Iter 11: Best Score 71.28

ABC Iter 12: Best Score 71.28

ABC Iter 13: Best Score 71.28

ABC Iter 14: Best Score 71.28

Bilan recherche : 1002 évaluations proxy réalisées.

Libération de la VRAM Proxy...



Entraînement Final (100 epochs)...

--> Accuracy Seed 42 : 87.83% | 281,522 paramètres



==================================================

RECHERCHE HYBRIDE - SEED: 43

==================================================

Préchargement du proxy sur GPU...

  Fait en 2.4s



Début Transformer...

Transformer Iter 0: New Best Score 46.50 (Depth: 5)

Transformer Iter 0: New Best Score 47.80 (Depth: 13)

Transformer Iter 0: New Best Score 53.92 (Depth: 7)

Transformer Iter 0: New Best Score 60.14 (Depth: 6)

Transformer Iter 0: New Best Score 61.30 (Depth: 8)

Transformer Iter 1: New Best Score 61.44 (Depth: 6)

Transformer Iter 2: New Best Score 63.60 (Depth: 8)

Transformer Iter 4: New Best Score 63.66 (Depth: 6)

Transformer Iter 5: New Best Score 67.30 (Depth: 9)

Transformer Iter 5: New Best Score 68.20 (Depth: 9)

Transformer Iter 16: New Best Score 68.36 (Depth: 21)

Transformer Iter 17: New Best Score 68.60 (Depth: 22)

Transformer Iter 18: New Best Score 69.92 (Depth: 20)



Début ABC...

ABC Iter 0: Best Score 70.54

ABC Iter 1: Best Score 70.58

ABC Iter 2: Best Score 71.44

ABC Iter 3: Best Score 71.44

ABC Iter 4: Best Score 71.44

ABC Iter 5: Best Score 71.44

ABC Iter 6: Best Score 71.44

ABC Iter 7: Best Score 71.44

ABC Iter 8: Best Score 71.44

ABC Iter 9: Best Score 71.44

ABC Iter 10: Best Score 71.44

ABC Iter 11: Best Score 71.44

ABC Iter 12: Best Score 71.44

ABC Iter 13: Best Score 71.44

ABC Iter 14: Best Score 71.44

Bilan recherche : 1006 évaluations proxy réalisées.

Libération de la VRAM Proxy...



Entraînement Final (100 epochs)...

--> Accuracy Seed 43 : 87.81% | 189,963 paramètres



==================================================

RECHERCHE HYBRIDE - SEED: 44

==================================================

Préchargement du proxy sur GPU...

  Fait en 2.6s



Début Transformer...

Transformer Iter 0: New Best Score 38.88 (Depth: 14)

Transformer Iter 0: New Best Score 46.06 (Depth: 11)

Transformer Iter 0: New Best Score 60.22 (Depth: 10)

Transformer Iter 0: New Best Score 61.26 (Depth: 5)

Transformer Iter 1: New Best Score 61.28 (Depth: 7)

Transformer Iter 16: New Best Score 62.20 (Depth: 7)

Transformer Iter 16: New Best Score 62.52 (Depth: 9)

Transformer Iter 17: New Best Score 63.40 (Depth: 10)

Transformer Iter 18: New Best Score 63.82 (Depth: 12)

Transformer Iter 19: New Best Score 64.84 (Depth: 9)



Début ABC...

ABC Iter 0: Best Score 67.86

ABC Iter 1: Best Score 67.86

ABC Iter 2: Best Score 67.88

ABC Iter 3: Best Score 68.20

ABC Iter 4: Best Score 68.34

ABC Iter 5: Best Score 68.50

ABC Iter 6: Best Score 69.90

ABC Iter 7: Best Score 70.36

ABC Iter 8: Best Score 70.36

ABC Iter 9: Best Score 70.36

ABC Iter 10: Best Score 70.36

ABC Iter 11: Best Score 70.36

ABC Iter 12: Best Score 70.36

ABC Iter 13: Best Score 70.36

ABC Iter 14: Best Score 70.36

Bilan recherche : 996 évaluations proxy réalisées.

Libération de la VRAM Proxy...



Entraînement Final (100 epochs)...

--> Accuracy Seed 44 : 84.81% | 215,402 paramètres



Moyenne: 86.82% ± 1.74%



{"accuracy": 0.8783, "evals": 1002, "time": 15146.287305355072, "n_parameters": 281522}



{"accuracy": 0.8781, "evals": 1006, "time": 16755.021039009094, "n_parameters": 189963}



{"accuracy": 0.8481, "evals": 996, "time": 10031.58276939392, "n_parameters": 215402}


```latex
\begin{table*}[htbp]
\caption{Comprehensive NAS Performance Comparison on CIFAR-10 (Budget & Ablation)}
\label{tab:comprehensive_comparison}
\begin{center}
\begin{tabular}{lcccc}
\toprule
\textbf{Algorithm Configuration} & \textbf{Evals (Avg)} & \textbf{Accuracy (Avg $\pm$ Std)} & \textbf{Params (Avg)} & \textbf{Time (min)} \\
\midrule
Random Search & $\sim$770 & 83.51\% $\pm$ 1.32\% & $\sim$133k & $\sim$41 \\
Simulated Annealing & $\sim$800 & \textbf{87.21\% $\pm$ 1.61\%} & $\sim$294k & $\sim$121 \\
ABC Only & $\sim$415 & 76.19\% $\pm$ 1.55\% & $\sim$247k & $\sim$24 \\
\midrule
\textbf{Hybrid (Malus + Early Exit)} & $\sim$770 & 83.39\% $\pm$ 1.75\% & \textbf{$\sim$97k} & $\sim$93 \\
\textbf{Hybrid (Malus + Full Budget)} & $\sim$1000 & 84.85\% $\pm$ 2.88\% & $\sim$174k & $\sim$192 \\
\textbf{Hybrid (No Malus + Full Budget)} & $\sim$1000 & 86.82\% $\pm$ 1.74\% & $\sim$229k & $\sim$233 \\
\bottomrule
\end{tabular}
\end{center}
\end{table*}
```


## CIfar 100
voici les resultats pour cifar 100
==================================================
RECHERCHE HYBRIDE CIFAR-100 - SEED: 42
==================================================
Préchargement du proxy sur GPU...
  Fait en 2.9s

Début Transformer...
C:\Users\ramigon\.conda\envs\pytorch_gpu\Lib\site-packages\torch\nn\modules\transformer.py:379: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(
Transformer Iter 0: New Best Score 23.26 (Depth: 7)
Transformer Iter 1: New Best Score 23.36 (Depth: 4)
Transformer Iter 2: New Best Score 24.92 (Depth: 7)
Transformer Iter 2: New Best Score 25.80 (Depth: 5)
Transformer Iter 6: New Best Score 26.36 (Depth: 4)

Début ABC...
ABC Iter 0: Best Score 28.28
ABC Iter 1: Best Score 28.50
ABC Iter 2: Best Score 28.50
ABC Iter 3: Best Score 28.80
ABC Iter 4: Best Score 28.80
ABC Iter 5: Best Score 29.26
ABC Iter 6: Best Score 29.26
ABC Iter 7: Best Score 30.16
ABC Iter 8: Best Score 30.16
ABC Iter 9: Best Score 30.38
ABC Iter 10: Best Score 30.70
ABC Iter 11: Best Score 34.38
ABC Iter 12: Best Score 35.06
ABC Iter 13: Best Score 36.36
ABC Iter 14: Best Score 36.36
Bilan recherche : 992 évaluations proxy réalisées.
Libération de la VRAM Proxy...

Entraînement Final (100 epochs)...
--> Accuracy Seed 42 : 57.58% | 449,333 paramètres

==================================================
RECHERCHE HYBRIDE CIFAR-100 - SEED: 43
==================================================
Préchargement du proxy sur GPU...
  Fait en 2.5s

Début Transformer...
Transformer Iter 0: New Best Score 17.90 (Depth: 5)
Transformer Iter 0: New Best Score 23.08 (Depth: 3)
Transformer Iter 0: New Best Score 24.40 (Depth: 7)
Transformer Iter 1: New Best Score 28.64 (Depth: 12)

Début ABC...
C:\Users\ramigon\.conda\envs\pytorch_gpu\Lib\site-packages\torch\nn\init.py:511: UserWarning: Initializing zero-element tensors is a no-op
  warnings.warn("Initializing zero-element tensors is a no-op")
ABC Iter 0: Best Score 29.52
ABC Iter 1: Best Score 29.52
ABC Iter 2: Best Score 29.52
ABC Iter 3: Best Score 30.30
ABC Iter 4: Best Score 30.30
ABC Iter 5: Best Score 30.30
ABC Iter 6: Best Score 33.82
ABC Iter 7: Best Score 33.90
ABC Iter 8: Best Score 34.52
ABC Iter 9: Best Score 34.52
ABC Iter 10: Best Score 34.52
ABC Iter 11: Best Score 34.52
ABC Iter 12: Best Score 34.52
ABC Iter 13: Best Score 34.52
ABC Iter 14: Best Score 34.52
Bilan recherche : 986 évaluations proxy réalisées.
Libération de la VRAM Proxy...

Entraînement Final (100 epochs)...
--> Accuracy Seed 43 : 52.71% | 232,220 paramètres

==================================================
RECHERCHE HYBRIDE CIFAR-100 - SEED: 44
==================================================
Préchargement du proxy sur GPU...
  Fait en 2.5s

Début Transformer...
Transformer Iter 0: New Best Score 11.70 (Depth: 14)
Transformer Iter 0: New Best Score 14.42 (Depth: 4)
Transformer Iter 0: New Best Score 22.80 (Depth: 9)
Transformer Iter 1: New Best Score 23.52 (Depth: 6)
Transformer Iter 1: New Best Score 27.72 (Depth: 5)
Transformer Iter 18: New Best Score 27.82 (Depth: 11)

Début ABC...
ABC Iter 0: Best Score 29.48
ABC Iter 1: Best Score 30.50
ABC Iter 2: Best Score 30.50
ABC Iter 3: Best Score 30.50
ABC Iter 4: Best Score 30.50
ABC Iter 5: Best Score 31.12
ABC Iter 6: Best Score 31.12
ABC Iter 7: Best Score 31.12
ABC Iter 8: Best Score 31.12
ABC Iter 9: Best Score 31.12
ABC Iter 10: Best Score 31.12
ABC Iter 11: Best Score 31.12
ABC Iter 12: Best Score 31.12
ABC Iter 13: Best Score 31.12
ABC Iter 14: Best Score 31.12
Bilan recherche : 995 évaluations proxy réalisées.
Libération de la VRAM Proxy...

Entraînement Final (100 epochs)...
--> Accuracy Seed 44 : 46.79% | 3,103,278 paramètres

Moyenne CIFAR-100: 52.36% ± 5.40%

{"accuracy": 0.5758, "evals": 992, "time": 2940.474399328232, "n_parameters": 449333}

{"accuracy": 0.5271, "evals": 986, "time": 3173.8185958862305, "n_parameters": 232220}

{"accuracy": 0.4679, "evals": 995, "time": 4008.2481400966644, "n_parameters": 3103278}