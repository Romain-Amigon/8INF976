# Autoregressive Transformer Controllers for Resource-Constrained Neural Architecture Search.

### 1. État de l'Art et Limites Actuelles
La recherche d'architectures neuronales (NAS) vise à automatiser la conception de modèles de Deep Learning. L'état de l'art actuel se divise en plusieurs familles :
* **Le RL classique (Zoph & Le, 2017) :** Utilise des réseaux récurrents (RNN) pour générer des architectures. Bien qu'efficace, la méthode est prohibitive (28 jours sur 800 GPUs pour CIFAR-10) et les RNN peinent à conserver le contexte des architectures profondes.
* **Differentiable NAS (DARTS) / One-Shot :** Plus rapides, mais souvent complexes à stabiliser, sujets à l'effondrement (collapse) et gourmands en VRAM (nécessitent de charger le super-graphe en mémoire).
* **Zero-Cost Proxies :** Ultra-rapides pour filtrer, mais ne construisent pas activement de nouvelles topologies de bout en bout.
* **L'émergence des Transformers en NAS (La limite du pré-entraînement) :** Récemment, la littérature a commencé à explorer l'intégration des modèles de langage (LLMs) dans la recherche d'architectures. Des approches d'avant-garde telles que GPT-NAS utilisent des modèles auto-régressifs pour injecter des connaissances a priori dans l'espace de recherche. Cependant, dans ces méthodes, le Transformer n'est utilisé que comme un opérateur de mutation au sein d'un algorithme évolutionnaire (EA) pour reconstruire des blocs de couches éliminés[cite: 11, 252]. De plus, ces modèles dépendent d'un pré-entraînement massif sur des centaines de milliers d'architectures préexistantes (ex: NAS-Bench-101), ce qui annule l'aspect frugal et limite leur adaptabilité immédiate à des tâches radicalement différentes de la vision par ordinateur (comme les données tabulaires asymétriques). (*GPT-NAS: Evolutionary Neural Architecture Search
with the Generative Pre-Trained Model*)

**Le Gap Scientifique :** Il manque une méthode générative pure (comme le RL) qui soit capable de comprendre intelligemment la structure d'un réseau pour converger rapidement, le tout exécutable sur un matériel standard (ex: simple GPU RTX 3060).

**Notre Positionnement (Le Transformer comme Contrôleur RL Pur) :** Plutôt que d'utiliser le Transformer comme un simple assistant de mutation nécessitant un lourd pré-entraînement, notre framework élève le Transformer au rang de Contrôleur RL autonome. Formé *from scratch* pour chaque nouvelle tâche, il génère les architectures couche par couche, exploitant son mécanisme de *Self-Attention* pour retenir le contexte global de la topologie.

---

### 2. L'Innovation : Le Transformer comme Contrôleur (L'Analogie du Texte)
Ce projet propose de remplacer les contrôleurs RNN traditionnels par un **Transformer auto-régressif**. 

La construction d'un réseau de neurones est mathématiquement similaire à la rédaction d'une phrase. Dans un texte, le choix du prochain mot dépend du contexte global de la phrase. En NAS, le choix de la prochaine couche (ex: `MaxPool`) n'a de sens que si le contrôleur "se souvient" qu'une couche d'extraction (ex: `Conv2D`) a été posée plusieurs étapes auparavant. 

Là où un RNN "oublie" les premières couches à cause de la disparition du gradient, le Transformer utilise son mécanisme de **Self-Attention** pour regarder simultanément l'intégralité des couches (*tokens*) précédentes. Il apprend ainsi une véritable "grammaire architecturale" :
* Il comprend les dépendances à long terme (macro-exploration).
* Il est guidé par une fonction de perte (Loss) multi-objective, incluant une **pénalité de taille** pour éviter les réseaux obèses, et un **bonus d'entropie** pour forcer l'exploration et éviter la convergence prématurée dans des minimums locaux.

---

### 3. Preuves d'Efficacité et Résultats Prometteurs
Les expérimentations menées sur un matériel contraint (processeur standard / 1 seul GPU) prouvent que le Transformer accélère drastiquement la recherche tout en maximisant la performance :

* **Frugalité et Efficience d'échantillonnage (Fashion-MNIST) :** En ne voyant que 3% du jeu de données (2000 images) sur des entraînements de 5 époques, le contrôleur a conçu un réseau atteignant **~86.85% de précision** en quelques minutes/heures, supplantant les métaheuristiques classiques (Recuit Simulé).
* **Capacité à atteindre le plafond (Breast Cancer) :** Le contrôleur trouve des modèles atteignant **~99%**, soit la limite mathématique inhérente à ce dataset bruité.
* **Flexibilité face aux données asymétriques (Credit Card Fraud) :** Sur un problème industriel contenant moins de 1% de fraudes, la fonction de récompense du RL a pu être modifiée pour optimiser le **F1-Score**. Les logs de recherche montrent une convergence fulgurante du Transformer (de 68.11 à 80.26 en 3 itérations). L'architecture générée rivalise avec les algorithmes de Gradient Boosting (XGBoost) en atteignant un **Rappel de 84% et une Précision de 71%** (avec un seuil optimal de 0.98), garantissant un excellent filtrage des fausses alertes.

---

### 4. Contribution Secondaire : L'Hybridation Mémétique
Pour pallier la difficulté inhérente du RL à ajuster finement des hyperparamètres continus (micro-exploitation), l'architecture brute générée par le Transformer est passée à une métaheuristique d'essaim (Artificial Bee Colony - ABC) pour un "Warm-Start". 
Les logs prouvent que cette étape finale permet de gratter les derniers pourcentages de performance (ex: passage de 80.26 à 81.88 sur le proxy de fraude) tout en bénéficiant d'un système d'Early Stopping strict pour économiser le temps de calcul.