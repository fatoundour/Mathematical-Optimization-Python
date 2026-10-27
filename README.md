# Projet M1 – Algorithmes et Modélisation

📅 **Date de dépôt** : 11/05/2025

👉 **Il est recommandé de consulter les notebooks avant d’explorer les fichiers de code.**

---

## 👥 Membres de l’équipe

- **BARKAN INES** 
- **ABOUNAIM ELIAS** 
- **AABADLI ADAM** (Chef d'équipe)
- **NDOUR FATOUMATA**

---

## 📁 Organisation des fichiers

- `Jupyter_Notebook/` : contient les notebooks pour chaque TP.
- `Code` : contient le code séparé.

---

## 🔧 TPs traités

Ce dépôt contient notre rendu final pour les TP suivants du polycopié:

### 1. Forme normale d'Hermite (TP 3.8.2)
- Fichier : `HERMITE.ipynb`
- Contenu : Implémentation de l’algorithme de réduction en forme normale d'Hermite, tests sur plusieurs matrices, vérification des solutions symboliques.

### 2. Équilibre de Nash (TP 3.8.9, questions 7-8)
- Fichier : `NASH.ipynb`
- Contenu : Modélisation avec PULP, résolution d’un jeu stratégique, calcul d’un équilibre de Nash via programmation linéaire.

### 3. Problème du commis voyageur (TP 3.8.8)
- Fichier : `TSP.ipynb`
- Contenu : Implémentation d'une solution au TSP avec génération de graphes, algorithme exact, visualisation du parcours optimal.

### 4. Réduction de base et vecteur le plus proche (TP 4.4.2)
- Fichier : `Reduction_base.ipynb`
- Contenu : Réduction de base de matrices, application au problème du vecteur le plus proche, tests sur cas simples.

---

## 🛠️ Instructions pour tester

### Étape 1 : Créer un environnement Python
```bash
conda create -n projet_algo python=3.10
conda activate projet_algo
```

### Étape 2 : Installer les bibliothèques nécessaires
```bash
pip install -r requirements.txt
```

> `scikit-learn` est utilisé uniquement pour une visualisation avec `KMeans` dans un exemple.

### Étape 3 : Lancer Jupyter

Si Jupyter n’est pas installé : `pip install notebook`.
```bash
jupyter notebook
```

### Étape 4 : Ouvrir les notebooks
Les fichiers `.ipynb` sont autoportants : ils contiennent à la fois le code, la documentation et les exemples.
Pour plus de tests veuillez suivre le shéma des exemples dans la fin de chaque notebook, ajouter une cellule de code et exécutez la.

## 📦 Tester les scripts Python

1. Reproduire les étapes 1 et 2 de la section **🛠️ Instructions pour tester** (création/activation de l’environnement et installation des dépendances).  
2. Depuis la racine du projet, lancez les tests de chaque module avec :

```bash
conda activate projet_algo
python Code/<nom_du_fichier>.py  
```
Remplacez <nom_du_fichier> par le nom du script Python correspondant :

1. **Hermite**  

```bash
python Code/Hermite.py
```
2. **Nash**

```bash
python Code/Nash.py
```

3. **Reduction_base**

```bash
python Code/Reduction_base.py
```

4. **Tsp**
```bash
python Code/Tsp.py
```

## ✍️ Ajouter vos propres tests

- Ouvrez le script Code/<nom_du_fichier>.py dans votre éditeur.

- À la fin du fichier, ajoutez votre bloc de test (adapté au module).

- Sauvegardez, puis relancez la commande de test :

```bash
conda activate projet_algo
python Code/<nom_du_fichier>.py  
```
## 🧪 Ce que vous pouvez tester

Vous trouverez ci-dessous une description détaillée des tests que vous pouvez réaliser pour chaque module du projet. Cela vous permettra de valider leur bon fonctionnement et de mieux comprendre leur comportement.

### Tests pour `Code/Hermite.py`

Pour valider les deux fonctions principales de `Hermite.py` :

- **hermite_fn(A)**  
  - *Objectif* : calculer la forme normale d’Hermite `H` et la matrice unimodulaire `U` telles que `A·U = H`.  
  - *Comment tester* :  
    1. Choisir une petite matrice `A` (par exemple 2×3) dont on connaît la forme normale.  
    2. Vérifier que `A @ U` produit bien une matrice triangulaire supérieure et qu’elle coïncide avec `H`.

- **solver(A, b)**  
  - *Objectif* : résoudre le système entier `Ax = b` ou signaler qu’il n’a pas de solution.  
  - *Comment tester* :  
    1. *Cas soluble* : prendre un petit système `A` et `b` pour lequel la solution entière est évidente, puis vérifier que `solver` la renvoie.  
    2. *Cas non soluble* : choisir un `b` qui n’appartient pas à l’image de `A` et s’assurer que la fonction renvoie `None` (ou affiche le bon message d’erreur).

### Tests pour `Code/Nash.py`

Pour valider les deux fonctions principales de `Nash.py` :

- **est_equilibre_nash(A, B, x, y)**  
  - *Objectif* : vérifier qu’un couple de stratégies mixtes `(x, y)` est un équilibre de Nash pour les matrices de gains `A` (joueur Ligne) et `B` (joueur Colonne).  
  - *Comment tester* :  
    1. Choisir un petit jeu dont on connaît l’équilibre (p. ex. dilemme du prisonnier ou pierre-feuille-ciseaux).  
    2. Fournir à la fonction la distribution `(x, y)` supposée équilibrée et s’assurer qu’elle renvoie **True**.  
    3. Tester une paire de stratégies non optimales et vérifier qu’elle renvoie **False**.

- **probleme(A, B)**  
  - *Objectif* : formuler et résoudre un MILP qui calcule un équilibre mixte du même jeu.  
  - *Comment tester* :  
    1. Exécuter la fonction sur plusieurs jeux classiques (somme nulle, dilemme du prisonnier, pierre-feuille-ciseaux).  
    2. Vérifier que le statut renvoyé est **Optimal**.  
    3. Contrôler que les vecteurs `x` et `y` produits sont des distributions de probabilité (toutes les composantes ≥ 0 et somme à 1).  
    4. Pour un jeu à somme nulle, s’assurer que `M_x` et `M_y` sont opposées.

### Tests pour `Code/Tsp.py`

Pour valider les principales fonctionnalités de `Tsp.py` :

- **generate_problem(num_clients)**  
  - *Objectif* : créer une instance aléatoire du TSP/VRP avec un dépôt au centre et `num_clients` clients.  
  - *Comment tester* : pour un petit `num_clients` (p. ex. 3), vérifier que le dictionnaire retourné contient exactement `num_clients + 1` positions (dont le dépôt) et que chaque client a bien un ID unique.

- **compute_distance_matrix(clients, num_clients)**  
  - *Objectif* : construire la liste des positions et un dictionnaire des distances de Manhattan pour chaque paire.  
  - *Comment tester* : choisir un jeu de points fixe (manuellement) et comparer les distances produites à celles calculées “à la main”.

- **is_valid_solution(solution, num_clients)**  
  - *Objectif* : vérifier qu’une tournée commence et finit au dépôt (0) et visite chaque client exactement une fois.  
  - *Comment tester* : fournir une tournée correcte et plusieurs incorrectes (double visite, omission, absence de retour au dépôt) et vérifier `True`/`False`.

- **tsp_dp(positions, distance_matrix)**  
  - *Objectif* : résoudre exactement le TSP par programmation dynamique.  
  - *Comment tester* : appliquer `tsp_dp` sur une petite instance (3 ou 4 nœuds) dont on connaît le parcours optimal et son coût, et vérifier la correspondance.

- **solve_tsp_linear(positions, distance_matrix)**  
  - *Objectif* : formuler et résoudre le TSP en MILP (Miller-Tucker-Zemlin).  
  - *Comment tester* : lancer sur la même petite instance que pour `tsp_dp`, vérifier que la distance minimale est identique et que le circuit renvoyé est valide.

- **local_search_tsp(n, distance_matrix)**  
  - *Objectif* : améliorer une tournée via l’heuristique 2-opt.  
  - *Comment tester* : générer une tournée aléatoire puis appeler `local_search_tsp`; vérifier que la solution retournée est valide (`is_valid_solution`) et que son coût est inférieur ou égal à celui de la tournée initiale.

- **christofides_tsp(positions, distance_matrix)**  
  - *Objectif* : exécuter l’approximation de Christofides (1.5-approximation).  
  - *Comment tester* : pour une instance connue, comparer le coût obtenu à celui de `tsp_dp` et s’assurer qu’il ne dépasse pas 1.5 × coût optimal, tout en renvoyant une tournée valide.

### Tests pour `Code/Reduction_base.py`

Pour valider les fonctionnalités principales du module :

- **gener_base(n, d)**  
  - *Objectif* : produire une matrice carrée aléatoire de rang plein, coefficients dans [–d, d].  
  - *Comment tester* : pour des petits paramètres (n = 3, d = 5), vérifiez que la matrice retournée est de dimension n×n et que son rang vaut bien n.

- **gener_unimodulaire(n, a, b)**  
  - *Objectif* : générer une matrice unimodulaire (|det| = 1) à partir de `a` opérations aléatoires et coefficients bornés par `b`.  
  - *Comment tester* : pour n = 4, a = 5, b = 3, contrôlez que U est inversible et que son déterminant est ±1.

- **LLL_reduce(B, delta=0.75)**  
  - *Objectif* : appliquer l’algorithme LLL pour réduire la base B en place.  
  - *Comment tester* : prenez une base B connue (p. ex. un réseau avec un vecteur court évident), exécutez `LLL_reduce(B.copy())` et vérifiez que la norme du vecteur le plus court dans la base réduite est bien inférieure ou égale à celle de la base d’origine.

- **experiment(n, d, a, b)**  
  - *Objectif* : réaliser plusieurs variantes de la base (originale, réduite, modulée par U, puis réduite) et mesurer la distance au vecteur v, retournant un DataFrame des résultats.  
  - *Comment tester* : pour n = 3, d = 5, a = 5, b = 2, appelez `experiment` et vérifiez que le DataFrame comporte exactement quatre lignes (« B original », « B LLL réduite », « B·U », « B·U LLL réduite ») avec des distances numériques non négatives.


> **Conseil** : 
Pour tester Nash.py et Hermite.py, commencez avec des matrices 2×2 simples, puis testez des cas un peu plus grands pour couvrir toutes les branches du code.

---

Projet universitaire collaboratif. Les implémentations sont fournies à titre pédagogique ; leur exactitude sur tous les cas n’a pas été vérifiée dans le cadre de cette mise en ligne.