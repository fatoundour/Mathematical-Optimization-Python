# -*- coding: utf-8 -*-
import numpy as np
import pulp as pl

# ==============================================
#  Vérification de l'équilibre de Nash 
# ==============================================

def est_equilibre_nash(A, B, x, y):
    # Gains attendus des joueurs
    gain_x = np.dot(x, np.dot(A, y))  # Gain du joueur Ligne
    gain_y = np.dot(x, np.dot(B, y))  # Gain du joueur Colonne
    
    # Vérification pour le joueur Ligne
    for i in range(A.shape[0]):
        z = np.zeros_like(x)
        z[i] = 1  # Stratégie pure
        if np.dot(z, np.dot(A, y)) > gain_x:
            return False 
    
    # Vérification pour le joueur Colonne
    for j in range(B.shape[1]):
        w = np.zeros_like(y)
        w[j] = 1  # Stratégie pure
        if np.dot(x, np.dot(B, w)) > gain_y:
            return False 

    return True  # (x, y) est un équilibre de Nash

# ==============================================
#  Formulation MILP pour trouver un équilibre
# ==============================================

def probleme(A, B):
    m, n = A.shape
    prob = pl.LpProblem("Nash", pl.LpMaximize)

    # Variables de stratégie pour Ligne et Colonne
    x = [pl.LpVariable(f"x_{i}", lowBound=0) for i in range(m)]
    y = [pl.LpVariable(f"y_{j}", lowBound=0) for j in range(n)]

    # Variables binaires pour le support de x et de y
    b_x = [pl.LpVariable(f"b_x_{i}", cat="Binary") for i in range(m)]
    b_y = [pl.LpVariable(f"b_y_{i}", cat="Binary") for i in range(n)]

    # Valeur maximale attendue pour Ligne et pour Colonne
    M_x = pl.LpVariable("M_x", lowBound=None)
    M_y = pl.LpVariable("M_y", lowBound=None)

    # Variables de regret pour chaque action de Ligne et Colonne
    r_x = [pl.LpVariable(f"r_x_{i}", lowBound=0) for i in range(m)]
    r_y = [pl.LpVariable(f"r_y_{i}", lowBound=0) for i in range(n)]

    # Contraintes de probabilité
    prob += pl.lpSum(x) == 1, "Sum_x"
    prob += pl.lpSum(y) == 1, "Sum_y"

    # Contraintes de lien entre x (respectivement y)
    # et le support lié b_x (respectivement b_y)
    for i in range(m):
        prob += x[i] <= b_x[i], f"Link_x_b_{i}"
        #prob += x[i] >= epsilon * b_x[i], f"LowerBound_x_{i}"
    for i in range(n):
        prob += y[i] <= b_y[i], f"Link_y_b_{i}"
        #prob += y[i] >= epsilon * b_y[i], f"LowerBound_y_{i}"

    for i in range(m):
        Ay_i = pl.lpSum([A[i, j] * y[j] for j in range(n)])
        # On impose que Ay_i <= M_x pour tout i
        prob += Ay_i <= M_x, f"UpperBound_Ay_{i}"
        # Si l'action est utilisée (b_x[i] = 1), on impose que Ay_i = M_x <=> ri = 0
        # sinon ri <= M_x - max A_i_j
        prob += Ay_i >= M_x - np.max(A)*(1 - b_x[i]), f"Equality_Ay_{i}"
        # Définir le regret pour l'action i : r_x_i = M_x - Ay_i
        prob += r_x[i] >= M_x - Ay_i, f"Regret_x_{i}"

    for i in range(n):
        xB_i = pl.lpSum([x[j] * B[j, i] for j in range(m)])
        # On impose que Ay_i <= M_y pour tout i
        prob += xB_i <= M_y, f"UpperBound_xB_{i}"
        # Si l'action est utilisée (b_y[i] = 1), on impose que xB_i = M_y <=> ri = 0
        # sinon ri <= M_y - max B_i_j
        prob += xB_i >= M_y - np.max(B)*(1 - b_y[i]), f"Equality_xB_{i}"
        # Définir le regret pour l'action i : r_y_i = M_x - xB_i
        prob += r_y[i] >= M_y - xB_i, f"Regret_y_{i}"

    # Objectif : maximiser la valeur du jeu pour Ligne
    prob += M_x + M_y - pl.lpSum(r_x) - pl.lpSum(r_y), "Maximize_Game_Value"

    prob.solve()
    
    # Affichage des résultats
    print("\n#----------------------------------------------\n")
    print("Status:", pl.LpStatus[prob.status])
    print("Optimal value of M_x =", pl.value(M_x))
    print("Optimal value of M_y =", pl.value(M_y))
    print("Strategy for Ligne (x):")
    for i in range(m):
        print(f"  x[{i}] =", pl.value(x[i]), "with binary b_x =", pl.value(b_x[i]))
    print("Strategy for Colonne (y):")
    for j in range(n):
        print(f"  y[{j}] =", pl.value(y[j]), "with binary b_y =", pl.value(b_y[j]))
    print("\n#----------------------------------------------\n")
    
# ==============================================
#                  Tests 
# ==============================================

def generate_random_game(m=10, n=10, zero_sum=False, seed=None):
    rng = np.random.default_rng(seed)
    A = rng.integers(low=-100, high=100, size=(m, n))

    if zero_sum:
        B = -A
    else:
        B = rng.integers(low=-100, high=100, size=(m, n))

    return A, B


if __name__ == "__main__":
    # Test équilibre de Nash pur/mixte
    A1 = np.array([[3, -1], [0, 2]])
    B1 = -A1
    x1 = np.array([0.5, 0.5])
    y1 = np.array([0.4, 0.6])
    print("Équilibre Nash ?", est_equilibre_nash(A1, B1, x1, y1))
    print("\n#----------------------------------------------\n")

    # Problème MILP basique
    A2 = np.array([[3, 1], [2, 4]])
    B2 = np.array([[1, 2], [3, 0]])
    print("\nJeu général A2,B2:")
    probleme(A2, B2)
    print("\n#----------------------------------------------\n")

    # Jeu à somme nulle
    A3 = np.array([[3, 0], [0, 3], [1, 1]])
    B3 = -A3
    print("\nJeu somme nulle A3,B3:")
    probleme(A3, B3)
    print("\n#----------------------------------------------\n")

    # Dilemme du prisonnier
    A4 = np.array([[3, 0], [5, 1]])
    B4 = np.array([[3, 5], [0, 1]])
    print("\nDilemme du prisonnier:")
    probleme(A4, B4)
    print("\n#----------------------------------------------\n")

    # Pierre-Feuille-Ciseaux
    A5 = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    B5 = -A5
    print("\nPierre-Feuille-Ciseaux:")
    probleme(A5, B5)
    print("\n#----------------------------------------------\n")

    # Grand Jeu aléatoire
    A6, B6 = generate_random_game(m=25, n=25, seed=42)
    print("\nJeu aléatoire taille 25x25:")
    probleme(A6, B6)
    print("\n#----------------------------------------------\n")