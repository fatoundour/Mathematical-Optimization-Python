# ==============================================
#                  Imports
# ==============================================
import numpy as np
import pandas as pd


# ==============================================
#           Génération de la base B
# ==============================================
def gener_base(n, d):
    while True:
        B = np.random.randint(-d, d+1, size=(n, n))
        if np.linalg.matrix_rank(B) == n:
            return B


# ==============================================
#       Génération de matrice unimodulaire U
# ==============================================
def gener_unimodulaire(n, a, b):
    U = np.eye(n, dtype=int)
    for _ in range(a):
        op = np.random.choice(['permute', 'sign', 'add'])
        if op == 'permute':
            i, j = np.random.choice(n, size=2, replace=False)
            # échange des colonnes i et j
            U[:, [i, j]] = U[:, [j, i]]
        elif op == 'sign':
            i = np.random.randint(0, n)
            # changement de signe de la colonne i
            U[:, i] *= -1
        else:  # add
            src, dst = np.random.choice(n, size=2, replace=False)
            k = np.random.randint(-b, b+1)
            # addition de k fois la colonne src à la colonne dst
            U[:, dst] += k * U[:, src]
    return U

# ==============================================
#           Génération du vecteur v
# ==============================================
def gener_vect(n, b):
    return np.random.randint(-b, b + 1, size=(n,))

# ==============================================
#    Décomposition de Gram–Schmidt et LLL
# ==============================================
#----------------------------------------------------------------------------
def GSO(B):
    """Retourne (Bo, M) pour B = Bo @ M via Gram–Schmidt."""
    n = B.shape[1]
    Bo = np.zeros_like(B, dtype=float)
    M = np.eye(n, dtype=float)
    for j in range(n):
        v = B[:, j].astype(float)
        for i in range(j):
            mu = (Bo[:, i] @ B[:, j]) / (Bo[:, i] @ Bo[:, i])
            M[i, j] = mu
            v -= mu * Bo[:, i]
        Bo[:, j] = v
    return Bo, M

#----------------------------------------------------------------------------
def normalize(B, M):
    """Applique les opérations unimodulaires correspondantes à M sur B."""
    n = M.shape[0]
    for j in range(n):
        for i in range(j):
            kij = int(round(M[i, j]))
            if kij:
                # Opération unimodulaire : C_j -= kij * C_i
                B[:, j] -= kij * B[:, i]
                M[i, j] -= kij

#----------------------------------------------------------------------------
def condition(Bo, M, k, delta=0.75):
    """Teste la condition LLL sur l’indice k."""
    b_k_norm2 = Bo[:, k] @ Bo[:, k]
    proj = Bo[:, k+1] + M[k, k+1] * Bo[:, k]
    return delta * b_k_norm2 <= (proj @ proj)

#----------------------------------------------------------------------------
def GSO_opt(B, Bo, M, j):
    """Échange colonnes j et j+1 de B puis met à jour Bo et M localement."""
    # 1) échange de colonnes j <-> j+1
    B[:, [j, j+1]] = B[:, [j+1, j]]
    # 2) recalcul partiel de GSO pour colonnes j et j+1
    n = B.shape[1]
    for col in (j, j+1):
        v = B[:, col].astype(float)
        for i in range(col):
            mu = (Bo[:, i] @ B[:, col]) / (Bo[:, i] @ Bo[:, i])
            M[i, col] = mu
            v -= mu * Bo[:, i]
        Bo[:, col] = v
    # 3) on doit mettre à jour M[k,j+1] pour k < j
    #    mais l’opération ci‑dessus l’a déjà fait pour j et j+1

#----------------------------------------------------------------------------
def LLL_reduce(B, delta=0.75):
    """Réduit la base B selon l’algorithme LLL en place."""
    Bo, M = GSO(B)
    n = B.shape[1]
    k = 0
    while k < n - 1:
        normalize(B, M)
        if condition(Bo, M, k, delta):
            k += 1
        else:
            GSO_opt(B, Bo, M, k)
            k = max(k-1, 0)
    return B

#----------------------------------------------------------------------------

# ==============================================
#            Expériences & Affichage
# ==============================================
def experiment(n, d, a, b):
    # 1. Génération
    B = gener_base(n, d)
    U = gener_unimodulaire(n, a, b)
    v = gener_vect(n, b)

    # 2. Variantes de la base
    B1 = B
    B2 = LLL_reduce(B.copy())
    B3 = B @ U
    B4 = LLL_reduce(B3.copy())

    # 3. Distance à v
    def dist(Bi):
        xi = np.linalg.solve(Bi, v)
        mi = np.rint(xi).astype(int)
        wi = Bi @ mi
        return np.linalg.norm(v - wi)

    distances = [dist(B1), dist(B2), dist(B3), dist(B4)]

    # 4. DataFrame
    return B, U, v, pd.DataFrame({
        'Base': [
            'B original',
            'B LLL réduite',
            'B·U',
            'B·U LLL réduite'
        ],
        'Distance': distances
    })

# ==============================================
#       Petits exemples d’utilisation
# ==============================================

# Test de génération de base (n=4, d=5)
B = gener_base(4, 5)
print("Base générée B =", B)
print("\n#----------------------------------------------\n")

# Exemple d'utilisation de gener_unimodulaire (n=5, a=10, b=3)
U = gener_unimodulaire(5, 10, 3)
print("Matrice unimodulaire U =", U)
print("\n#----------------------------------------------\n")

# Exemple d'utilisation de gener_vect (n=5, b=10)
v = gener_vect(5, 10)
print("Vecteur v =", v)
print("\n#----------------------------------------------\n")

# Exemple supplémentaire (n=3, d=5)
B_tmp = gener_base(3, 5)
print("Base générée (exemple) B_tmp =", B_tmp)
print("\n#----------------------------------------------\n")

# Exemple supplémentaire U_tmp
U_tmp = gener_unimodulaire(3, 5, 2)
print("Matrice unimodulaire U_tmp =", U_tmp)
print("\n#----------------------------------------------\n")

# Exemple supplémentaire v_tmp
v_tmp = gener_vect(3, 5)
print("Vecteur v_tmp =", v_tmp)
print("\n#----------------------------------------------\n")

# ==============================================
#           Blocs d’exemples d’utilisation
# ==============================================

if __name__ == "__main__":
    # Exemple 1
    B, U, v, df = experiment(n=4, d=5,  a=10, b=3)
    print("## Exemple 1 – v =", v)
    print(df, "\n")
    print("\n#----------------------------------------------\n")

    # Exemple 2
    B, U, v, df = experiment(n=4, d=8,  a=15, b=4)
    print("## Exemple 2 – v =", v)
    print(df, "\n")
    print("\n#----------------------------------------------\n")

    # Exemple 3
    B, U, v, df = experiment(n=5, d=5,  a=10, b=3)
    print("## Exemple 3 – v =", v)
    print(df, "\n")
    print("\n#----------------------------------------------\n")

    # Exemple 4
    B, U, v, df = experiment(n=4, d=5,  a=20, b=6)
    print("## Exemple 4 – v =", v)
    print(df, "\n")
    print("\n#----------------------------------------------\n")

    # Exemple 5
    B, U, v, df = experiment(n=4, d=20, a=10, b=10)
    print("## Exemple 5 – v =", v)
    print(df)
    print("\n#----------------------------------------------\n")
