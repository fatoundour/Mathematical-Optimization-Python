import numpy as np
from sympy import symbols, Matrix, simplify

#--------------------------------------------------------------------------------------------------------------------------------------------------
# Algorithme d'Euclide étendu
def euclide_etendu(a, b):
    if b == 0:
        return a, 1, 0
    pgcd, x1, y1 = euclide_etendu(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return pgcd, x, y

#--------------------------------------------------------------------------------------------------------------------------------------------------
# Matrice identité de la taille de A (n x n)
def matrice_Uid(A):
    return np.eye(A.shape[1], dtype=int)

#--------------------------------------------------------------------------------------------------------------------------------------------------
# Matrice unimodulaire M utilisée pour transformation
def matrice_M(A, i, j, v1, v2, v3, v4):
    M = matrice_Uid(A)
    M[i, i] = v1
    M[i, j] = v2
    M[j, i] = v3
    M[j, j] = v4
    return M

#--------------------------------------------------------------------------------------------------------------------------------------------------
# Élimination des éléments à droite de la diagonale (i, j > i)
def elimination_dessous(A, U, i, j):
    a = A[i, i]
    b = A[i, j]
    if b == 0:
        return
    g, x, y = euclide_etendu(a, b)
    M = matrice_M(A, i, j, x, -b // g, y, a // g)
    A[:] = A @ M
    U[:] = U @ M

#--------------------------------------------------------------------------------------------------------------------------------------------------
# Rendre la diagonale positive si besoin
def inverse_signe_neg(A, U, i):
    if A[i, i] <= 0:
        A[:, i] *= -1
        U[:, i] *= -1

#--------------------------------------------------------------------------------------------------------------------------------------------------
# Réduction des éléments à gauche de la diagonale (i, j < i)
def contrainte_inf_gauche(A, U, i, j):
    if A[i, i] == 0:
        return
    q = A[i, j] // A[i, i]
    if q != 0:
        M = matrice_M(A, j, i, 1, 0, -q, 1)
        A[:] = A @ M
        U[:] = U @ M

#--------------------------------------------------------------------------------------------------------------------------------------------------
# Réduction en forme normale d'Hermite
def hermite_fn(A):
    m, n = A.shape
    U = matrice_Uid(A)

    for i in range(m):
        for j in range(i + 1, n):
            if A[i, j] != 0:
                elimination_dessous(A, U, i, j)

        if A[i, i] < 0:
            inverse_signe_neg(A, U, i)

        for j in range(i):
            contrainte_inf_gauche(A, U, i, j)

    return A, U

#--------------------------------------------------------------------------------------------------------------------------------------------------
# Résolution du système Ax = b en utilisant la forme normale d'Hermite
def solver(A, b):
    H, U = hermite_fn(A.copy())
    H = Matrix(H)
    U = Matrix(U)
    b = Matrix(b).reshape(len(b), 1)
    m, n = H.shape

    D = H[:, :m]
    if any(D[i, i] == 0 for i in range(m)):
        print("❌ D non inversible.")
        return None

    y_head = []
    b_work = b.copy()
    for i in reversed(range(m)):
        if b_work[i] % D[i, i] != 0:
            print("❌ D⁻¹·b n’est pas entier → pas de solution.")
            return None
        q = b_work[i] // D[i, i]
        y_head.insert(0, q)
        b_work -= q * D[:, i]

    num_libres = n - m
    if num_libres == 0:
        v = Matrix(y_head)
    else:
        z_syms = [symbols('z')] if num_libres == 1 else list(symbols(f'z0:{num_libres}'))
        v = Matrix(list(y_head) + z_syms)

    x = simplify(U @ v)

    print("📐 Solution du système Ax = b :")
    print(f"x = ({', '.join(map(str, x))})ᵀ")
    return x

#--------------------------------------------------------------------------------------------------------------------------------------------------

################################
###           TESTS          ### 
################################

# Test 1 : Vérification de AU = H
A = np.array([
    [2, 3, 4],
    [2, 4, 6]
], dtype=int)
A_orig = A.copy()
H, U = hermite_fn(A)
print("H (forme normale d'Hermite) :\n", H)
print("U (matrice unimodulaire) :\n", U)
print("AU =\n", A_orig @ U)
print("Vérification : A @ U == H ?\n", np.all(A_orig @ U == H))
assert np.all(A_orig @ U == H), "❌ Erreur : A @ U n'est pas égal à H"
print("\n#----------------------------------------------\n")

# Test 2 : Résolution Ax = b (exemple simple)
A = np.array([
    [2, 3, 4],
    [2, 4, 6]
], dtype=int)
b = np.array([3, 4])
print("\n🔍 Contraintes du système Ax = b :")
print("2x + 3y + 4z = 3")
print("2x + 4y + 6z = 4")
print("\n#----------------------------------------------\n")

x = solver(A, b)

# Test 3 : Résolution Ax = b (exemple complexe)
A = np.array([
    [2,  4,  6,  8, 10, 12,  0,  0,  0,  0],
    [0,  1,  2,  3,  4,  5,  0,  1,  2,  3],
    [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    [0,  0,  1,  0, -1,  2,  0,  0,  1,  0],
    [3,  6,  9, 12, 15, 18,  1,  0, -1,  0],
    [1, -1,  0,  1, -1,  2,  2,  2,  2,  2]
], dtype=int)
b = np.array([60, 30, 10, 1, 90, 6])
print("\n🔍 Contraintes du système Ax = b (grand système) :")
for i, row in enumerate(A):
    terms = " + ".join(f"{coef}x{j}" for j, coef in enumerate(row) if coef != 0)
    print(f"{terms} = {b[i]}")
print("\n#----------------------------------------------\n")

x = solver(A, b)
