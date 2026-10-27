# -*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pulp as pl
import time
from collections import defaultdict

# ==============================================================
#                   TSP & VRP Utilities
# --------------------------------------------------------------
# Diverses méthodes pour résoudre le problème du voyageur 
# de commerce (TSP), incluant :
#   - Programmation dynamique (exacte)
#   - Optimisation linéaire (MILP avec contraintes de sous-tours)
#   - Recherche locale (2-opt)
#   - Approximation de Christofides
#   - Approche VRP simplifiée par K-Means et équilibrage central
# --------------------------------------------------------------
# ==============================================================

# --------------------------------------------
# Génération aléatoire du problème : positions
# --------------------------------------------

def generate_problem(num_clients):
    """
    Génère une instance aléatoire du problème TSP/VRP :
    - On considère un dépôt au centre de la grille.
    - On place `num_clients` clients aléatoirement sur une grille de taille `num_clients^2`.

    Paramètres :
    -------------
    num_clients : int
        Nombre de clients à visiter (hors dépôt).

    Retourne :
    ---------
    clients : dict
        Clé = (x, y), valeur = liste d'IDs de clients à cette position.
    """
    grid_size = num_clients**2
    clients = {}
    # Dépôt initialisé au centre
    clients[((grid_size-1)/2, (grid_size-1)/2)] = [0]

    # Placement aléatoire des clients
    for i in range(1, num_clients + 1):
        x = random.randint(0, grid_size - 1)
        y = random.randint(0, grid_size - 1)
        clients.setdefault((x, y), []).append(i)
    return clients

# ----------------------------------
# Calcul de la distance de Manhattan
# ----------------------------------

def manhattan_distance(p1, p2):
    """
    Calcule la distance de Manhattan entre deux points.

    Paramètres :
    -------------
    p1, p2 : tuple de deux entiers
        Coordonnées (x, y).

    Retourne :
    ---------
    dist : int
        Somme des différences absolues sur chaque axe.
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# --------------------------------------------------
# Construction de la matrice des distances complètes
# --------------------------------------------------

def compute_distance_matrix(clients, num_clients, verbose=False):
    """
    Construit la liste des positions et un dictionnaire de distances
    pour toutes les paires (i, j) de positions.

    Paramètres :
    -------------
    clients : dict
        Dictionnaire des positions de clients.
    num_clients : int
        Nombre de clients (hors dépôt).
    verbose : bool
        Si True, affiche la progression du calcul.

    Retourne :
    ---------
    positions : list
        Liste des positions (x, y) incluant le dépôt en premier.
    distance_matrix : dict
        Clé = (i, j), valeur = distance manhattan.
    """
    grid_size = num_clients**2
    positions = list(clients.keys())
    distance_matrix = {}
    # On ajoute la distance zéro pour (0,0) : centre
    distance_matrix[(0, 0)] = 0

    # Calcul pair à pair pour i < j
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if verbose:
                print(f"compute matrix distance {positions[i]} à {positions[j]}")
            d = manhattan_distance(positions[i], positions[j])
            distance_matrix[(i, j)] = d
            distance_matrix[(j, i)] = d
    return positions, distance_matrix

# ----------------------------------------
# Vérification de la validité d'une tournée
# ----------------------------------------

def is_valid_solution(solution, num_clients):
    """
    Vérifie si une tournée (liste de clients) est valide :
    - Démarre et se termine au dépôt (0).
    - Visite chaque client exactement une fois.

    Paramètres :
    -------------
    solution : list
        Suite de visites (IDs).
    num_clients : int
        Nombre de clients (hors dépôt).

    Retourne :
    ---------
    bool
        True si la solution est valide.
    """
    # Le circuit ne commence/termine pas au dépôt
    if solution[0] != 0 or solution[-1] != 0:
        return False
    
    # On ignore les deux zéros (départ/retour)
    visited = solution[1:-1]
    if len(visited) != num_clients:
        return False  # Tous les clients n'ont pas été visités
    
    # Tous les IDs 1..n doivent être présents
    return set(visited) == set(range(1, num_clients + 1))

# ================================
#    TSP par Programmation Dynamique
# ================================

# visited est un masque binaire qui représente les villes visités ex sur quatre villes 0011 signifie que le noeud 0 et 1 est visité 
def tsp_dp(positions, distance_matrix):
    """
    Résout le TSP de manière exacte via programmation dynamique
    avec bitmask (complexité O(n^2 2^n)).

    Paramètres :
    -------------
    positions : list
        Positions des nœuds, dont l'indice 0 est le dépôt.
    distance_matrix : dict
        Dictionnaire des distances.

    Retourne :
    ---------
    min_cost : float
        Coût optimal.
    path : list
        Ordre des visites (IDs), incluant retour au dépôt.
    """
    n = len(positions)
    memo = {}
    parent = {}

    def dp(visited, last):
        # Cas mémorisé
        if (visited, last) in memo:
            return memo[(visited, last)]
        # Si tous visités, retourner coût vers dépôt
        if visited == (1 << n) - 1:
            return distance_matrix.get((last, 0), float('inf')), [0]
        min_cost = float('inf')
        best_next = None
        for city in range(n):
            if not (visited & (1 << city)):  # Si la ville n'a pas encore été visitée
                cost, path = dp(visited | (1 << city), city)
                cost += distance_matrix.get((last, city), float('inf'))
                if cost < min_cost:
                    min_cost = cost
                    best_next = city
                    best_path = path
        
        memo[(visited, last)] = (min_cost, [best_next] + best_path)
        parent[(visited, last)] = best_next
        return memo[(visited, last)]

    cost, path = dp(1, 0)
    return cost, [0] + path

# ==================================================
#    TSP par Optimisation Linéaire (MILP U-V)
# ==================================================

def extract_ordered_circuit(x_vars, n):
    """
    Reconstruit la tournée ordonnée à partir de variables x[i][j]=1.

    Paramètres :
    -------------
    x_vars : dict
        Variables de décision.
    n : int
        Nombre de nœuds.

    Retourne :
    ---------
    ordered_circuit : list
        Circuit séquentiel.
    """
    path_dict = {i: j for i in range(n) for j in range(n) if i != j and pl.value(x_vars[i][j]) == 1}
    start = 0
    ordered_circuit = [start]
    while len(ordered_circuit) < n:
        next_node = path_dict[ordered_circuit[-1]]
        ordered_circuit.append(next_node)
    ordered_circuit.append(0)  # Retour au dépôt
    return ordered_circuit


# -----------------------------------------------------------------------------------

def solve_tsp_linear(positions, distance_matrix):
    """
    Formule et résout le TSP en MILP (Minimisation linéaire) en utilisant
    la formulation de Miller-Tucker-Zemlin (U-V) pour éliminer les sous-tours.

    Paramètres :
    -------------
    positions : list
        Positions des nœuds.
    distance_matrix : dict
        Distances.

    Retourne :
    ---------
    ordered_circuit : list
        Circuit optimal ordonné.
    total_distance : float
        Coût du circuit.
    """
    n = len(positions)
    
    # Définition du problème
    prob = pl.LpProblem("TSP", pl.LpMinimize)
    
    # Variables de décision : x[i][j] = 1 si on va de i à j, 0 sinon
    x = pl.LpVariable.dicts("x", (range(n), range(n)), cat="Binary")

    # methode du potentiel
    u = pl.LpVariable.dicts("u", range(1, n), lowBound=1, upBound=n-1, cat="Continuous")

    # Fonction objectif : Minimisation de la distance parcourue
    prob += pl.lpSum(distance_matrix.get((i, j), 0) * x[i][j] for i in range(n) for j in range(n) if i != j)

    # Chaque nœud doit être visité exactement une fois
    for i in range(n):
        prob += pl.lpSum(x[i][j] for j in range(n) if i != j) == 1  # Sortie
        prob += pl.lpSum(x[j][i] for j in range(n) if i != j) == 1  # Entrée

    # Contraintes de sous-tour
    for i in range(1, n):  # On commence à 1, car u[0] est fixe implicitement
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + (n - 1) * x[i][j] <= n - 2

    # Résolution
    prob.solve()

    # Extraction du circuit optimal (s'affiche pas dans l'ordre à arranger )
    circuit = []
    for i in range(n):
        for j in range(n):
            if pl.value(x[i][j]) == 1:
                circuit.append((i, j))

    ordered_circuit = extract_ordered_circuit(x, n)
    total_distance = pl.value(prob.objective)
    return ordered_circuit, total_distance


# ==============================
#    Recherche Locale (2-opt)
# ==============================

def generate_random_circuit(n):
    """
    Génère une tournée initiale aléatoire (départ et arrivée au dépôt).
    """
    circuit = list(range(1, n))
    random.shuffle(circuit)
    return [0] + circuit + [0]

# -----------------------------------------------------------------------------------
def get_neighbors(circuit):
    """
    Génère tous les voisins de type 2-opt d'une tournée.
    """
    n = len(circuit) - 1
    neighbors = []
    for i in range(1, n-1):
        for j in range(i+2, n):
            new_circ = circuit[:i] + circuit[i:j+1][::-1] + circuit[j+1:]
            neighbors.append(new_circ)
    return neighbors

# -----------------------------------------------------------------------------------
def distance(circuit, distance_matrix):
    """
    Calcule la distance totale d'une tournée.
    """
    return sum(distance_matrix[(circuit[i], circuit[i+1])]
               for i in range(len(circuit)-1))

# -----------------------------------------------------------------------------------
def local_search_tsp(n, distance_matrix, max_iterations=100000, time_limit=60):
    """
    Améliore une tournée via l'heuristique 2-opt jusqu'à stagnation.

    Paramètres :
    -------------
    n : int
        Nombre de nœuds.
    distance_matrix : dict
        Dictionnaire des distances.
    max_iterations : int
        Nombre maximal d'itérations.
    time_limit : float
        Durée limite en secondes.

    Retourne :
    ---------
    circuit : list
        Meilleure tournée trouvée.
    best_distance : float
        Coût associé.
    """
    circuit = generate_random_circuit(n)
    best_distance = distance(circuit, distance_matrix)
    
    improvement = True
    iterations = 0
    start_time = time.time()
    
    while improvement and iterations < max_iterations and (time.time() - start_time) < time_limit:
        improvement = False
        for neighbor in get_neighbors(circuit):
            new_distance = distance(neighbor, distance_matrix)
            if new_distance < best_distance:
                circuit = neighbor
                best_distance = new_distance
                improvement = True
                break
        iterations += 1

    return circuit, best_distance

# =============================================
#      Approximation de Christofides (heuristique)
# =============================================

def prim_mst(positions, distance_matrix):
    """
    Construit l'arbre couvrant minimum (MST) via Prim naïf (O(n^2)).
    """
    n = len(positions)
    mst = {0: {}}
    visited = {0}
    edges = []

    while len(visited) < n:
        min_edge = None
        min_cost = float('inf')
        for u in visited:
            for v in range(n):
                if v not in visited and distance_matrix[(u, v)] < min_cost:
                    min_cost = distance_matrix[(u, v)]
                    min_edge = (u, v)
        
        if min_edge:
            u, v = min_edge
            visited.add(v)
            mst[u][v] = min_cost
            mst[v] = {u: min_cost}
            edges.append(min_edge)
            
    return mst

# -----------------------------------------------------------------------------------
def find_odd_degree_nodes(mst):
    """
    Identifie les sommets de degré impair dans le MST.
    """
    return [node for node, neigh in mst.items() if len(neigh) % 2 != 0]

# -----------------------------------------------------------------------------------
def find_perfect_matching(distance_matrix, odd_nodes):
    """
    Trouve un couplage parfait pondéré glouton parmi les nœuds impairs.
    """
    matching = []
    while odd_nodes:
        u = odd_nodes.pop()
        min_cost = float('inf')
        v_match = None
        for v in odd_nodes:
            if distance_matrix[(u, v)] < min_cost:
                min_cost = distance_matrix[(u, v)]
                v_match = v
        matching.append((u, v_match))
        odd_nodes.remove(v_match)
    return matching

# -----------------------------------------------------------------------------------
def create_eulerian_distance_matrix(mst, matching, distance_matrix):
    """
    Fusionne MST et couplage pour créer un graphe eulérien (tous degrés pairs).
    """
    eulerian_distance_matrix = {u: neighbors.copy() for u, neighbors in mst.items()}

    for u, v in matching:
        cost = distance_matrix.get((u, v), float('inf'))
        eulerian_distance_matrix[u][v] = cost
        eulerian_distance_matrix[v][u] = cost

    return eulerian_distance_matrix

# -----------------------------------------------------------------------------------
def find_eulerian_circuit(eulerian_graph, start=0):
    """
    Obtient un circuit eulérien (Hierholzer) dans un graphe à degrés pairs.
    """
    adj_list = defaultdict(list)
    
    # Utiliser eulerian_graph pour créer la liste d'adjacence
    for u, neighbors in eulerian_graph.items():
        for v in neighbors:
            adj_list[u].append(v)
    
    stack = [start]  # Début du parcours à partir du sommet 'start'
    circuit = []  # Liste pour stocker le circuit eulérien
    
    while stack:
        u = stack[-1]  # Le sommet actuel
        if adj_list[u]:  # Si le sommet a des voisins
            v = adj_list[u].pop()  # On prend un voisin et on l'ajoute à la pile
            stack.append(v)
        else:  # Sinon, on a fini d'explorer ce sommet
            circuit.append(stack.pop())  # On ajoute le sommet à notre circuit
    
    # Retourner le circuit inversé pour qu'il soit dans le bon ordre
    return circuit[::-1]

# -----------------------------------------------------------------------------------
def create_hamiltonian_circuit(eulerian_circuit, start=0):
    """
    Transforme un circuit eulérien en circuit hamiltonien sans répétition.
    """
    visited = set()
    hamiltonian_circuit = []
    for node in eulerian_circuit:
        if node not in visited:
            hamiltonian_circuit.append(node)
            visited.add(node)
    hamiltonian_circuit.append(start)
    return hamiltonian_circuit

# -----------------------------------------------------------------------------------
def calculate_tsp_cost(circuit, distance_matrix):
    """
    Calcule le coût d'une tournée (TSP) fermée.
    """
    total_cost = 0
    for i in range(len(circuit) - 1):
        total_cost += distance_matrix.get((circuit[i], circuit[i + 1]), float('inf'))
    total_cost += distance_matrix.get((circuit[-1], circuit[0]), float('inf'))
    return total_cost

# -----------------------------------------------------------------------------------
def christofides_tsp(positions, distance_matrix):
    """
    Applique l'heuristique de Christofides pour un 1.5-approx TSP.
    """
    mst= prim_mst(positions, distance_matrix) # construit l'arbre couvrant minimum
    odd_nodes = find_odd_degree_nodes(mst) # recupere les sommets de degré impair

    matching = find_perfect_matching(distance_matrix, odd_nodes) # trouvre un couplage parfait
    eulerian_graph = create_eulerian_distance_matrix(mst, matching, distance_matrix) # ajoute le couplage à l'arbre couvrant pour avoir un que des sommets de degré pair
    eulerian_circuit = find_eulerian_circuit(eulerian_graph) # trouver le circuit eulerien (un graphe avec que des sommets de degré pair en a forcément un)
    hamiltonian_circuit = create_hamiltonian_circuit(eulerian_circuit) # crée le circuit hamiltonien à partir du circuit eulérien
    total_cost = calculate_tsp_cost(hamiltonian_circuit, distance_matrix) # calcule le cout
    return hamiltonian_circuit, total_cost  

# ==============================================
#       Tests et comparaisons de performances
# ==============================================

# -----------------------------------------------------------------------------------
def test_algorithm(algorithm, positions, distance_matrix, time_limit=60):
    """
    Lance un algorithme de résolution TSP et mesure temps + succès.
    """
    start = time.time()
    result = algorithm(positions, distance_matrix)
    elapsed = time.time() - start
    return elapsed < time_limit, elapsed, result

print("\n#----------------------------------------------\n")

# -----------------------------------------------------------------------------------
def run_tests_dp(positions, distance_matrix):
    print("Test programmation dynamique...")
    success, time_taken, (cost, circuit) = test_algorithm(tsp_dp, positions, distance_matrix)
    print(f"Coût tsp_dp: {cost}")
    print("Circuit tsp_dp:", circuit)
    print(f"Temps: {time_taken:.2f}s, Succès: {'Oui' if success else 'Non'}\n")
    print("\n#----------------------------------------------\n")
# -----------------------------------------------------------------------------------
def run_tests_opti(positions, distance_matrix):
    print("Test optimisation linéaire...")
    success, time_taken, (path, cost) = test_algorithm(solve_tsp_linear, positions, distance_matrix)
    print(f"Coût optimisation linéaire: {cost}")
    print("Chemin optimisation linéaire:", path)
    print(f"Temps: {time_taken:.2f}s, Succès: {'Oui' if success else 'Non'}\n")
    print("\n#----------------------------------------------\n")
# -----------------------------------------------------------------------------------
def run_tests_rl(n, distance_matrix):
    print("Test recherche locale...")
    success, time_taken, (circuit, best_distance) = test_algorithm(lambda p, d: local_search_tsp(n, d), None, distance_matrix)
    print(f"Coût recherche locale: {best_distance}")
    print("Circuit recherche locale:", circuit)
    print(f"Temps: {time_taken:.2f}s, Succès: {'Oui' if success else 'Non'}\n")
    print("\n#----------------------------------------------\n")
# -----------------------------------------------------------------------------------
def run_tests_christofides(positions, distance_matrix):
    print("Test approximation Christofides...")
    success, time_taken, (circuit, cost) = test_algorithm(christofides_tsp, positions, distance_matrix)
    print(f"Coût Christofides: {cost}")
    print("Circuit Christofides:", circuit)
    print(f"Temps: {time_taken:.2f}s, Succès: {'Oui' if success else 'Non'}\n")
    print("\n#----------------------------------------------\n")
# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    num_clients = 10
    print(f"Test pour {num_clients} clients:")
    clients = generate_problem(num_clients)

    if num_clients <= 15 :
        for key, item in clients.items():
            print(f"position : {key}, client(s) : {item}")
    positions, distance_matrix = compute_distance_matrix(clients, num_clients, False)
    if num_clients <= 15 :
        for i, j in enumerate(positions) :
            print(f"({i},{j}) \n")
        for key, item in distance_matrix.items():
            print(f"position : {key}, distance : {item}") 
    print("\n#------------- TESTS ALGORITHMES ---------------\n") 
    run_tests_dp(positions, distance_matrix)
    run_tests_opti(positions, distance_matrix)
    run_tests_rl(num_clients, distance_matrix)
    run_tests_christofides(positions, distance_matrix)
    print("\n#----------------------------------------------\n")

# ==================================================
#   Fonctions de benchmark qualité et taille tests
# ==================================================


def test_une_taille(method, n, max_time=60):
    clients = generate_problem(n)
    positions, distance_matrix = compute_distance_matrix(clients, n)

    if method == "local":
        algo = lambda p, d: local_search_tsp(n, d, max_iterations=500, time_limit=max_time)
    elif method == "DP":
        algo = tsp_dp
    elif method == "MILP":
        algo = solve_tsp_linear
    elif method == "Christofides":
        algo = christofides_tsp
    else:
        return "Méthode inconnue"

    import time
    start = time.perf_counter()
    try:
        success, time_taken, result = test_algorithm(algo, positions, distance_matrix, time_limit=max_time)
    except:
        success, time_taken, result = False, None, None
    elapsed = time.perf_counter() - start

    return {
        "method": method,
        "n": n,
        "success": success,
        "time_taken": round(elapsed, 3),
        "tim": time_taken,
        "result": result
    }

# -----------------------------------------------------------------------------------
def evaluate_quality(approx_cost, optimal_cost):
    return round((approx_cost - optimal_cost) / optimal_cost * 100, 2)

# -----------------------------------------------------------------------------------
def run_quality_multiple_tests(n, runs=5, max_time=60):
    qualities = {
        "recherche_locale": [],
        "Christofides": []
    }

    for i in range(runs):
        print(f"Test {i+1}/{runs}")
        clients = generate_problem(n)
        positions, distance_matrix = compute_distance_matrix(clients, n)

        # Solution optimale via MILP
        _, _, [opt_circuit, opt_cost] = test_algorithm(solve_tsp_linear, positions, distance_matrix)

        # recherche_locale
        _, _, [circuit_recherche_locale, cost_recherche_localet] = test_algorithm(
            lambda p, d: local_search_tsp(n, d, max_iterations=500, time_limit=max_time), positions, distance_matrix
        )
        error_recherche_locale = evaluate_quality(cost_recherche_localet, opt_cost)
        qualities["recherche_locale"].append(error_recherche_locale)

        # Christofides
        _, _, [circuit_christofides, cost_christofides] = test_algorithm(christofides_tsp, positions, distance_matrix)
        error_christofides = evaluate_quality(cost_christofides, opt_cost)
        qualities["Christofides"].append(error_christofides)

    # Moyennes
    avg_recherche_locale = sum(qualities["recherche_locale"]) / runs
    avg_chris = sum(qualities["Christofides"]) / runs

    return {
        "n": n,
        "avg_recherche_locale_error_%": round(avg_recherche_locale, 2),
        "avg_christofides_error_%": round(avg_chris, 2)
    }
# -----------------------------------------------------------------------------------
# Exemple d'exécution et affichage
if __name__ == "__main__":
    result = run_quality_multiple_tests(20, runs=5)
    print("Benchmark qualité (n=20):", result)

# ==============================================
#     VRP simplifié avec K-Means et équilibrage
# ==============================================

def solve_real_problem_with_kmeans_and_central_balancing(n=1000, k=10, max_iterations=1000, time_limit=60, alpha=0.5):
    """
    Approche VRP :
    1. Sépare 10% des clients proches du dépôt et les clients éloignés.
    2. Clusterisation K-Means des clients éloignés.
    3. Répartition des clients centraux pour équilibrer la charge.
    4. Optimisation locale 2-opt pour chaque cluster.

    Retourne :
        dict contenant distances, tailles et tournées.
    """
    # Génération des données
    clients_dict = generate_problem(n)
    positions, distance_matrix = compute_distance_matrix(clients_dict, n)
    depot = positions[0]
    client_positions = positions[1:]

    # Identifier les clients les plus proches du dépôt
    distances_to_depot = [np.linalg.norm(np.array(p) - np.array(depot)) for p in client_positions]
    num_central = int(n * 0.1)  # 10% des clients les plus proches
    central_indices = np.argsort(distances_to_depot)[:num_central]
    outer_indices = np.argsort(distances_to_depot)[num_central:]

    # Appliquer KMeans sur les clients "éloignés"
    outer_positions = [client_positions[i] for i in outer_indices]
    kmeans = KMeans(n_clusters=k, random_state=0).fit(outer_positions)

    clusters = [[] for _ in range(k)]
    for idx, label in zip(outer_indices, kmeans.labels_):
        clusters[label].append(idx)

    # Équilibrage : attribuer les clients centraux aux clusters "sous-chargés"
    cluster_distances = [0] * k
    cluster_sizes = [len(cl) for cl in clusters]

    for central_idx in central_indices:
        # Choisir le meilleur cluster selon un score pondéré
        best_score = float('inf')
        best_cluster = None
        for i in range(k):
            # Approximation du coût d’ajouter ce client : distance + poids taille
            if cluster_sizes[i] == 0:
                score = float('inf')
            else:
                dist = np.linalg.norm(np.array(client_positions[central_idx]) - np.array(depot))
                score = alpha * cluster_distances[i] + (1 - alpha) * cluster_sizes[i]
            if score < best_score:
                best_score = score
                best_cluster = i
        clusters[best_cluster].append(central_idx)
        cluster_sizes[best_cluster] += 1

    # Optimiser les tournées
    total_distances = []
    circuits = []

    for i, cluster in enumerate(clusters):
        cluster_with_depot = [0] + [idx + 1 for idx in cluster]
        global_to_local = {g_idx: l_idx for l_idx, g_idx in enumerate(cluster_with_depot)}

        local_distance_matrix = {
            (global_to_local[i], global_to_local[j]): distance_matrix[(i, j)]
            for i in cluster_with_depot for j in cluster_with_depot if i != j
        }

        tour, dist = local_search_tsp(len(cluster_with_depot), local_distance_matrix,
                                      max_iterations=max_iterations, time_limit=time_limit)

        global_tour = [cluster_with_depot[i] for i in tour]
        total_distances.append(dist)
        circuits.append(global_tour)

        print(f"Livreur {i+1}: {len(cluster)} clients, distance = {dist:,.0f}")

    max_distance = max(total_distances)
    mean_distance = np.mean(total_distances)
    mean_clients = np.mean([len(c) for c in clusters])

    print(f"\nDistance maximale (livreur le plus chargé): {max_distance:,.0f}")
    print(f"Distances moyennes par livreur: {mean_distance:,.2f}")
    print(f"Nombre moyen de clients par livreur: {mean_clients:.2f}")

    return {
        "max_distance": max_distance,
        "average_distance": mean_distance,
        "average_cluster_size": mean_clients,
        "positions": positions,
        "clusters": clusters,
        "tours": circuits
    }
# -----------------------------------------------------------------------------------
def plot_clusters(positions, clusters, depot_index=0):
    """
    Affiche les clusters sur un plan 2D, avec le dépôt.
    
    positions : liste des positions (le dépôt doit être en position 0)
    clusters : liste de clusters, chacun contenant les indices des clients (hors dépôt)
    """
    colors = cm.tab10(np.linspace(0, 1, len(clusters)))
    plt.figure(figsize=(10, 10))

    # Affichage des clients par cluster
    for i, cluster in enumerate(clusters):
        cluster_pos = [positions[j+1] for j in cluster]  # +1 car positions[0] = dépôt
        xs, ys = zip(*cluster_pos)
        plt.scatter(xs, ys, color=colors[i], label=f"Livreur {i+1}", s=20)

    # Affichage du dépôt
    depot_x, depot_y = positions[depot_index]
    plt.scatter(depot_x, depot_y, c='black', s=100, marker='X', label='Dépôt')

    plt.title("Répartition des clients par livreur")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()
    
# -----------------------------------------------------------------------------------
def plot_optimized_tours(positions, tours):
    """
    Trace les tournées optimisées pour chaque livreur.
    """
    depot = positions[0]
    plt.figure(figsize=(10, 10))
    colors = plt.cm.get_cmap("tab10", len(tours))

    for i, tour in enumerate(tours):
        pts = [positions[idx] for idx in tour]
        x, y = zip(*pts)
        plt.plot(x, y, marker='o', label=f"Livreur {i+1}", color=colors(i))

    plt.scatter(*depot, color="black", marker="X", s=100, label="Dépôt")
    plt.title("Tournées optimisées par livreur")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

# -----------------------------------------------------------------------------------
# Exécution VRP K-Means + équilibrage + affichage
if __name__ == "__main__":
    result = solve_real_problem_with_kmeans_and_central_balancing()
    plot_clusters(result["positions"], result["clusters"])
    plot_optimized_tours(result["positions"], result["tours"])
