"""
Détection des collisions et règles de réaction.
"""

import numpy as np


def detect_collisions(pos_A, pos_B, r_c):
    """
    Renvoie toutes les paires (i, j) telles que la distance entre la particule A n°i
    et la particule B n°j est strictement inférieure à r_c.

    Version naïve : on calcule les N_A x N_B distances d'un coup par broadcasting NumPy.
    On compare les distances au carré à r_c² pour éviter de calculer des racines.
    Coût en mémoire et en temps proportionnel à N_A x N_B.

    Paramètres
    ----------
    pos_A : ndarray (N_A, 2)
        Positions des particules A (µm).
    pos_B : ndarray (N_B, 2)
        Positions des particules B (µm).
    r_c : float
        Rayon de réaction (µm).

    Retour
    ------
    ndarray (n_paires, 2) d'entiers
        Chaque ligne [i, j] donne l'indice i dans pos_A et l'indice j dans pos_B.
        Une même particule peut apparaître dans plusieurs paires (par exemple un A
        proche de deux B) : c'est à la fonction de réaction de gérer ce cas.
    """
    # differences[i, j] = pos_A[i] - pos_B[j], tableau (N_A, N_B, 2)
    differences = pos_A[:, np.newaxis, :] - pos_B[np.newaxis, :, :]
    distances_carre = np.sum(differences ** 2, axis=2)  # tableau (N_A, N_B), en µm²
    i, j = np.nonzero(distances_carre < r_c ** 2)
    return np.column_stack((i, j))


if __name__ == "__main__":
    from scipy.spatial.distance import cdist

    # Test 1 — cas construit à la main, r_c = 1 µm
    #   A0 = (0, 0)   et B0 = (0.5, 0) : distance 0.5 -> collision
    #   A0 = (0, 0)   et B2 = (0, -0.8): distance 0.8 -> collision (A0 touche deux B)
    #   A0 = (0, 0)   et B3 = (1, 0)   : distance exactement 1 -> pas de collision (inégalité stricte)
    #   A1 = (10, 10) et B1 = (10, 12) : distance 2 -> pas de collision
    #   A2 = (5, 5)   n'est proche d'aucun B
    pos_A = np.array([[0.0, 0.0], [10.0, 10.0], [5.0, 5.0]])
    pos_B = np.array([[0.5, 0.0], [10.0, 12.0], [0.0, -0.8], [1.0, 0.0]])
    paires = detect_collisions(pos_A, pos_B, r_c=1.0)
    attendu = np.array([[0, 0], [0, 2]])
    print(f"Test 1 — paires détectées : {paires.tolist()}, attendues : {attendu.tolist()}")
    assert np.array_equal(paires, attendu)

    # Test 2 — aucune particule B : aucune paire, sans erreur
    paires_vides = detect_collisions(pos_A, np.empty((0, 2)), r_c=1.0)
    print(f"Test 2 — sans particule B : {len(paires_vides)} paire")
    assert paires_vides.shape == (0, 2)

    # Test 3 — comparaison avec scipy.spatial.distance.cdist sur un tirage aléatoire
    rng = np.random.default_rng(42)
    pos_A = rng.uniform(0, 100, size=(500, 2))
    pos_B = rng.uniform(0, 100, size=(500, 2))
    r_c = 2.0
    paires = detect_collisions(pos_A, pos_B, r_c)
    attendu = np.argwhere(cdist(pos_A, pos_B) < r_c)
    print(f"Test 3 — 500 A et 500 B aléatoires, r_c = {r_c} µm : {len(paires)} paires, "
          f"identiques à cdist : {np.array_equal(paires, attendu)}")
    assert np.array_equal(paires, attendu)

    # Ordre de grandeur attendu pour le test 3 : chaque paire (A, B) est en collision
    # avec la probabilité pi·r_c²/L² (loin des bords), d'où N_A·N_B·pi·r_c²/L² paires.
    print(f"         nombre de paires attendu en moyenne ≈ {500 * 500 * np.pi * r_c**2 / 100**2:.0f}")

    print("Tous les tests passent.")
