"""
Détection des collisions et règles de réaction.
"""

import numpy as np

from config import A, B, C


def detect_collisions(pos_A, pos_B, r_c):
    """
    Renvoie toutes les paires (i, j) telles que la distance entre la particule A n°i
    et la particule B n°j est strictement inférieure à r_c.

    Méthode de la grille de cellules : on découpe le plan en cellules carrées de côté r_c.
    Deux particules à distance < r_c sont alors dans la même cellule ou dans deux cellules
    voisines (écart d'au plus 1 cellule selon x et selon y). Pour chaque A, on ne calcule
    donc la distance qu'aux B des 9 cellules qui l'entourent, au lieu de tous les B.
    Coût à peu près proportionnel à N_A + N_B (au lieu de N_A x N_B pour la version naïve).

    Étapes :
    1. numéro de cellule (cx, cy) de chaque particule, résumé en une clé entière ;
    2. les B sont triés par clé : les B d'une même cellule sont côte à côte, et
       np.searchsorted donne pour chaque cellule la plage [debut, fin[ de ses B ;
    3. pour chacun des 9 décalages (dx, dy), on liste les paires candidates (A, B de la
       cellule voisine) puis on garde celles à distance < r_c (distances au carré,
       pour éviter les racines).
    Les paires sont renvoyées triées par i puis par j, dans le même ordre que la version
    naïve : la simulation consomme donc les tirages aléatoires dans le même ordre.

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
    if len(pos_A) == 0 or len(pos_B) == 0:
        return np.empty((0, 2), dtype=int)

    # 1. Cellules, comptées à partir du coin inférieur gauche du nuage : numéros >= 0
    origine = np.minimum(pos_A.min(axis=0), pos_B.min(axis=0))
    cellules_A = np.floor((pos_A - origine) / r_c).astype(int)  # (N_A, 2) : (cx, cy)
    cellules_B = np.floor((pos_B - origine) / r_c).astype(int)
    # clé = (cx + 1)·largeur + (cy + 1) ; le +1 et la marge de largeur évitent qu'un voisin
    # en cy - 1 ou cy + 1 tombe sur une clé de la colonne cx voisine
    largeur = max(cellules_A[:, 1].max(), cellules_B[:, 1].max()) + 3
    cles_B = (cellules_B[:, 0] + 1) * largeur + (cellules_B[:, 1] + 1)

    # 2. B triés par cellule
    ordre_B = np.argsort(cles_B, kind="stable")
    cles_B_triees = cles_B[ordre_B]

    # 3. Paires candidates avec chacune des 9 cellules voisines
    liste_i, liste_j = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cles_voisines = (cellules_A[:, 0] + 1 + dx) * largeur + (cellules_A[:, 1] + 1 + dy)
            debut = np.searchsorted(cles_B_triees, cles_voisines, side="left")
            fin = np.searchsorted(cles_B_triees, cles_voisines, side="right")
            nombre = fin - debut  # nombre de B dans la cellule voisine de chaque A
            # Le A n°i est répété nombre[i] fois ; ses candidats sont aux places
            # debut[i], debut[i] + 1, ..., fin[i] - 1 du tableau trié.
            i = np.repeat(np.arange(len(pos_A)), nombre)
            rang = np.arange(nombre.sum()) - np.repeat(np.cumsum(nombre) - nombre, nombre)
            liste_i.append(i)
            liste_j.append(ordre_B[np.repeat(debut, nombre) + rang])
    i = np.concatenate(liste_i)
    j = np.concatenate(liste_j)

    proches = np.sum((pos_A[i] - pos_B[j]) ** 2, axis=1) < r_c ** 2  # distances au carré (µm²)
    i, j = i[proches], j[proches]
    tri = np.lexsort((j, i))  # tri par i, puis par j à i égal
    return np.column_stack((i[tri], j[tri]))


def react(pairs, p, rng):
    """
    Tire au sort les paires en collision qui réagissent, sans qu'une particule réagisse deux fois.

    1. Chaque paire réagit indépendamment avec la probabilité p.
    2. Si une particule figure dans plusieurs paires retenues (un A proche de deux B,
       par exemple), on ne garde qu'une de ces paires. Les paires retenues sont
       parcourues dans un ordre aléatoire, pour ne pas favoriser les petits indices.

    Paramètres
    ----------
    pairs : ndarray (n_paires, 2) d'entiers
        Paires [i, j] renvoyées par detect_collisions.
    p : float
        Probabilité de réaction d'une paire par pas de temps (entre 0 et 1).
    rng : numpy.random.Generator
        Générateur aléatoire.

    Retour
    ------
    ndarray (n_reactions, 2) d'entiers
        Paires qui réagissent ; chaque i et chaque j y apparaît au plus une fois.
    """
    tirees = pairs[rng.random(len(pairs)) < p]
    tirees = tirees[rng.permutation(len(tirees))]

    deja_utilises_A, deja_utilises_B = set(), set()
    retenues = []
    for i, j in tirees:
        if i not in deja_utilises_A and j not in deja_utilises_B:
            retenues.append((i, j))
            deja_utilises_A.add(i)
            deja_utilises_B.add(j)
    return np.array(retenues, dtype=int).reshape(-1, 2)


def reaction_step(positions, labels, r_c, p, rng):
    """
    Applique la réaction A + B → C pendant un pas de temps.

    Pour chaque paire qui réagit : la particule A devient une particule C placée au
    milieu du segment [A, B], et la particule B est supprimée. C'est équivalent à
    « retirer A et B puis ajouter C », sans avoir à agrandir les tableaux.

    Paramètres
    ----------
    positions : ndarray (N, 2)
        Positions (µm).
    labels : ndarray (N,)
        Espèces (A, B ou C).
    r_c : float
        Rayon de réaction (µm).
    p : float
        Probabilité de réaction d'une paire en collision par pas de temps.
    rng : numpy.random.Generator
        Générateur aléatoire.

    Retour
    ------
    positions : ndarray (N - n_reactions, 2)
    labels : ndarray (N - n_reactions,)
    n_reactions : int
        Nombre de réactions pendant ce pas.
    """
    indices_A = np.flatnonzero(labels == A)  # indices globaux des particules A
    indices_B = np.flatnonzero(labels == B)
    paires = detect_collisions(positions[indices_A], positions[indices_B], r_c)
    reactions = react(paires, p, rng)
    if len(reactions) == 0:
        return positions, labels, 0

    # Passage des indices locaux (dans pos_A, pos_B) aux indices globaux
    globaux_A = indices_A[reactions[:, 0]]
    globaux_B = indices_B[reactions[:, 1]]

    positions = positions.copy()
    labels = labels.copy()
    positions[globaux_A] = (positions[globaux_A] + positions[globaux_B]) / 2
    labels[globaux_A] = C
    garder = np.ones(len(labels), dtype=bool)
    garder[globaux_B] = False
    return positions[garder], labels[garder], len(reactions)


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

    # Test 3 — comparaison avec scipy.spatial.distance.cdist (toutes les distances) sur des
    # tirages aléatoires ; r_c = 7.3 µm ne divise pas L, r_c = 60 µm donne de très grandes cellules
    rng = np.random.default_rng(42)
    pos_A = rng.uniform(0, 100, size=(500, 2))
    pos_B = rng.uniform(0, 100, size=(500, 2))
    for r_c in (2.0, 7.3, 60.0):
        paires = detect_collisions(pos_A, pos_B, r_c)
        attendu = np.argwhere(cdist(pos_A, pos_B) < r_c)
        print(f"Test 3 — 500 A et 500 B aléatoires, r_c = {r_c} µm : {len(paires)} paires, "
              f"identiques à cdist : {np.array_equal(paires, attendu)}")
        assert np.array_equal(paires, attendu)
    r_c = 2.0

    # Ordre de grandeur attendu pour le test 3 : chaque paire (A, B) est en collision
    # avec la probabilité pi·r_c²/L² (loin des bords), d'où N_A·N_B·pi·r_c²/L² paires.
    print(f"         nombre de paires attendu en moyenne ≈ {500 * 500 * np.pi * r_c**2 / 100**2:.0f}")

    # Test 4 — react avec p = 1 : un A en collision avec deux B ne réagit qu'une fois
    paires = np.array([[0, 0], [0, 1], [1, 1], [2, 2]])
    retenues = react(paires, 1.0, rng)
    print(f"Test 4 — paires {paires.tolist()} avec p = 1 -> retenues {retenues.tolist()}")
    assert len(set(retenues[:, 0])) == len(retenues) and len(set(retenues[:, 1])) == len(retenues)
    assert [2, 2] in retenues.tolist()  # paire sans conflit : toujours retenue

    # Test 5 — react avec p = 0 : aucune réaction
    assert len(react(paires, 0.0, rng)) == 0
    print("Test 5 — p = 0 : aucune réaction")

    # Test 6 — reaction_step sur un cas construit à la main, p = 1, r_c = 1 µm
    #   A0 = (0, 0) et B1 = (0.5, 0) réagissent -> C en (0.25, 0), B1 supprimé
    #   A2 = (10, 10) est seul -> reste A
    positions = np.array([[0.0, 0.0], [0.5, 0.0], [10.0, 10.0]])
    labels = np.array([A, B, A])
    positions, labels, n = reaction_step(positions, labels, 1.0, 1.0, rng)
    print(f"Test 6 — {n} réaction ; labels = {labels.tolist()}, positions = {positions.tolist()}")
    assert n == 1
    assert labels.tolist() == [C, A]
    assert np.allclose(positions, [[0.25, 0.0], [10.0, 10.0]])

    print("Tous les tests passent.")
