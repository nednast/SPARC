"""
Génération des particules et déplacement brownien dans la boîte de simulation.

Représentation de l'état du système :
- positions : tableau NumPy (N, 2) de flottants, colonnes x et y en µm,
              avec 0 <= x <= L et 0 <= y <= L
- labels    : tableau NumPy (N,) d'entiers, code de l'espèce de chaque
              particule (A, B ou C, voir config.py)

La particule i a pour position positions[i] et pour espèce labels[i].
"""

import numpy as np

from config import A, B


def init_particles(n_a, n_b, L, rng):
    """
    Place n_a particules A et n_b particules B uniformément dans la boîte [0, Lx[ x [0, Ly[.

    Paramètres
    ----------
    n_a, n_b : int
        Nombres initiaux de particules A et B.
    L : float ou array-like (2,)
        Côté de la boîte carrée (µm), ou dimensions (Lx, Ly) d'une boîte rectangulaire (µm).
    rng : numpy.random.Generator
        Générateur aléatoire (créé avec np.random.default_rng(SEED)).

    Retour
    ------
    positions : ndarray (n_a + n_b, 2)
        Coordonnées (x, y) en µm.
    labels : ndarray (n_a + n_b,)
        Espèce de chaque particule : les n_a premières sont A, les n_b suivantes sont B.
    """
    if n_a < 0 or n_b < 0:
        raise ValueError("Les nombres de particules doivent être positifs ou nuls.")
    if np.any(np.asarray(L) <= 0):
        raise ValueError("La taille de la boîte L doit être strictement positive.")

    n = n_a + n_b
    positions = rng.uniform(0.0, L, size=(n, 2))  # avec L = (Lx, Ly) : x dans [0, Lx[, y dans [0, Ly[
    labels = np.concatenate([
        np.full(n_a, A, dtype=np.int8),
        np.full(n_b, B, dtype=np.int8),
    ])
    return positions, labels


def reflect(positions, L):
    """
    Applique les conditions aux limites réfléchissantes sur les 4 parois de [0, Lx] x [0, Ly].

    Une particule qui dépasse une paroi est renvoyée symétriquement à l'intérieur,
    comme un miroir : x < 0 devient -x, et x > L devient 2L - x.
    Pour traiter aussi le cas (rare) d'un saut plus grand que L, on replie la
    coordonnée modulo 2L : le mouvement réfléchi entre deux miroirs est périodique
    de période 2L, et sur [L, 2L[ il redescend de L vers 0.
    Avec L = (Lx, Ly), NumPy applique Lx à la colonne x et Ly à la colonne y (broadcasting).

    Paramètres
    ----------
    positions : ndarray (N, 2)
        Positions (µm), éventuellement hors de la boîte.
    L : float ou array-like (2,)
        Côté de la boîte carrée (µm), ou dimensions (Lx, Ly) d'une boîte rectangulaire (µm).

    Retour
    ------
    ndarray (N, 2)
        Nouvelles positions (µm), toutes dans la boîte.
    """
    L = np.asarray(L)
    positions = np.mod(positions, 2 * L)      # ramène dans [0, 2L[
    # sur ]L, 2L[ (partie « retour » du mouvement), x devient 2L - x
    return np.where(positions > L, 2 * L - positions, positions)


def step(positions, D, dt, L, rng):
    """
    Effectue un pas de temps de diffusion brownienne, puis applique les parois.

    Chaque coordonnée reçoit un déplacement indépendant Δx = sqrt(2·D·dt)·ξ,
    avec ξ ~ N(0, 1). C'est le schéma d'Euler-Maruyama de l'équation de
    Langevin sans force : la variance du déplacement par axe vaut 2·D·dt.

    Paramètres
    ----------
    positions : ndarray (N, 2)
        Positions actuelles (µm).
    D : float
        Coefficient de diffusion (µm²/s).
    dt : float
        Pas de temps (s).
    L : float ou array-like (2,)
        Côté de la boîte carrée (µm), ou dimensions (Lx, Ly) d'une boîte rectangulaire (µm).
    rng : numpy.random.Generator
        Générateur aléatoire.

    Retour
    ------
    ndarray (N, 2)
        Positions après le pas de temps (µm), dans la boîte.
    """
    deplacements = np.sqrt(2 * D * dt) * rng.standard_normal(positions.shape)
    return reflect(positions + deplacements, L)


if __name__ == "__main__":
    # Démonstration : génère l'état initial et l'enregistre sous forme de nuage de points.
    from analysis import plot_snapshot
    from config import SEED, L as L_BOITE, N_A, N_B

    rng = np.random.default_rng(SEED)
    positions, labels = init_particles(N_A, N_B, L_BOITE, rng)

    print(f"{len(positions)} particules générées "
          f"({np.sum(labels == A)} A, {np.sum(labels == B)} B)")
    print(f"x dans [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}] µm, "
          f"y dans [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}] µm")

    chemin = plot_snapshot(positions, labels, 0.0, L_BOITE,
                           params={"N_A": N_A, "N_B": N_B, "L": f"{L_BOITE:g} µm"},
                           nom="tmp_snapshot_demo_initial")  # nom fixe : réécrit à chaque lancement
    print(f"Figure enregistrée : {chemin}")

    # Test 1 — reflect dans une boîte rectangulaire Lx = 200 µm, Ly = 50 µm, cas calculés à la main :
    #   (210, -3) -> x = 2·200 - 210 = 190, y = 3 ; (5, 60) -> y = 2·50 - 60 = 40 ;
    #   (-450, 120) : saut plus grand que la boîte -> x = 50 (mod 400), y = 20 (mod 100)
    boite = (200.0, 50.0)
    reflechies = reflect(np.array([[210.0, -3.0], [5.0, 60.0], [-450.0, 120.0]]), boite)
    print(f"Test 1 — reflect dans une boîte 200 x 50 µm : {reflechies.tolist()}")
    assert np.allclose(reflechies, [[190.0, 3.0], [5.0, 40.0], [50.0, 20.0]])

    # Test 2 — 1000 pas de diffusion dans la boîte rectangulaire : aucune particule ne sort,
    # et les x dépassent bien 50 µm (Lx et Ly ne sont pas confondus)
    pos_rect, _ = init_particles(200, 200, boite, rng)
    for _ in range(1000):
        pos_rect = step(pos_rect, 10.0, 0.01, boite, rng)
    n_hors = np.sum((pos_rect < 0) | (pos_rect > boite))
    print(f"Test 2 — 1000 pas dans la boîte 200 x 50 µm : {n_hors} coordonnée(s) hors de la boîte, "
          f"x max = {pos_rect[:, 0].max():.1f} µm, y max = {pos_rect[:, 1].max():.1f} µm")
    assert n_hors == 0
    assert pos_rect[:, 0].max() > 50

    print("Tous les tests passent.")
