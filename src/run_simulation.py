"""
Point d'entrée de la simulation SPARC : boucle temporelle.

Lancement depuis la racine du dépôt :
    python src/run_simulation.py
"""

import numpy as np

from analysis import plot_snapshot
from config import SEED, L, N_A, N_B, D, DT, N_STEPS
from particles import init_particles, step


def run_simulation(n_steps, positions, D, dt, L, rng, save_every=None):
    """
    Fait évoluer les particules pendant n_steps pas de temps (diffusion seule, sans réaction).

    Paramètres
    ----------
    n_steps : int
        Nombre de pas de temps.
    positions : ndarray (N, 2)
        Positions initiales (µm).
    D : float
        Coefficient de diffusion (µm²/s).
    dt : float
        Pas de temps (s).
    L : float
        Côté de la boîte carrée (µm).
    rng : numpy.random.Generator
        Générateur aléatoire.
    save_every : int ou None
        Si renseigné, les positions sont enregistrées tous les save_every pas
        (ainsi qu'à l'instant initial).

    Retour
    ------
    positions : ndarray (N, 2)
        Positions finales (µm).
    historique : list de ndarray (N, 2)
        Positions enregistrées (liste vide si save_every est None).
    """
    historique = [positions.copy()] if save_every else []
    for n in range(1, n_steps + 1):
        positions = step(positions, D, dt, L, rng)
        if save_every and n % save_every == 0:
            historique.append(positions.copy())
    return positions, historique


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    positions_init, labels = init_particles(N_A, N_B, L, rng)
    n_particules = len(positions_init)

    # Test 1 — confinement : on vérifie à chaque pas qu'aucune particule n'est sortie
    # de la boîte et que le nombre de particules est conservé.
    _, historique = run_simulation(N_STEPS, positions_init, D, DT, L, rng, save_every=1)
    n_hors_max = max(np.sum((p < 0) | (p > L)) for p in historique)
    n_perdues_max = max(abs(len(p) - n_particules) for p in historique)
    print(f"Confinement sur {N_STEPS} pas ({N_STEPS * DT:.0f} s) : "
          f"max de coordonnées hors de [0, L] = {n_hors_max}, "
          f"max de particules perdues = {n_perdues_max}")

    # Instantanés au début et à la fin de la simulation
    params = {"N": n_particules, "L": f"{L:g} µm", "D": f"{D:g} µm²/s", "dt": f"{DT:g} s"}
    for n in (0, N_STEPS):
        chemin = plot_snapshot(historique[n], labels, n * DT, L, params=params)
        print(f"Instantané enregistré : {chemin}")

    # Test 2 — validation de la diffusion : en 2D, le déplacement quadratique moyen
    # vaut <r²> = 4·D·t en milieu libre. On le mesure sur une durée courte
    # (déplacement typique sqrt(4Dt) ≈ 6 µm << L) et uniquement pour les particules
    # parties à plus de 15 µm des parois, que les parois n'ont presque pas pu atteindre.
    n_court = 100
    t_court = n_court * DT
    pos_fin, _ = run_simulation(n_court, positions_init, D, DT, L, rng)
    r2 = np.sum((pos_fin - positions_init) ** 2, axis=1)
    loin = np.all((positions_init > 15) & (positions_init < L - 15), axis=1)
    msd = r2[loin].mean()
    erreur_stat = r2[loin].std() / np.sqrt(loin.sum())  # écart-type de la moyenne
    theorie = 4 * D * t_court
    print(f"Déplacement quadratique moyen après {t_court:.1f} s ({loin.sum()} particules) : "
          f"mesuré = {msd:.1f} ± {erreur_stat:.1f} µm², théorique 4Dt = {theorie:.1f} µm², "
          f"écart = {abs(msd - theorie) / erreur_stat:.1f} écart-type(s)")
