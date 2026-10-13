"""
Point d'entrée de la simulation SPARC : boucle temporelle.

Lancement depuis la racine du dépôt :
    python src/run_simulation.py
"""

import numpy as np

from analysis import plot_snapshot
from config import SEED, L, N_A, N_B, D, DT, N_STEPS, R_C, P, A, B, C
from particles import init_particles, step
from reactions import reaction_step


def run_simulation(n_steps, positions, labels, D, dt, L, r_c, p, rng, save_every=None):
    """
    Fait évoluer le système pendant n_steps pas de temps.

    À chaque pas : déplacement brownien de toutes les particules (avec parois
    réfléchissantes), puis réaction A + B → C pour les paires en collision.

    Paramètres
    ----------
    n_steps : int
        Nombre de pas de temps.
    positions : ndarray (N, 2)
        Positions initiales (µm).
    labels : ndarray (N,)
        Espèces initiales (A, B ou C).
    D : float
        Coefficient de diffusion (µm²/s).
    dt : float
        Pas de temps (s).
    L : float
        Côté de la boîte carrée (µm).
    r_c : float
        Rayon de réaction (µm).
    p : float
        Probabilité de réaction d'une paire en collision par pas de temps (p = 0 : diffusion seule).
    rng : numpy.random.Generator
        Générateur aléatoire.
    save_every : int ou None
        Si renseigné, l'état est enregistré tous les save_every pas (ainsi qu'à l'instant initial).

    Retour
    ------
    positions, labels : état final.
    historique : list de tuples (n, positions, labels)
        États enregistrés, n étant le numéro du pas (liste vide si save_every est None).
        Le nombre de particules diminue au fil des réactions, d'où une copie par instant.
    """
    historique = [(0, positions.copy(), labels.copy())] if save_every else []
    for n in range(1, n_steps + 1):
        positions = step(positions, D, dt, L, rng)
        if p > 0:
            positions, labels, _ = reaction_step(positions, labels, r_c, p, rng)
        if save_every and n % save_every == 0:
            historique.append((n, positions.copy(), labels.copy()))
    return positions, labels, historique


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    positions_init, labels_init = init_particles(N_A, N_B, L, rng)

    # Simulation avec réaction, état enregistré à chaque pas pour les vérifications
    _, _, historique = run_simulation(N_STEPS, positions_init, labels_init, D, DT, L,
                                      R_C, P, rng, save_every=1)

    # Test 1 — confinement : aucune coordonnée hors de la boîte, à aucun pas
    n_hors_max = max(np.sum((pos < 0) | (pos > L)) for _, pos, _ in historique)
    print(f"Confinement sur {N_STEPS} pas ({N_STEPS * DT:.0f} s) : "
          f"max de coordonnées hors de [0, L] = {n_hors_max}")

    # Test 2 — conservation de la matière : chaque C contient un A et un B,
    # donc N_A + N_C et N_B + N_C restent constants
    ecart_max = max(
        max(abs(np.sum(lab == A) + np.sum(lab == C) - N_A),
            abs(np.sum(lab == B) + np.sum(lab == C) - N_B))
        for _, _, lab in historique)
    print(f"Conservation : écart maximal sur N_A + N_C et N_B + N_C = {ecart_max}")

    # Populations à quelques instants
    for n in (0, 300, N_STEPS):
        lab = historique[n][2]
        print(f"  t = {n * DT:5.1f} s : A = {np.sum(lab == A)}, B = {np.sum(lab == B)}, "
              f"C = {np.sum(lab == C)}")

    # Instantanés : début, après 300 pas, fin
    params = {"N_A": N_A, "N_B": N_B, "L": f"{L:g} µm", "D": f"{D:g} µm²/s",
              "dt": f"{DT:g} s", "r_c": f"{R_C:g} µm", "p": P}
    for n in (0, 300, N_STEPS):
        _, pos, lab = historique[n]
        chemin = plot_snapshot(pos, lab, n * DT, L, params=params)
        print(f"Instantané enregistré : {chemin}")

    # Test 3 — validation de la diffusion (sans réaction, p = 0) : en 2D, le déplacement
    # quadratique moyen vaut <r²> = 4·D·t en milieu libre. On le mesure sur une durée courte
    # (déplacement typique sqrt(4Dt) ≈ 6 µm << L) et uniquement pour les particules
    # parties à plus de 15 µm des parois, que les parois n'ont presque pas pu atteindre.
    n_court = 100
    t_court = n_court * DT
    pos_fin, _, _ = run_simulation(n_court, positions_init, labels_init, D, DT, L, R_C, 0.0, rng)
    r2 = np.sum((pos_fin - positions_init) ** 2, axis=1)
    loin = np.all((positions_init > 15) & (positions_init < L - 15), axis=1)
    msd = r2[loin].mean()
    erreur_stat = r2[loin].std() / np.sqrt(loin.sum())  # écart-type de la moyenne
    theorie = 4 * D * t_court
    print(f"Déplacement quadratique moyen après {t_court:.1f} s ({loin.sum()} particules) : "
          f"mesuré = {msd:.1f} ± {erreur_stat:.1f} µm², théorique 4Dt = {theorie:.1f} µm², "
          f"écart = {abs(msd - theorie) / erreur_stat:.1f} écart-type(s)")
