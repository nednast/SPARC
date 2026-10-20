"""
Point d'entrée de la simulation SPARC : boucle temporelle.

Lancement depuis la racine du dépôt :
    python src/run_simulation.py                 # simulation de référence et vérifications (~8 s)
    python src/run_simulation.py --convergence   # étude de convergence en dt et en N (issue #9)
"""

import argparse
import time

import numpy as np

from analysis import (plot_snapshot, plot_populations, plot_convergence_dt, plot_convergence_n,
                      count_species, relative_gaps, save_populations, load_populations, save_table)
from config import (SEED, L, N_A, N_B, D, DT, N_STEPS, R_C, P, A, B, C,
                    FACTEURS_DT, N_RUNS_DT, N_TOTAUX, N_RUNS_N)
from particles import init_particles, step
from reactions import reaction_step
from theory import rate_constant, border_factor, solve_mass_action


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
    comptages : ndarray (n_steps + 1, 3) d'entiers
        Nombres de particules [N_A, N_B, N_C] à chaque pas, de l'instant initial (ligne 0)
        au dernier pas (ligne n_steps). L'instant du pas n est n·dt.
    historique : list de tuples (n, positions, labels)
        États enregistrés, n étant le numéro du pas (liste vide si save_every est None).
        Le nombre de particules diminue au fil des réactions, d'où une copie par instant.
    """
    comptages = np.zeros((n_steps + 1, 3), dtype=int)
    comptages[0] = count_species(labels)
    historique = [(0, positions.copy(), labels.copy())] if save_every else []
    for n in range(1, n_steps + 1):
        positions = step(positions, D, dt, L, rng)
        if p > 0:
            positions, labels, _ = reaction_step(positions, labels, r_c, p, rng)
        comptages[n] = count_species(labels)
        if save_every and n % save_every == 0:
            historique.append((n, positions.copy(), labels.copy()))
    return positions, labels, comptages, historique


def run_replicas(n_a, n_b, L, D, dt, n_steps, r_c, p, n_runs, rng):
    """
    Lance n_runs simulations indépendantes (nouvelles positions initiales à chaque fois).

    Paramètres
    ----------
    n_a, n_b : int
        Nombres initiaux de particules A et B.
    L : float
        Côté de la boîte carrée (µm).
    D : float
        Coefficient de diffusion (µm²/s).
    dt : float
        Pas de temps (s).
    n_steps : int
        Nombre de pas de temps de chaque simulation.
    r_c : float
        Rayon de réaction (µm).
    p : float
        Probabilité de réaction d'une paire en collision par pas de temps.
    n_runs : int
        Nombre de simulations.
    rng : numpy.random.Generator
        Générateur aléatoire, partagé par toutes les simulations.

    Retour
    ------
    trajectoires : ndarray (n_runs, n_steps + 1)
        N_A à chaque pas, pour chaque simulation.
    duree : float
        Temps de calcul moyen d'une simulation (s, temps réel de la machine).
    """
    trajectoires = np.zeros((n_runs, n_steps + 1))
    debut = time.perf_counter()
    for r in range(n_runs):
        positions, labels = init_particles(n_a, n_b, L, rng)
        _, _, comptages, _ = run_simulation(n_steps, positions, labels, D, dt, L, r_c, p, rng)
        trajectoires[r] = comptages[:, A]
    return trajectoires, (time.perf_counter() - debut) / n_runs


def convergence_dt(rng):
    """
    Convergence en dt (issue #9) : simulations avec dt = DT / facteur, à p / dt constant.

    p est une probabilité par pas : en divisant dt par 2 sans toucher à p, on diviserait
    aussi k par 2 et on changerait la physique. On divise donc p par le même facteur,
    pour que k = p·π·r_c²/dt reste identique. La durée simulée (N_STEPS·DT) est la même
    pour tous les dt, et on compare N_A aux instants communs multiples de DT.

    Écrit resultats/convergence_dt.csv et figures/fig_convergence_dt.png.
    """
    t = np.arange(N_STEPS + 1) * DT  # instants communs (s)
    k = rate_constant(P, R_C, DT)
    c0 = np.array([N_A, N_B, 0]) / L ** 2
    n_theorie = solve_mass_action(c0, k, t)[:, A] * L ** 2
    n_bord = solve_mass_action(c0, k * border_factor(R_C, L), t)[:, A] * L ** 2
    biais_bord = (n_bord - n_theorie) / n_theorie

    print(f"Convergence en dt : N_A = N_B = {N_A}, L = {L:g} µm, k = {k:.4f} µm²/s, "
          f"{N_RUNS_DT} simulations par dt, biais de bord prédit à t = {t[-1]:g} s : {biais_bord[-1]:+.2%}")
    resultats = []
    for facteur in FACTEURS_DT:
        dt, p = DT / facteur, P / facteur
        trajectoires, duree = run_replicas(N_A, N_B, L, D, dt, N_STEPS * facteur, R_C, p, N_RUNS_DT, rng)
        ecarts = relative_gaps(trajectoires[:, ::facteur], n_theorie)  # un point sur `facteur` : instants de t
        resultats.append({"dt": dt, "p": p, "duree": duree, **ecarts})
        print(f"  dt = {dt:<7g} s, p = {p:<8g} : écart final = {ecarts['moyen'][-1]:+.2%} "
              f"± {ecarts['incertitude'][-1]:.2%}, écart quadratique moyen = {ecarts['rms']:.2%}, "
              f"{duree:.1f} s par simulation")

    params = {"N_A": N_A, "N_B": N_B, "L": f"{L:g} µm", "D": f"{D:g} µm²/s", "r_c": f"{R_C:g} µm",
              "k": f"{k:.3g} µm²/s", "simulations par dt": N_RUNS_DT, "SEED": SEED}
    tableau = [[r["dt"], r["p"], r["moyen"][-1], r["incertitude"][-1], biais_bord[-1], r["rms"],
                r["rms_incertitude"], r["duree"]] for r in resultats]
    chemin = save_table("convergence_dt", ["dt (s)", "p", "ecart final", "incertitude ecart final",
                                           "biais de bord predit", "ecart rms", "incertitude ecart rms",
                                           "duree par simulation (s)"], tableau, params)
    print(f"  Tableau enregistré : {chemin}")
    chemin = plot_convergence_dt(t, resultats, biais_bord, params, nom="fig_convergence_dt")
    print(f"  Figure enregistrée : {chemin}")


def convergence_n(rng):
    """
    Convergence en N (issue #9) : N = N_A + N_B variable, avec N_A = N_B.

    La concentration initiale est gardée égale à celle de la configuration de référence
    (N_A / L² = 0.05 µm⁻²) : la boîte est agrandie, L = sqrt(N_A / c0). Ainsi la
    loi d'action de masse prédit la même évolution des concentrations pour tous les N,
    et seul le nombre de particules change.

    Écrit resultats/convergence_N.csv et figures/fig_convergence_N.png.
    """
    t = np.arange(N_STEPS + 1) * DT  # instants (s)
    k = rate_constant(P, R_C, DT)
    c_initiale = N_A / L ** 2  # concentration initiale de A et de B (µm⁻²)

    print(f"Convergence en N : [A]₀ = [B]₀ = {c_initiale:g} µm⁻², dt = {DT:g} s, k = {k:.4f} µm²/s")
    resultats = []
    for n_total, n_runs in zip(N_TOTAUX, N_RUNS_N):
        n_a = n_total // 2
        L_n = np.sqrt(n_a / c_initiale)  # côté de la boîte (µm)
        c0 = np.array([n_a, n_a, 0]) / L_n ** 2
        n_theorie = solve_mass_action(c0, k, t)[:, A] * L_n ** 2
        n_bord = solve_mass_action(c0, k * border_factor(R_C, L_n), t)[:, A] * L_n ** 2
        trajectoires, duree = run_replicas(n_a, n_a, L_n, D, DT, N_STEPS, R_C, P, n_runs, rng)
        ecarts = relative_gaps(trajectoires, n_theorie)
        resultats.append({"N": n_total, "L": L_n, "n_runs": n_runs, "duree": duree,
                          "rms": ecarts["rms"], "rms_incertitude": ecarts["rms_incertitude"],
                          "final": ecarts["moyen"][-1], "final_incertitude": ecarts["incertitude"][-1],
                          "biais_bord": n_bord[-1] / n_theorie[-1] - 1})
        print(f"  N = {n_total:<5d} (L = {L_n:5.1f} µm, {n_runs} simulations) : "
              f"écart quadratique moyen = {ecarts['rms']:.2%} ± {ecarts['rms_incertitude']:.2%}, "
              f"écart final = {ecarts['moyen'][-1]:+.2%} ± {ecarts['incertitude'][-1]:.2%} "
              f"(bord prédit : {resultats[-1]['biais_bord']:+.2%}), {duree:.1f} s par simulation")

    params = {"[A]₀ = [B]₀": f"{c_initiale:g} µm⁻²", "D": f"{D:g} µm²/s", "dt": f"{DT:g} s",
              "r_c": f"{R_C:g} µm", "p": P, "k": f"{k:.3g} µm²/s", "SEED": SEED}
    tableau = [[r["N"], r["L"], r["n_runs"], r["rms"], r["rms_incertitude"], r["final"],
                r["final_incertitude"], r["biais_bord"], r["duree"]] for r in resultats]
    chemin = save_table("convergence_N", ["N", "L (um)", "simulations", "ecart rms", "incertitude ecart rms",
                                          "ecart final", "incertitude ecart final", "biais de bord predit",
                                          "duree par simulation (s)"], tableau, params)
    print(f"  Tableau enregistré : {chemin}")
    chemin = plot_convergence_n(resultats, t[-1], params, nom="fig_convergence_N")
    print(f"  Figure enregistrée : {chemin}")


def convergence_study():
    """Étude de convergence complète (issue #9) : en dt puis en N. Durée : plusieurs minutes."""
    rng = np.random.default_rng(SEED)
    convergence_dt(rng)
    convergence_n(rng)


def check_layer1():
    """
    Simulation de référence de la couche 1 et vérifications : confinement, conservation,
    comparaison à la loi d'action de masse, instantanés, déplacement quadratique moyen.
    """
    rng = np.random.default_rng(SEED)
    positions_init, labels_init = init_particles(N_A, N_B, L, rng)

    # Simulation avec réaction, état enregistré à chaque pas pour les vérifications
    _, _, comptages, historique = run_simulation(N_STEPS, positions_init, labels_init, D, DT, L,
                                      R_C, P, rng, save_every=1)

    # Test 1 — confinement : aucune coordonnée hors de la boîte, à aucun pas
    n_hors_max = max(np.sum((pos < 0) | (pos > L)) for _, pos, _ in historique)
    print(f"Confinement sur {N_STEPS} pas ({N_STEPS * DT:.0f} s) : "
          f"max de coordonnées hors de [0, L] = {n_hors_max}")

    # Test 2 — conservation de la matière : chaque C contient un A et un B,
    # donc N_A + N_C et N_B + N_C restent constants
    n_a, n_b, n_c = comptages[:, A], comptages[:, B], comptages[:, C]
    ecart_max = max(np.max(np.abs(n_a + n_c - N_A)), np.max(np.abs(n_b + n_c - N_B)))
    print(f"Conservation : écart maximal sur N_A + N_C et N_B + N_C = {ecart_max}")

    # Populations à quelques instants
    for n in (0, 300, N_STEPS):
        print(f"  t = {n * DT:5.1f} s : A = {n_a[n]}, B = {n_b[n]}, C = {n_c[n]}")

    # Sauvegarde de l'historique des populations, puis relecture pour vérifier le fichier
    params = {"N_A": N_A, "N_B": N_B, "L": f"{L:g} µm", "D": f"{D:g} µm²/s",
              "dt": f"{DT:g} s", "n_steps": N_STEPS, "r_c": f"{R_C:g} µm", "p": P, "SEED": SEED}
    t = np.arange(N_STEPS + 1) * DT
    chemin = save_populations("populations_couche1", t, comptages, L, params)
    t_relu, comptages_relus = load_populations("populations_couche1")
    print(f"Populations enregistrées : {chemin} — relecture identique : "
          f"{np.allclose(t_relu, t) and np.array_equal(comptages_relus, comptages)}")

    # Test 3 — comparaison à la loi d'action de masse (issue #8) : k = p·π·r_c²/dt,
    # résolution par solve_ivp en concentrations (µm⁻²), puis retour en nombres de particules (× L²)
    k = rate_constant(P, R_C, DT)
    c0 = comptages[0] / L ** 2
    comptages_theorie = solve_mass_action(c0, k, t) * L ** 2
    ecart_rel = np.abs(n_a - comptages_theorie[:, A]) / comptages_theorie[:, A]
    print(f"Loi d'action de masse : k = {k:.4f} µm²/s, "
          f"N_A(t = {N_STEPS * DT:.0f} s) simulé = {n_a[-1]}, théorique = {comptages_theorie[-1, A]:.1f}")
    print(f"  écart relatif sur N_A : moyen = {ecart_rel.mean():.1%}, max = {ecart_rel.max():.1%}")
    chemin = plot_populations(t, comptages, comptages_theorie,
                              params=params | {"k": f"{k:.3g} µm²/s"},
                              nom="fig_populations_couche1")
    print(f"Figure enregistrée : {chemin}")

    # Instantanés : début, après 300 pas, fin
    for n in (0, 300, N_STEPS):
        _, pos, lab = historique[n]
        chemin = plot_snapshot(pos, lab, n * DT, L, params=params)
        print(f"Instantané enregistré : {chemin}")

    # Test 4 — validation de la diffusion (sans réaction, p = 0) : en 2D, le déplacement
    # quadratique moyen vaut <r²> = 4·D·t en milieu libre. On le mesure sur une durée courte
    # (déplacement typique sqrt(4Dt) ≈ 6 µm << L) et uniquement pour les particules
    # parties à plus de 15 µm des parois, que les parois n'ont presque pas pu atteindre.
    n_court = 100
    t_court = n_court * DT
    pos_fin, _, _, _ = run_simulation(n_court, positions_init, labels_init, D, DT, L, R_C, 0.0, rng)
    r2 = np.sum((pos_fin - positions_init) ** 2, axis=1)
    loin = np.all((positions_init > 15) & (positions_init < L - 15), axis=1)
    msd = r2[loin].mean()
    erreur_stat = r2[loin].std() / np.sqrt(loin.sum())  # écart-type de la moyenne
    theorie = 4 * D * t_court
    print(f"Déplacement quadratique moyen après {t_court:.1f} s ({loin.sum()} particules) : "
          f"mesuré = {msd:.1f} ± {erreur_stat:.1f} µm², théorique 4Dt = {theorie:.1f} µm², "
          f"écart = {abs(msd - theorie) / erreur_stat:.1f} écart-type(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulation SPARC de la réaction A + B → C.")
    parser.add_argument("--convergence", action="store_true",
                        help="lance l'étude de convergence en dt et en N (issue #9, une quinzaine de minutes)")
    if parser.parse_args().convergence:
        convergence_study()
    else:
        check_layer1()
