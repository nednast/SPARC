"""
Point d'entrée de la simulation SPARC : boucle temporelle.

Lancement depuis la racine du dépôt :
    python src/run_simulation.py                 # simulation de référence et vérifications (~8 s)
    python src/run_simulation.py --convergence   # étude de convergence en dt et en N (issue #9)
    python src/run_simulation.py --diffusion     # écart à l'action de masse en fonction de D
    python src/run_simulation.py --animation separe [--pas 50000] [--video fichier.mp4]
                                                 # animation en direct (ou enregistrée en vidéo)
"""

import argparse
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from analysis import (plot_snapshot, plot_populations, plot_convergence_dt, plot_convergence_n, plot_diffusion,
                      count_species, relative_gaps, save_populations, load_populations, save_table)
from config import (SEED, L, L_SEPARE, N_A, N_B, N_A_SEPARE, N_B_SEPARE, D, DT, N_STEPS, R_C, P, A, B, C, NOMS_ESPECES,
                    COULEURS_ESPECES, PAS_PAR_IMAGE,
                    FACTEURS_DT, N_RUNS_DT, N_TOTAUX, N_RUNS_N, VALEURS_D, N_TOTAL_D, N_RUNS_D,
                    FACTEURS_P, VALEURS_D_P)
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
        if n_total % 2:
            raise ValueError(f"N_TOTAUX doit contenir des nombres pairs (N_A = N_B) : {n_total}")
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


def diffusion_study():
    """
    Limitation par la diffusion : simulations à p, dt et k fixés, pour plusieurs valeurs de D.

    La loi d'action de masse ne fait pas intervenir D : elle prédit la même évolution pour
    tous les D. Mais k = p·π·r_c²/dt suppose une répartition uniforme des particules. Quand
    une paire A-B proche réagit, elle disparaît ; si la diffusion est lente, les voisins ne
    sont pas remplacés assez vite, il y a moins de paires en collision que prévu, et la
    réaction est plus lente que la théorie (N_A simulé au-dessus de N_A théorique).
    Cet écart doit donc diminuer quand D augmente. On compare N_A à la loi d'action de masse
    déjà corrigée du bord (k · border_factor), pour isoler l'effet de la diffusion.

    2e partie : variation de p à k·t_final constant (p = P·f pendant N_STEPS / f pas), pour
    D = 10 et 40 µm²/s. La théorie prédit alors le même N_A final pour tous les p. Si l'écart
    vient de la diffusion, il dépend de k/D, donc de p/D, et doit tendre vers 0 quand p → 0 ;
    un biais d'une autre origine resterait à une valeur fixe.

    Écrit resultats/diffusion.csv, figures/fig_diffusion.png et resultats/diffusion_p.csv.
    """
    rng = np.random.default_rng(SEED)
    t = np.arange(N_STEPS + 1) * DT  # instants (s)
    k = rate_constant(P, R_C, DT)
    c_initiale = N_A / L ** 2  # concentration de référence (µm⁻²)
    if N_TOTAL_D % 2:
        raise ValueError(f"N_TOTAL_D doit être pair (N_A = N_B) : {N_TOTAL_D}")
    n_a = N_TOTAL_D // 2
    L_d = np.sqrt(n_a / c_initiale)  # côté de la boîte (µm)
    c0 = np.array([n_a, n_a, 0]) / L_d ** 2
    n_bord = solve_mass_action(c0, k * border_factor(R_C, L_d), t)[:, A] * L_d ** 2

    print(f"Limitation par la diffusion : N = {N_TOTAL_D}, L = {L_d:.1f} µm, k = {k:.4f} µm²/s, "
          f"facteur de bord = {border_factor(R_C, L_d):.4f}, {N_RUNS_D} simulations par D")
    resultats = []
    for d in VALEURS_D:
        trajectoires, duree = run_replicas(n_a, n_a, L_d, d, DT, N_STEPS, R_C, P, N_RUNS_D, rng)
        ecarts = relative_gaps(trajectoires, n_bord)
        resultats.append({"D": d, "duree": duree, **ecarts})
        print(f"  D = {d:<5g} µm²/s : écart final = {ecarts['moyen'][-1]:+.2%} ± {ecarts['incertitude'][-1]:.2%}, "
              f"écart quadratique moyen = {ecarts['rms']:.2%} ± {ecarts['rms_incertitude']:.2%}, "
              f"{duree:.1f} s par simulation")

    params = {"N_A": n_a, "N_B": n_a, "L": f"{L_d:.0f} µm", "dt": f"{DT:g} s", "r_c": f"{R_C:g} µm",
              "p": P, "k": f"{k:.3g} µm²/s", "simulations par D": N_RUNS_D, "SEED": SEED}
    tableau = [[r["D"], r["moyen"][-1], r["incertitude"][-1], r["rms"], r["rms_incertitude"], r["duree"]]
               for r in resultats]
    chemin = save_table("diffusion", ["D (um2/s)", "ecart final", "incertitude ecart final", "ecart rms",
                                      "incertitude ecart rms", "duree par simulation (s)"], tableau, params)
    print(f"  Tableau enregistré : {chemin}")
    chemin = plot_diffusion(t, resultats, params, nom="fig_diffusion")
    print(f"  Figure enregistrée : {chemin}")

    # 2e partie : variation de p à k·t_final constant
    print(f"Variation de p à k·t_final constant : {N_RUNS_D} simulations par couple (D, p)")
    lignes = []
    for d in VALEURS_D_P:
        for facteur in FACTEURS_P:
            if (N_STEPS / facteur) % 1:
                raise ValueError(f"N_STEPS / facteur doit être entier : {N_STEPS} / {facteur}")
            n_steps, p = int(N_STEPS / facteur), P * facteur
            t_p = np.arange(n_steps + 1) * DT  # instants (s)
            k_p = rate_constant(p, R_C, DT)
            n_bord_p = solve_mass_action(c0, k_p * border_factor(R_C, L_d), t_p)[:, A] * L_d ** 2
            trajectoires, duree = run_replicas(n_a, n_a, L_d, d, DT, n_steps, R_C, p, N_RUNS_D, rng)
            ecarts = relative_gaps(trajectoires, n_bord_p)
            lignes.append([d, p, t_p[-1], p / d, ecarts["moyen"][-1], ecarts["incertitude"][-1], duree])
            print(f"  D = {d:<4g} µm²/s, p = {p:<6g} ({t_p[-1]:g} s simulées), p/D = {p / d:.2e} s/µm² : "
                  f"écart final = {ecarts['moyen'][-1]:+.2%} ± {ecarts['incertitude'][-1]:.2%}, "
                  f"{duree:.1f} s par simulation")

    params_p = {"N_A": n_a, "N_B": n_a, "L": f"{L_d:.0f} µm", "dt": f"{DT:g} s", "r_c": f"{R_C:g} µm",
                "k·t_final": f"{k * N_STEPS * DT:.3g} µm²", "simulations par couple": N_RUNS_D, "SEED": SEED}
    chemin = save_table("diffusion_p", ["D (um2/s)", "p", "duree simulee (s)", "p/D (s/um2)", "ecart final",
                                        "incertitude ecart final", "duree par simulation (s)"], lignes, params_p)
    print(f"  Tableau enregistré : {chemin}")


def check_layer1():
    """
    Simulation de référence de la couche 1 et vérifications : confinement, conservation,
    comparaison à la loi d'action de masse, instantanés, déplacement quadratique moyen.
    """
    rng = np.random.default_rng(SEED)
    positions_init, labels_init = init_particles(N_A, N_B, L, rng)

    # Simulation avec réaction. L'état est enregistré tous les 100 pas seulement : le garder à
    # chaque pas coûterait N_STEPS copies des positions (≈ 85 Mo pour N = 1000). L'état final
    # est ajouté à part, pour que l'instantané de fin existe même si N_STEPS n'est pas un multiple de 100.
    pos_final, lab_final, comptages, historique = run_simulation(N_STEPS, positions_init, labels_init,
                                                                 D, DT, L, R_C, P, rng, save_every=100)
    etats = {n: (pos, lab) for n, pos, lab in historique} | {N_STEPS: (pos_final, lab_final)}

    # Test 1 — confinement : aucune coordonnée hors de la boîte, aux instants enregistrés
    # (reflect garantit le confinement à chaque pas par construction, voir aussi le test 2 de particles.py)
    n_hors_max = max(np.sum((pos < 0) | (pos > L)) for pos, _ in etats.values())
    print(f"Confinement sur {len(etats)} instants répartis sur {N_STEPS * DT:.0f} s : "
          f"max de coordonnées hors de [0, L] = {n_hors_max}")
    assert n_hors_max == 0, "des particules sont sorties de la boîte"

    # Test 2 — conservation de la matière : chaque C contient un A et un B,
    # donc N_A + N_C et N_B + N_C restent constants
    n_a, n_b, n_c = comptages[:, A], comptages[:, B], comptages[:, C]
    ecart_max = max(np.max(np.abs(n_a + n_c - N_A)), np.max(np.abs(n_b + n_c - N_B)))
    print(f"Conservation : écart maximal sur N_A + N_C et N_B + N_C = {ecart_max}")
    assert ecart_max == 0, "la matière n'est pas conservée"

    # Populations à quelques instants
    for n in (0, 300, N_STEPS):
        print(f"  t = {n * DT:5.1f} s : A = {n_a[n]}, B = {n_b[n]}, C = {n_c[n]}")

    # Sauvegarde de l'historique des populations, puis relecture pour vérifier le fichier
    params = {"N_A": N_A, "N_B": N_B, "L": f"{L:g} µm", "D": f"{D:g} µm²/s",
              "dt": f"{DT:g} s", "n_steps": N_STEPS, "r_c": f"{R_C:g} µm", "p": P, "SEED": SEED}
    t = np.arange(N_STEPS + 1) * DT
    chemin = save_populations("populations_couche1", t, comptages, L, params)
    t_relu, comptages_relus = load_populations("populations_couche1")
    relecture_ok = np.allclose(t_relu, t) and np.array_equal(comptages_relus, comptages)
    print(f"Populations enregistrées : {chemin} — relecture identique : {relecture_ok}")
    assert relecture_ok, "le CSV relu diffère des populations simulées"

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

    # Instantanés : début, après 300 pas (multiple de 100, donc enregistré), fin
    for n in (0, 300, N_STEPS):
        pos, lab = etats[n]
        # nom fixe (brouillon tmp_, ignoré par git) : réécrit à chaque lancement au lieu de s'accumuler
        chemin = plot_snapshot(pos, lab, n * DT, L, params=params, nom=f"tmp_snapshot_couche1_t={n * DT:.2f}s")
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
    # 3 écarts-types : un écart plus grand n'arrive par hasard qu'environ 3 fois sur 1000
    assert abs(msd - theorie) < 3 * erreur_stat, "le déplacement quadratique moyen s'écarte de 4·D·t"


def animate(mode, n_steps, video=None):
    """
    Anime la simulation de référence : la simulation avance pendant l'affichage, PAS_PAR_IMAGE pas par image.

    Paramètres
    ----------
    mode : str
        Initialisation (voir init_particles) : "uniforme" (boîte carrée L, couche 1)
        ou "separe" (boîte L_SEPARE, N_A_SEPARE A à gauche et N_B_SEPARE B à droite, couche 2).
    n_steps : int
        Nombre total de pas de temps simulés (durée simulée n_steps·DT, en s).
    video : str ou None
        None : animation en direct dans une fenêtre. Sinon, chemin du fichier vidéo
        enregistré (.mp4 : nécessite ffmpeg ; .gif : fonctionne partout, fichier plus gros).
    """
    if video is None:
        plt.switch_backend("TkAgg")  # analysis.py impose « Agg » (sans fenêtre) ; Tk est fourni avec Python
    rng = np.random.default_rng(SEED)
    boite, n_a, n_b = (L, N_A, N_B) if mode == "uniforme" else (L_SEPARE, N_A_SEPARE, N_B_SEPARE)
    positions, labels = init_particles(n_a, n_b, boite, rng, mode=mode)
    lx, ly = np.broadcast_to(boite, 2)

    # même mise en page que plot_snapshot
    fig, ax = plt.subplots(figsize=(6.5 * max(1, lx / ly) ** 0.5, 6))
    nuages = {espece: ax.scatter([], [], s=8, color=COULEURS_ESPECES[espece], label=NOMS_ESPECES[espece])
              for espece in (A, B, C)}
    ax.set_xlim(0, lx)
    ax.set_ylim(0, ly)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.legend(title="Espèce", loc="upper left", bbox_to_anchor=(1.02, 1))
    parametres = f"init {mode}, D = {D:g} µm²/s, dt = {DT:g} s, r_c = {R_C:g} µm, p = {P:g}, SEED = {SEED}"

    def mise_a_jour(image):
        """Avance de PAS_PAR_IMAGE pas (sauf pour la 1re image : état initial) et redessine les nuages."""
        nonlocal positions, labels
        if image > 0:
            for _ in range(PAS_PAR_IMAGE):
                positions = step(positions, D, DT, boite, rng)
                positions, labels, _ = reaction_step(positions, labels, R_C, P, rng)
        n_a, n_b, n_c = count_species(labels)
        for espece, nuage in nuages.items():
            nuage.set_offsets(positions[labels == espece])
        ax.set_title(f"t = {image * PAS_PAR_IMAGE * DT:.1f} s — A = {n_a}, B = {n_b}, C = {n_c}\n{parametres}",
                     fontsize=10)

    n_images = n_steps // PAS_PAR_IMAGE + 1
    animation = FuncAnimation(fig, mise_a_jour, frames=n_images, interval=30, repeat=False)
    if video is None:
        plt.show()
    else:
        animation.save(video, fps=30, dpi=100)
        print(f"Vidéo enregistrée : {video} ({n_images} images, {n_steps * DT:g} s simulées)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulation SPARC de la réaction A + B → C.")
    parser.add_argument("--convergence", action="store_true",
                        help="lance l'étude de convergence en dt et en N (issue #9, une quinzaine de minutes)")
    parser.add_argument("--diffusion", action="store_true",
                        help="lance l'étude de l'écart à l'action de masse en fonction de D puis de p "
                             "(une vingtaine de minutes)")
    parser.add_argument("--animation", choices=("uniforme", "separe"),
                        help="anime la simulation en direct, avec A et B mélangés ou séparés")
    parser.add_argument("--pas", type=int, default=N_STEPS,
                        help=f"nombre de pas de temps de l'animation (défaut : {N_STEPS})")
    parser.add_argument("--video", metavar="FICHIER",
                        help="avec --animation : enregistre la vidéo (.mp4 ou .gif) au lieu d'ouvrir une fenêtre")
    arguments = parser.parse_args()
    if arguments.animation:
        animate(arguments.animation, arguments.pas, arguments.video)
    elif arguments.convergence:
        convergence_study()
    elif arguments.diffusion:
        diffusion_study()
    else:
        check_layer1()
