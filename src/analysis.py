"""
Analyse et visualisation des résultats de la simulation.
"""

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # tracé sans fenêtre : les figures sont uniquement enregistrées dans des fichiers
import matplotlib.pyplot as plt
import numpy as np

from config import A, B, C, NOMS_ESPECES, COULEURS_ESPECES

# Dossiers à la racine du dépôt, quel que soit le dossier depuis lequel on lance le script
DOSSIER_FIGURES = Path(__file__).resolve().parent.parent / "figures"
DOSSIER_RESULTATS = Path(__file__).resolve().parent.parent / "resultats"


def count_species(labels):
    """
    Compte les particules de chaque espèce.

    Paramètres
    ----------
    labels : ndarray (N,)
        Espèce de chaque particule (A, B ou C).

    Retour
    ------
    ndarray (3,) d'entiers
        [N_A, N_B, N_C]. np.bincount compte les occurrences de 0 (A), 1 (B) et 2 (C) ;
        minlength=3 garantit trois cases même si une espèce a disparu.
    """
    return np.bincount(labels, minlength=3)[[A, B, C]]


def compute_density_profile(positions, labels, n_bins, L):
    """
    Profils de concentration de A, B et C le long de x : histogramme par tranches verticales.

    La boîte est découpée en n_bins tranches de largeur Lx / n_bins, sur toute la hauteur Ly.
    La concentration d'une espèce dans une tranche est son nombre de particules divisé par
    l'aire de la tranche (Lx / n_bins)·Ly.

    Paramètres
    ----------
    positions : ndarray (N, 2)
        Positions des particules (µm).
    labels : ndarray (N,)
        Espèce de chaque particule (A, B ou C).
    n_bins : int
        Nombre de tranches en x.
    L : float ou array-like (2,)
        Côté de la boîte carrée (µm), ou dimensions (Lx, Ly) d'une boîte rectangulaire (µm).

    Retour
    ------
    centres : ndarray (n_bins,)
        Abscisse du centre de chaque tranche (µm).
    profils : ndarray (n_bins, 3)
        Concentrations [A], [B], [C] dans chaque tranche (µm⁻²), colonnes dans l'ordre A, B, C.
    """
    lx, ly = np.broadcast_to(L, 2)
    bords = np.linspace(0.0, lx, n_bins + 1)
    # np.histogram inclut le bord droit dans la dernière tranche : une particule en x = Lx est comptée
    comptages = np.column_stack([np.histogram(positions[labels == espece, 0], bins=bords)[0]
                                 for espece in (A, B, C)])
    centres = (bords[:-1] + bords[1:]) / 2
    return centres, comptages / ((lx / n_bins) * ly)


def save_populations(nom, t, comptages, L, params=None):
    """
    Enregistre l'évolution des populations dans resultats/<nom>.csv.

    Colonnes : t (s), N_A, N_B, N_C (nombres de particules) et c_A, c_B, c_C
    (concentrations surfaciques = N / L², en µm⁻²). Les paramètres de la simulation
    sont écrits en commentaire (lignes commençant par #) en tête du fichier.

    Paramètres
    ----------
    nom : str
        Nom du fichier, sans extension.
    t : ndarray (n,)
        Instants (s).
    comptages : ndarray (n, 3)
        Nombres de particules A, B, C à chaque instant.
    L : float
        Côté de la boîte carrée (µm), pour calculer les concentrations.
    params : dict ou None
        Paramètres de simulation à noter dans l'en-tête.

    Retour
    ------
    pathlib.Path
        Chemin du fichier CSV enregistré.
    """
    concentrations = comptages / L ** 2
    colonnes = ["t (s)", "N_A", "N_B", "N_C", "c_A (µm^-2)", "c_B (µm^-2)", "c_C (µm^-2)"]
    # %d : comptages entiers exacts ; %.6e : concentrations en notation scientifique
    return save_table(nom, colonnes, np.column_stack((t, comptages, concentrations)), params,
                      fmt=["%.6g", "%d", "%d", "%d", "%.6e", "%.6e", "%.6e"])


def load_populations(nom):
    """
    Relit un fichier écrit par save_populations.

    Retour
    ------
    t : ndarray (n,)
        Instants (s).
    comptages : ndarray (n, 3) d'entiers
        Nombres de particules A, B, C.
    """
    donnees = np.loadtxt(DOSSIER_RESULTATS / f"{nom}.csv", delimiter=",", encoding="utf-8")
    return donnees[:, 0], donnees[:, 1:4].astype(int)


def relative_gaps(trajectoires, n_theorie):
    """
    Écarts relatifs entre plusieurs simulations indépendantes et la théorie.

    Pour chaque simulation r et chaque instant : e[r, n] = (N_sim[r, n] - N_th[n]) / N_th[n].

    Paramètres
    ----------
    trajectoires : ndarray (n_runs, n)
        Nombre de particules (d'une espèce) dans chaque simulation, aux mêmes instants.
    n_theorie : ndarray (n,)
        Nombre prédit par la théorie aux mêmes instants.

    Retour
    ------
    dict
        "moyen"            : ndarray (n,), écart relatif moyen sur les simulations (biais) ;
        "incertitude"      : ndarray (n,), écart-type de cette moyenne (écart-type / √n_runs) ;
        "rms"              : float, moyenne sur les simulations de l'écart quadratique moyen
                             dans le temps, sqrt(moyenne_n e[r, n]²) : biais + fluctuations ;
        "rms_incertitude"  : float, écart-type de cette moyenne.
    """
    n_runs = len(trajectoires)
    ecarts = (trajectoires - n_theorie) / n_theorie
    rms_par_run = np.sqrt(np.mean(ecarts ** 2, axis=1))
    return {
        "moyen": ecarts.mean(axis=0),
        "incertitude": ecarts.std(axis=0, ddof=1) / np.sqrt(n_runs),
        "rms": rms_par_run.mean(),
        "rms_incertitude": rms_par_run.std(ddof=1) / np.sqrt(n_runs),
    }


def save_table(nom, colonnes, donnees, params=None, fmt="%.6g"):
    """
    Enregistre un tableau de résultats numériques dans resultats/<nom>.csv.

    Paramètres
    ----------
    nom : str
        Nom du fichier, sans extension.
    colonnes : list de str
        Noms des colonnes, avec leur unité.
    donnees : array-like (n_lignes, n_colonnes)
        Valeurs du tableau.
    params : dict ou None
        Paramètres à noter en commentaire dans l'en-tête.
    fmt : str ou list de str
        Format d'écriture de np.savetxt, commun à toutes les colonnes ou un par colonne.
        Par défaut 6 chiffres significatifs.

    Retour
    ------
    pathlib.Path
        Chemin du fichier CSV enregistré.
    """
    entete = ""
    if params:
        entete += "Paramètres : " + _texte_params(params) + "\n"
    entete += ",".join(colonnes)
    DOSSIER_RESULTATS.mkdir(exist_ok=True)
    chemin = DOSSIER_RESULTATS / f"{nom}.csv"
    np.savetxt(chemin, np.asarray(donnees), delimiter=",", header=entete, encoding="utf-8", fmt=fmt)
    return chemin


def _texte_params(params):
    """Paramètres de simulation mis en forme pour un titre de figure (« clé = valeur, ... »)."""
    return ", ".join(f"{cle} = {valeur}" for cle, valeur in params.items()) if params else ""


def _enregistrer(fig, nom, prefixe_brouillon):
    """Enregistre la figure en PNG 150 dpi dans figures/ (brouillon horodaté si nom est None)."""
    if nom is None:
        # microsecondes incluses : deux figures créées dans la même seconde ne s'écrasent pas
        horodatage = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        nom = f"tmp_{prefixe_brouillon}_{horodatage}"
    DOSSIER_FIGURES.mkdir(exist_ok=True)
    chemin = DOSSIER_FIGURES / f"{nom}.png"
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chemin


def plot_convergence_dt(t, resultats, biais_bord, params=None, nom=None):
    """
    Figure de convergence en dt : écart relatif de N_A à la loi d'action de masse.

    (a) écart relatif moyen en fonction du temps pour chaque dt, avec une bande de ±1
        écart-type de la moyenne ; (b) écart relatif à l'instant final en fonction de dt.
    Sur les deux, le tireté noir est le biais prédit par la seule correction de bord.

    Paramètres
    ----------
    t : ndarray (n,)
        Instants (s).
    resultats : list de dict
        Un dict par valeur de dt, avec les clés "dt" (s) et "moyen", "incertitude"
        (sortie de relative_gaps, sans unité).
    biais_bord : ndarray (n,)
        Écart relatif prédit en remplaçant k par k · border_factor(r_c, L).
    params : dict ou None
        Paramètres communs à afficher dans le titre.
    nom : str ou None
        Nom du fichier (sans extension) dans figures/ ; brouillon horodaté par défaut.

    Retour
    ------
    pathlib.Path
        Chemin du fichier PNG enregistré.
    """
    couleurs = plt.cm.viridis(np.linspace(0, 0.8, len(resultats)))
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))
    for res, couleur in zip(resultats, couleurs):
        ax_a.plot(t, 100 * res["moyen"], color=couleur, lw=1, label=f"dt = {res['dt']:g} s")
        ax_a.fill_between(t, 100 * (res["moyen"] - res["incertitude"]),
                          100 * (res["moyen"] + res["incertitude"]), color=couleur, alpha=0.2)
        ax_b.errorbar(res["dt"], 100 * res["moyen"][-1], yerr=100 * res["incertitude"][-1],
                      fmt="o", color=couleur, capsize=4)
    ax_a.plot(t, 100 * biais_bord, "k--", lw=1.5, label="biais prédit (effet de bord)")
    ax_a.axhline(0, color="gray", lw=0.8)
    ax_a.set_xlim(t[0], t[-1])
    ax_a.set_xlabel("t (s)")
    ax_a.set_ylabel("écart relatif moyen sur N_A (%)")
    ax_a.set_title("(a) Écart à la loi d'action de masse au cours du temps", fontsize=10)
    ax_a.legend(fontsize=8)
    ax_a.grid(alpha=0.3)

    dts = [res["dt"] for res in resultats]
    ax_b.axhline(100 * biais_bord[-1], color="k", ls="--", lw=1.5, label="biais prédit (effet de bord)")
    ax_b.axhline(0, color="gray", lw=0.8)
    ax_b.set_xscale("log")
    ax_b.set_xticks(dts)
    ax_b.set_xticklabels([f"{dt:g}" for dt in dts])
    ax_b.minorticks_off()
    ax_b.set_xlabel("dt (s)")
    ax_b.set_ylabel(f"écart relatif sur N_A à t = {t[-1]:g} s (%)")
    ax_b.set_title("(b) Écart final en fonction du pas de temps", fontsize=10)
    ax_b.legend(fontsize=8)
    ax_b.grid(alpha=0.3)

    fig.suptitle("Convergence en dt (p/dt constant)\n" + _texte_params(params), fontsize=9)
    fig.tight_layout()
    return _enregistrer(fig, nom, "convergence_dt")


def plot_convergence_n(resultats, t_final, params=None, nom=None):
    """
    Figure de convergence en N : écart de N_A à la loi d'action de masse quand N augmente.

    (a) écart quadratique moyen (biais + fluctuations) en fonction de N, en échelle
        log-log, avec une droite de pente -1/2 en référence ; (b) écart relatif moyen à
        l'instant final, comparé au biais prédit par la correction de bord.

    Paramètres
    ----------
    resultats : list de dict
        Un dict par valeur de N, avec les clés "N" (N_A + N_B), "rms", "rms_incertitude",
        "final", "final_incertitude" et "biais_bord" (sans unité).
    t_final : float
        Instant final (s), pour les légendes.
    params : dict ou None
        Paramètres communs à afficher dans le titre.
    nom : str ou None
        Nom du fichier (sans extension) dans figures/ ; brouillon horodaté par défaut.

    Retour
    ------
    pathlib.Path
        Chemin du fichier PNG enregistré.
    """
    n = np.array([res["N"] for res in resultats])
    rms = np.array([res["rms"] for res in resultats])
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_a.errorbar(n, 100 * rms, yerr=[100 * res["rms_incertitude"] for res in resultats],
                  fmt="o", color="tab:blue", capsize=4, label="simulations")
    reference = 100 * rms[-1] * np.sqrt(n[-1] / n)  # pente -1/2 passant par le plus grand N
    ax_a.plot(n, reference, "k:", label="référence ∝ N$^{-1/2}$")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xticks(n)
    ax_a.set_xticklabels([str(v) for v in n])
    ax_a.minorticks_off()
    ax_a.set_xlabel("N = N_A + N_B")
    ax_a.set_ylabel("écart relatif quadratique moyen sur N_A (%)")
    ax_a.set_title("(a) Écart total (biais + fluctuations)", fontsize=10)
    ax_a.legend(fontsize=8)
    ax_a.grid(alpha=0.3, which="both")

    ax_b.errorbar(n, [100 * res["final"] for res in resultats],
                  yerr=[100 * res["final_incertitude"] for res in resultats],
                  fmt="o", color="tab:blue", capsize=4, label="simulations (moyenne)")
    ax_b.plot(n, [100 * res["biais_bord"] for res in resultats], "k--", marker="x",
              label="biais prédit (effet de bord)")
    ax_b.axhline(0, color="gray", lw=0.8)
    ax_b.set_xscale("log")
    ax_b.set_xticks(n)
    ax_b.set_xticklabels([str(v) for v in n])
    ax_b.minorticks_off()
    ax_b.set_xlabel("N = N_A + N_B")
    ax_b.set_ylabel(f"écart relatif moyen sur N_A à t = {t_final:g} s (%)")
    ax_b.set_title("(b) Biais à l'instant final", fontsize=10)
    ax_b.legend(fontsize=8)
    ax_b.grid(alpha=0.3)

    fig.suptitle("Convergence en N (concentration initiale constante)\n" + _texte_params(params),
                 fontsize=9)
    fig.tight_layout()
    return _enregistrer(fig, nom, "convergence_N")


def plot_diffusion(t, resultats, params=None, nom=None):
    """
    Figure de l'étude en D : écart relatif de N_A à la loi d'action de masse corrigée du bord.

    (a) écart relatif moyen en fonction du temps pour chaque D, avec une bande de ±1
        écart-type de la moyenne ; (b) écart à l'instant final en fonction de D (échelle
        logarithmique en D), avec une courbe de référence ∝ 1/D passant par le plus petit D.

    Paramètres
    ----------
    t : ndarray (n,)
        Instants (s).
    resultats : list de dict
        Un dict par valeur de D, avec les clés "D" (µm²/s) et "moyen", "incertitude"
        (sortie de relative_gaps, sans unité), dans l'ordre croissant de D.
    params : dict ou None
        Paramètres communs à afficher dans le titre.
    nom : str ou None
        Nom du fichier (sans extension) dans figures/ ; brouillon horodaté par défaut.

    Retour
    ------
    pathlib.Path
        Chemin du fichier PNG enregistré.
    """
    couleurs = plt.cm.viridis(np.linspace(0, 0.8, len(resultats)))
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.5))
    for res, couleur in zip(resultats, couleurs):
        ax_a.plot(t, 100 * res["moyen"], color=couleur, lw=1, label=f"D = {res['D']:g} µm²/s")
        ax_a.fill_between(t, 100 * (res["moyen"] - res["incertitude"]),
                          100 * (res["moyen"] + res["incertitude"]), color=couleur, alpha=0.2)
        ax_b.errorbar(res["D"], 100 * res["moyen"][-1], yerr=100 * res["incertitude"][-1],
                      fmt="o", color=couleur, capsize=4)
    ax_a.axhline(0, color="gray", lw=0.8)
    ax_a.set_xlim(t[0], t[-1])
    ax_a.set_xlabel("t (s)")
    ax_a.set_ylabel("écart relatif moyen sur N_A (%)")
    ax_a.set_title("(a) Écart à l'action de masse corrigée du bord, au cours du temps", fontsize=10)
    ax_a.legend(fontsize=8)
    ax_a.grid(alpha=0.3)

    d = np.array([res["D"] for res in resultats])
    reference = 100 * resultats[0]["moyen"][-1] * d[0] / d  # ∝ 1/D, passant par le plus petit D
    ax_b.plot(d, reference, "k:", label="référence ∝ 1/D")
    ax_b.axhline(0, color="gray", lw=0.8)
    ax_b.set_xscale("log")
    ax_b.set_xticks(d)
    ax_b.set_xticklabels([f"{v:g}" for v in d])
    ax_b.minorticks_off()
    ax_b.set_xlabel("D (µm²/s)")
    ax_b.set_ylabel(f"écart relatif sur N_A à t = {t[-1]:g} s (%)")
    ax_b.set_title("(b) Écart final en fonction du coefficient de diffusion", fontsize=10)
    ax_b.legend(fontsize=8)
    ax_b.grid(alpha=0.3)

    fig.suptitle("Limitation par la diffusion (p, dt et k fixés)\n" + _texte_params(params), fontsize=9)
    fig.tight_layout()
    return _enregistrer(fig, nom, "diffusion")


def plot_populations(t, comptages, comptages_theorie, params=None, nom=None):
    """
    Trace l'évolution des populations A, B, C : simulation (trait plein) et théorie (tirets).

    Paramètres
    ----------
    t : ndarray (n,)
        Instants (s).
    comptages : ndarray (n, 3)
        Nombres de particules A, B, C dans la simulation.
    comptages_theorie : ndarray (n, 3)
        Nombres de particules A, B, C prédits par la théorie (concentrations × L²).
    params : dict ou None
        Paramètres de simulation à afficher sous le titre.
    nom : str ou None
        Nom du fichier (sans extension) dans figures/. Par défaut, brouillon horodaté
        « tmp_populations_<date>-<heure>-<microsecondes> ».

    Retour
    ------
    pathlib.Path
        Chemin du fichier PNG enregistré.
    """
    # Si N_A = N_B au départ, les courbes de A et B sont confondues : A est tracé plus
    # épais, en dessous, pour qu'il reste visible autour de B.
    epaisseurs = {A: 3.0, B: 1.2, C: 1.2}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for espece in (A, B, C):
        couleur = COULEURS_ESPECES[espece]
        ax.plot(t, comptages[:, espece], color=couleur, lw=epaisseurs[espece],
                label=f"{NOMS_ESPECES[espece]} (simulation)")
        ax.plot(t, comptages_theorie[:, espece], color=couleur, ls="--", lw=epaisseurs[espece] + 0.3,
                label=f"{NOMS_ESPECES[espece]} (action de masse)")

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Nombre de particules")
    titre = "Populations : simulation vs loi d'action de masse"
    if params:
        titre += "\n" + _texte_params(params)
    ax.set_title(titre, fontsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(alpha=0.3)
    return _enregistrer(fig, nom, "populations")


def plot_density_profiles(centres, profils, instants, params=None, nom=None):
    """
    Trace les profils de concentration de A, B et C à plusieurs instants (un graphique par espèce).

    Paramètres
    ----------
    centres : ndarray (n_bins,)
        Abscisse du centre de chaque tranche (µm), voir compute_density_profile.
    profils : list de ndarray (n_bins, 3)
        Concentrations [A], [B], [C] (µm⁻²), une entrée par instant.
    instants : list de float
        Instants correspondants (s).
    params : dict ou None
        Paramètres de simulation à afficher sous le titre.
    nom : str ou None
        Nom du fichier (sans extension) dans figures/. Par défaut, brouillon horodaté
        « tmp_profils_<date>-<heure>-<microsecondes> ».

    Retour
    ------
    pathlib.Path
        Chemin du fichier PNG enregistré.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for ax, espece in zip(axes, (A, B, C)):
        for i, (profil, t) in enumerate(zip(profils, instants)):
            # même couleur par espèce, de plus en plus foncée avec le temps
            ax.step(centres, profil[:, espece], where="mid", color=COULEURS_ESPECES[espece],
                    alpha=0.25 + 0.75 * (i + 1) / len(instants), label=f"t = {t:g} s")
        ax.set_ylabel(f"[{NOMS_ESPECES[espece]}] (µm⁻²)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    axes[-1].set_xlabel("x (µm)")
    axes[-1].set_xlim(0, centres[-1] + centres[0])  # Lx : dernier centre (Lx − largeur/2) + premier (largeur/2)
    titre = "Profils de concentration le long de x"
    if params:
        titre += "\n" + _texte_params(params)
    axes[0].set_title(titre, fontsize=9)
    return _enregistrer(fig, nom, "profils")


def plot_snapshot(positions, labels, t, L, params=None, nom=None):
    """
    Trace l'état du système à l'instant t (nuage de points coloré par espèce) et l'enregistre.

    Paramètres
    ----------
    positions : ndarray (N, 2)
        Positions des particules (µm).
    labels : ndarray (N,)
        Espèce de chaque particule (A, B ou C).
    t : float
        Instant représenté (s).
    L : float ou array-like (2,)
        Côté de la boîte carrée (µm), ou dimensions (Lx, Ly) d'une boîte rectangulaire (µm).
    params : dict ou None
        Paramètres de simulation à afficher sous le titre, par ex. {"D": "10 µm²/s", "dt": "0.01 s"}.
    nom : str ou None
        Nom du fichier (sans extension) dans figures/. Par défaut, nom horodaté
        « tmp_snapshot_t=<t>s_<date>-<heure>-<microsecondes> » : brouillon ignoré par git.

    Retour
    ------
    pathlib.Path
        Chemin du fichier PNG enregistré.
    """
    lx, ly = np.broadcast_to(L, 2)
    fig, ax = plt.subplots(figsize=(6.5 * max(1, lx / ly) ** 0.5, 6))  # élargie pour une boîte allongée
    for espece in (A, B, C):
        masque = labels == espece
        ax.scatter(positions[masque, 0], positions[masque, 1], s=8,
                   color=COULEURS_ESPECES[espece],
                   label=f"{NOMS_ESPECES[espece]} ({np.sum(masque)})")

    ax.set_xlim(0, lx)
    ax.set_ylim(0, ly)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    titre = f"État du système à t = {t:.2f} s"
    if params:
        titre += "\n" + _texte_params(params)
    ax.set_title(titre, fontsize=10)
    ax.legend(title="Espèce (effectif)", loc="upper left", bbox_to_anchor=(1.02, 1))
    return _enregistrer(fig, nom, f"snapshot_t={t:.2f}s")
