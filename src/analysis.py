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
    donnees = np.column_stack((t, comptages, concentrations))
    entete = ""
    if params:
        entete += "Paramètres : " + ", ".join(f"{cle} = {valeur}" for cle, valeur in params.items()) + "\n"
    entete += "t (s),N_A,N_B,N_C,c_A (µm^-2),c_B (µm^-2),c_C (µm^-2)"

    DOSSIER_RESULTATS.mkdir(exist_ok=True)
    chemin = DOSSIER_RESULTATS / f"{nom}.csv"
    np.savetxt(chemin, donnees, delimiter=",", header=entete, encoding="utf-8",
               fmt=["%.6g", "%d", "%d", "%d", "%.6e", "%.6e", "%.6e"])
    return chemin


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
    L : float
        Côté de la boîte carrée (µm).
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
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for espece in (A, B, C):
        masque = labels == espece
        ax.scatter(positions[masque, 0], positions[masque, 1], s=8,
                   color=COULEURS_ESPECES[espece],
                   label=f"{NOMS_ESPECES[espece]} ({np.sum(masque)})")

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    titre = f"État du système à t = {t:.2f} s"
    if params:
        titre += "\n" + ", ".join(f"{cle} = {valeur}" for cle, valeur in params.items())
    ax.set_title(titre, fontsize=10)
    ax.legend(title="Espèce (effectif)", loc="upper left", bbox_to_anchor=(1.02, 1))

    if nom is None:
        # microsecondes incluses : deux figures créées dans la même seconde ne s'écrasent pas
        horodatage = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        nom = f"tmp_snapshot_t={t:.2f}s_{horodatage}"
    DOSSIER_FIGURES.mkdir(exist_ok=True)
    chemin = DOSSIER_FIGURES / f"{nom}.png"
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)  # libère la mémoire quand on trace de nombreux instantanés
    return chemin
