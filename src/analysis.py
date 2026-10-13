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

# Dossier figures/ à la racine du dépôt, quel que soit le dossier depuis lequel on lance le script
DOSSIER_FIGURES = Path(__file__).resolve().parent.parent / "figures"


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
