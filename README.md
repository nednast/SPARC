# SPARC — Simulation stochastique d'une réaction A + B → C par particules

Projet de **Modélisation mathématique** — BUT 3 Informatique, année universitaire 2026-2027.

> **État du projet :** en cours de développement. Les sections marquées *(à compléter)* seront renseignées au fur et à mesure de l'avancement (voir les issues GitHub).

---

## Sommaire

1. [Présentation](#1-présentation)
2. [Modèle](#2-modèle)
3. [Structure du dépôt](#3-structure-du-dépôt)
4. [Installation](#4-installation)
5. [Lancement](#5-lancement)
6. [Reproductibilité](#6-reproductibilité)
7. [Unités et paramètres](#7-unités-et-paramètres)
8. [Figures du rapport](#8-figures-du-rapport)
9. [Validation](#9-validation)
10. [Usage de l'IA générative](#10-usage-de-lia-générative)
11. [Équipe](#11-équipe)
12. [Références](#12-références)

---

## 1. Présentation

Ce projet modélise une **réaction chimique bimoléculaire** à l'échelle des particules individuelles. Deux espèces, A et B, se déplacent par marche aléatoire (diffusion brownienne) dans une boîte bidimensionnelle. Lorsqu'une particule A et une particule B se rencontrent, elles peuvent réagir pour former une particule C.

**Question étudiée :** dans quelles conditions une description microscopique et stochastique (particules individuelles) retrouve-t-elle les lois macroscopiques déterministes (loi d'action de masse, équations de réaction-diffusion), et que se passe-t-il quand ces conditions ne sont plus réunies (faible nombre de particules, réactifs initialement séparés, réactions non linéaires) ?

---

## 2. Modèle

Le projet est construit en quatre couches successives, chacune s'appuyant sur la précédente.

### Couche 1 — Réaction en milieu homogène

- N_A particules A et N_B particules B sont réparties uniformément dans une boîte carrée de côté L.
- À chaque pas de temps dt, chaque particule effectue un déplacement gaussien :

  Δx = √(2·D·dt) · ξ,  ξ ~ 𝒩(0, 1) (indépendamment selon x et y)

- Les parois sont **réfléchissantes** : une particule qui sortirait du domaine est renvoyée symétriquement à l'intérieur.
- Une paire (A, B) dont la distance est inférieure au rayon de réaction r_c réagit avec une probabilité p : les deux particules sont retirées et une particule C est créée à leur position moyenne. Chaque particule ne peut réagir qu'une seule fois par pas de temps.
- Les populations obtenues sont comparées à la **loi d'action de masse** en champ moyen :

  d[A]/dt = d[B]/dt = −k·[A]·[B],  d[C]/dt = +k·[A]·[B]

  résolue numériquement avec `scipy.integrate.solve_ivp`. La constante k est reliée aux paramètres microscopiques (p, r_c, D, dt).

### Couche 2 — Réactifs initialement séparés

- A est placé dans la partie gauche d'une boîte rectangulaire, B dans la partie droite.
- Une zone de réaction se forme à l'interface. On mesure les profils de concentration le long de x et la position du front au cours du temps.
- Les résultats sont confrontés aux modèles de réaction-diffusion (lois d'échelle du front, équations de type Fisher-KPP).

### Couche 3 — Réactions autocatalytique et réversible

- Autocatalyse : C + A → 2A (constante k₂).
- Réaction inverse : C → A + B (probabilité de dissociation par pas de temps, constante k_rev).
- Exploration systématique de l'espace des paramètres (k₂, k_rev) : stabilisation, extinction, oscillations éventuelles.

### Couche 4 — Stochastique vs déterministe

- Comparaison du comportement à petit N (20 à 50 particules : fluctuations importantes, extinction possible d'une espèce) et à grand N (plusieurs milliers : convergence vers la solution déterministe).

---

## 3. Structure du dépôt

```
SPARC/
├── README.md              # ce fichier
├── requirements.txt       # dépendances Python
├── .gitignore
├── src/
│   ├── config.py          # paramètres globaux et graine aléatoire (SEED)
│   ├── particles.py       # initialisation des particules, déplacement brownien, conditions aux limites
│   ├── reactions.py       # détection des collisions et règles de réaction
│   ├── theory.py          # équations différentielles de champ moyen (solve_ivp), formules théoriques
│   ├── analysis.py        # comptage des espèces, profils de densité, suivi du front, tracés
│   └── run_simulation.py  # point d'entrée : boucle temporelle et sauvegarde des résultats
├── figures/               # figures générées (PNG/PDF) destinées au rapport
├── resultats/             # sorties numériques (CSV) — régénérées par les scripts, non versionnées
└── report/                # sources LaTeX du rapport
```

---

## 4. Installation

### Prérequis

- Python ≥ 3.10
- Git

### Dépendances

| Bibliothèque | Usage |
|---|---|
| `numpy` | tableaux, tirages aléatoires, calcul vectorisé des distances |
| `scipy` | résolution des équations différentielles (`solve_ivp`) |
| `matplotlib` | figures et animations |

Conformément aux consignes, aucune bibliothèque spécialisée (moteur physique, code de dynamique moléculaire, etc.) n'est utilisée : le modèle (déplacement, conditions aux limites, réactions, schéma en temps) est entièrement écrit par le groupe.

### Mise en place de l'environnement

```bash
git clone https://github.com/nednast/SPARC.git
cd SPARC
python -m venv venv
```

Activation de l'environnement virtuel :

```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (cmd)
venv\Scripts\activate.bat

# Linux / macOS
source venv/bin/activate
```

Installation des dépendances :

```bash
pip install -r requirements.txt
```

Vérification :

```bash
python -c "import numpy, scipy, matplotlib; print('OK')"
```

---

## 5. Lancement

Toutes les commandes sont exécutées depuis la racine du dépôt, environnement activé.

```bash
python src/run_simulation.py
```

Pour l'instant (issue #8), ce script lance la simulation avec réaction A + B → C sur 5000 pas (50 s simulées), puis :

- vérifie le confinement des particules dans la boîte et la conservation de la matière (N_A + N_C et N_B + N_C constants) ;
- affiche les populations à t = 0, 3 et 50 s ;
- enregistre l'historique des populations dans `resultats/populations_couche1.csv` ;
- compare les populations à la loi d'action de masse (k = p·π·r_c²/dt, résolue par `solve_ivp`) et enregistre `figures/fig_populations_couche1.png` ;
- sauvegarde trois instantanés dans `figures/` ;
- compare le déplacement quadratique moyen sans réaction à la valeur théorique 4·D·t.

Durée : environ 8 s.

Les fonctions de `src/theory.py` se testent seules (comparaison de `solve_ivp` à la solution exacte, conservation, correction de bord de k vérifiée par tirage aléatoire) :

```bash
python src/theory.py
```

Étude de convergence (issue #9) :

```bash
python src/run_simulation.py --convergence
```

- **en dt** : dt = 0.01, 0.005 et 0.0025 s, avec p/dt constant pour garder le même k, 10 simulations indépendantes par valeur ;
- **en N** : N = N_A + N_B = 200, 500, 1000 et 2000 (N_A = N_B), à concentration initiale constante (0.05 µm⁻²), la boîte étant agrandie avec N ; 40, 20, 10 et 10 simulations respectivement ;
- écrit `resultats/convergence_dt.csv`, `resultats/convergence_N.csv`, `figures/fig_convergence_dt.png` et `figures/fig_convergence_N.png`. Les paramètres de l'étude sont dans `src/config.py` (`FACTEURS_DT`, `N_RUNS_DT`, `N_TOTAUX`, `N_RUNS_N`).

*(à compléter : options de ligne de commande, choix de la couche à simuler, scripts de génération de chaque figure)*

### Temps d'exécution typique

*(à compléter après mesure, en précisant la machine utilisée)*

| Simulation | Paramètres | Durée |
|---|---|---|
| Couche 1, configuration de référence (`python src/run_simulation.py`) | N_A = N_B = 500, 5000 pas | ≈ 8 s |
| Études de convergence (`--convergence`) | voir ci-dessus | ≈ 13 min |
| — une simulation, N = 200 / 500 / 1000 / 2000 (dt = 0.01 s) | 5000 pas | 0.3 / 1.2 / 5.8 / 27.8 s |
| — une simulation, N = 1000, dt = 0.005 / 0.0025 s | 10 000 / 20 000 pas | 11.6 / 23.3 s |

Temps mesurés sous Windows 11, Python 3.14 *(à compléter : processeur de la machine)*. Le coût d'un pas croît comme N_A × N_B (détection naïve des collisions).
| Couche 2, front de réaction | *à définir* | *à mesurer* |

---

## 6. Reproductibilité

- Tous les tirages aléatoires passent par un unique générateur `numpy.random.default_rng(SEED)`, avec `SEED = 42` défini dans `src/config.py`. Aucun appel à l'aléatoire global de NumPy (`np.random.rand`, etc.) n'est utilisé.
- Les historiques de populations sont sauvegardés dans `resultats/` au format CSV (colonnes `t (s), N_A, N_B, N_C, c_A, c_B, c_C`, concentrations en µm⁻², paramètres de la simulation en en-tête) afin de pouvoir refaire les analyses sans relancer les simulations. Lecture : `load_populations(nom)` dans `src/analysis.py`. Ces fichiers ne sont pas versionnés : ils sont régénérés par `python src/run_simulation.py`.
- Les paramètres exacts de chaque figure du rapport sont listés à la [section 8](#8-figures-du-rapport).

---

## 7. Unités et paramètres

Unités retenues dans tout le code (définies dans `src/config.py`) : **longueur en micromètres (µm), temps en secondes (s)**. Le coefficient de diffusion D s'exprime donc en µm²/s.

| Symbole | Signification | Unité | Valeur actuelle |
|---|---|---|---|
| L | côté de la boîte | µm | 100 |
| N_A, N_B | nombres initiaux de particules A et B | — | 500, 500 |
| D | coefficient de diffusion | µm²/s | 10 |
| dt | pas de temps | s | 0.01 |
| n_steps | nombre de pas de temps (durée simulée = n_steps · dt) | — | 5000 (50 s) |
| r_c | rayon de réaction | µm | 2 (provisoire) |
| p | probabilité de réaction d'une paire en collision, par pas de temps | — | 0.001 |
| k | constante de vitesse macroscopique (2D), k = p·π·r_c²/dt | µm²/s | 1.257 |
| k₂ | constante de la réaction autocatalytique | µm²/s | *(issue #14)* |
| k_rev | constante de la réaction inverse | 1/s | *(issue #15)* |
| SEED | graine du générateur aléatoire | — | 42 |

---

## 8. Figures du rapport

*(à compléter au fur et à mesure)* — pour chaque figure : commande exacte et paramètres utilisés.

| Figure | Description | Commande | Paramètres |
|---|---|---|---|
| 1 | Configuration initiale (couche 1) | | |
| 2 | Populations A, B, C : simulation vs loi d'action de masse (`fig_populations_couche1.png`) | `python src/run_simulation.py` | N_A = N_B = 500, L = 100 µm, D = 10 µm²/s, dt = 0.01 s, 5000 pas, r_c = 2 µm, p = 0.001, k = 1.257 µm²/s, SEED = 42 |
| 3 | Convergence en dt (`fig_convergence_dt.png`) | `python src/run_simulation.py --convergence` | N_A = N_B = 500, L = 100 µm, D = 10 µm²/s, r_c = 2 µm, dt = 0.01 / 0.005 / 0.0025 s avec p = 0.001 / 0.0005 / 0.00025 (k = 1.257 µm²/s), 50 s simulées, 10 simulations par dt, SEED = 42 |
| 4 | Convergence en N (`fig_convergence_N.png`) | `python src/run_simulation.py --convergence` | N = N_A + N_B = 200 / 500 / 1000 / 2000, [A]₀ = [B]₀ = 0.05 µm⁻² (L = 44.7 / 70.7 / 100 / 141.4 µm), D = 10 µm²/s, dt = 0.01 s, r_c = 2 µm, p = 0.001, 50 s simulées, 40 / 20 / 10 / 10 simulations, SEED = 42 |
| … | | | |

---

## 9. Validation

Critères quantitatifs retenus (détaillés dans le rapport) :

- **Invariants :** conservation de la matière (N_A + N_C et N_B + N_C constants pour A + B → C), aucune particule hors du domaine — écarts mesurés à chaque pas.
- **Sensibilité à la discrétisation :** stabilité des résultats lorsque dt diminue (dt, dt/2, dt/4) et lorsque N augmente.
- **Comparaison à la théorie :** loi d'action de masse (couche 1), lois de réaction-diffusion pour le front (couche 2), convergence en N vers le champ moyen (couche 4).
- **Cohérence dimensionnelle :** unités de toutes les grandeurs et vérification des ordres de grandeur.

### Résultats de l'étude de convergence (couche 1, issue #9)

Écart relatif de N_A à la loi d'action de masse, (N_A simulé − N_A théorique) / N_A théorique : moyenne sur les simulations ± écart-type de la moyenne. « Écart final » : à t = 50 s. « Écart quadratique » : racine de la moyenne dans le temps des écarts au carré (biais + fluctuations), moyennée sur les simulations. « Bord prédit » : écart attendu si l'on remplace k par k × `border_factor(r_c, L)`.

| Étude | Valeur | Simulations | Écart final | Bord prédit | Écart quadratique |
|---|---|---|---|---|---|
| dt (N = 1000) | 0.01 s | 10 | +0.3 % ± 1.8 % | +1.3 % | 3.7 % |
| | 0.005 s | 10 | +0.5 % ± 1.9 % | +1.3 % | 5.4 % |
| | 0.0025 s | 10 | +2.3 % ± 1.6 % | +1.3 % | 3.3 % |
| N (dt = 0.01 s) | 200 | 40 | +5.1 % ± 2.1 % | +2.9 % | 9.2 % ± 0.7 % |
| | 500 | 20 | +3.9 % ± 1.3 % | +1.8 % | 5.2 % ± 0.7 % |
| | 1000 | 10 | +3.0 % ± 1.9 % | +1.3 % | 4.8 % ± 0.7 % |
| | 2000 | 10 | +5.4 % ± 0.9 % | +0.9 % | 3.7 % ± 0.6 % |

Données brutes : `resultats/convergence_dt.csv` et `resultats/convergence_N.csv`.

---

## 10. Usage de l'IA générative

Des outils d'IA générative sont utilisés comme aide (relecture, explication de notions, suggestions de code). Tout le code et tout le contenu du rapport sont compris, vérifiés et justifiés par les membres du groupe. Le détail des outils et de leurs usages figure dans une section dédiée du rapport.

---

## 11. Équipe

| Membre | Rôle principal |
|---|---|
| Anastasia Nediak | *(à compléter)* |
| *(à compléter)* | *(à compléter)* |
| *(à compléter)* | *(à compléter)* |

Le tableau détaillé de répartition des tâches figure en annexe du rapport.

---

## 12. Références

- M. von Smoluchowski, « Versuch einer mathematischen Theorie der Koagulationskinetik kolloider Lösungen », *Zeitschrift für physikalische Chemie*, 92, 129–168, 1917.
- R. A. Fisher, « The wave of advance of advantageous genes », *Annals of Eugenics*, 7, 355–369, 1937.
- A. Kolmogorov, I. Petrovsky, N. Piskunov, « Étude de l'équation de la diffusion avec croissance de la quantité de matière et son application à un problème biologique », *Bulletin de l'Université d'État de Moscou*, Série internationale, Section A, 1, 1937.
- L. Gálfi, Z. Rácz, « Properties of the reaction front in an A + B → C type reaction-diffusion process », *Physical Review A*, 38, 3151–3154, 1988.
- D. T. Gillespie, « Exact stochastic simulation of coupled chemical reactions », *The Journal of Physical Chemistry*, 81, 2340–2361, 1977.

*(liste à compléter)*
