"""
Paramètres globaux de la simulation SPARC.

Unités utilisées dans tout le projet (cohérence dimensionnelle) :
- longueur : micromètre (µm)
- temps    : seconde (s)
- coefficient de diffusion D : µm²/s

Toute nouvelle constante doit être ajoutée ici, avec son unité en commentaire.
"""

# Graine du générateur aléatoire : tous les tirages passent par
# numpy.random.default_rng(SEED) pour que les résultats soient reproductibles.
SEED = 42

# Codes des espèces, stockés dans le tableau `labels`
A = 0
B = 1
C = 2
NOMS_ESPECES = {A: "A", B: "B", C: "C"}
COULEURS_ESPECES = {A: "tab:blue", B: "tab:red", C: "tab:green"}  # identiques sur toutes les figures

# Boîte de simulation
L = 100.0  # côté de la boîte carrée (µm)
L_SEPARE = (300.0, 100.0)  # boîte rectangulaire (Lx, Ly) de la couche 2, réactifs séparés (µm) :
                           # A dans le tiers gauche, B dans le tiers droit (100 x 100 µm chacun),
                           # donc même concentration initiale que la couche 1 (500 / 100² = 0.05 µm⁻²)

# Populations initiales (nombre de particules, sans unité)
N_A = 500
N_B = 500

# Diffusion et discrétisation en temps
D = 10.0      # coefficient de diffusion (µm²/s), ordre de grandeur d'une protéine en solution
DT = 0.01     # pas de temps (s) ; déplacement typique par axe et par pas : sqrt(2*D*DT) ≈ 0.45 µm
N_STEPS = 5000  # nombre de pas de temps (sans unité) ; durée simulée = N_STEPS * DT = 50 s

# Réaction
R_C = 2.0  # rayon de réaction (µm) : une paire A-B à distance < R_C peut réagir.
           # Doit rester nettement supérieur au déplacement relatif d'une paire par pas,
           # sqrt(2 * 2*D*DT) ≈ 0.9 µm, sinon des particules « se croisent » sans être détectées.
           # Valeur provisoire, à ajuster avec p (issues #6 et #8).
P = 0.001  # probabilité de réaction d'une paire A-B en collision, par pas de temps (sans unité).
           # Attention : p est défini « par pas », donc le taux de réaction réel dépend de DT.
           # Pour comparer plusieurs DT (issue #9), il faudra garder p / DT constant.

# Étude de convergence (issue #9) : python src/run_simulation.py --convergence
FACTEURS_DT = (1, 2, 4)             # pas testés : dt = DT / facteur, avec p / dt constant (même k)
N_RUNS_DT = 10                      # simulations indépendantes par valeur de dt
N_TOTAUX = (200, 500, 1000, 2000)   # N = N_A + N_B testés (N_A = N_B), à concentration initiale
                                    # constante : la boîte est agrandie avec N
N_RUNS_N = (40, 20, 10, 10)         # simulations indépendantes pour chaque N : plus pour les
                                    # petits N, plus bruités et moins coûteux

# Étude de la limitation par la diffusion : python src/run_simulation.py --diffusion
VALEURS_D = (2.5, 5.0, 10.0, 20.0, 40.0)  # coefficients de diffusion testés (µm²/s), p, dt et k fixés.
                                          # Pour D = 40, déplacement relatif d'une paire par pas
                                          # sqrt(4·D·DT) ≈ 1.3 µm : encore inférieur à R_C
N_TOTAL_D = 10000   # N = N_A + N_B (N_A = N_B) à la concentration de référence (0.05 µm⁻²) :
                    # boîte de 316 µm, effet de bord faible (≈ 0.5 %)
N_RUNS_D = 10       # simulations indépendantes par valeur de D (et par couple (D, p) ci-dessous)
FACTEURS_P = (0.5, 1, 2, 4)  # 2e partie : p = P·facteur pendant N_STEPS / facteur pas, donc k·t_final
                             # constant (même avancement prédit). N_STEPS / facteur doit être entier
VALEURS_D_P = (10.0, 40.0)   # coefficients de diffusion (µm²/s) pour lesquels on fait varier p
