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

# Boîte de simulation
L = 100.0  # côté de la boîte carrée (µm)

# Populations initiales (nombre de particules, sans unité)
N_A = 500
N_B = 500
