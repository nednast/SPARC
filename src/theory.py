"""
Équations de champ moyen et formules théoriques.

Couche 1 — loi d'action de masse pour A + B → C en milieu homogène (2D) :

    d[A]/dt = d[B]/dt = -k·[A]·[B],   d[C]/dt = +k·[A]·[B]

[A], [B], [C] sont des concentrations surfaciques (µm⁻²) et k une constante
de vitesse en µm²/s (en 2D, k·[A]·[B] doit être en µm⁻²/s).
"""

import numpy as np
from scipy.integrate import solve_ivp


def rate_constant(p, r_c, dt):
    """
    Constante de vitesse macroscopique k déduite des paramètres microscopiques.

    Raisonnement (régime limité par la réaction, p petit) :
    - les particules sont réparties uniformément ; une paire (A, B) donnée est en
      collision (distance < r_c) avec la probabilité π·r_c² / L² ;
    - il y a donc en moyenne N_A·N_B·π·r_c² / L² paires en collision ;
    - chacune réagit avec la probabilité p pendant dt, d'où un nombre de réactions
      par seconde égal à (p / dt)·N_A·N_B·π·r_c² / L² ;
    - en divisant par L² pour passer aux concentrations [A] = N_A / L² :
          d[A]/dt = -(p·π·r_c² / dt)·[A]·[B],   donc   k = p·π·r_c² / dt.

    Hypothèses : p petit (une paire qui se rencontre a le temps de se « mélanger »
    avant de réagir, la répartition reste uniforme) et effets de bord négligés
    (près d'une paroi, une partie du disque de rayon r_c est hors de la boîte).

    Paramètres
    ----------
    p : float
        Probabilité de réaction d'une paire en collision, par pas de temps.
    r_c : float
        Rayon de réaction (µm).
    dt : float
        Pas de temps (s).

    Retour
    ------
    float
        Constante de vitesse k (µm²/s).
    """
    return p * np.pi * r_c ** 2 / dt


def mass_action_rhs(t, c, k):
    """
    Second membre de la loi d'action de masse pour A + B → C, au format de solve_ivp.

    Paramètres
    ----------
    t : float
        Instant (s), non utilisé (système autonome) mais exigé par solve_ivp.
    c : ndarray (3,)
        Concentrations [A], [B], [C] (µm⁻²).
    k : float
        Constante de vitesse (µm²/s).

    Retour
    ------
    list de 3 floats
        Dérivées d[A]/dt, d[B]/dt, d[C]/dt (µm⁻²/s).
    """
    c_a, c_b, _ = c
    vitesse = k * c_a * c_b  # vitesse de réaction (µm⁻²/s)
    return [-vitesse, -vitesse, vitesse]


def solve_mass_action(c0, k, t):
    """
    Résout numériquement la loi d'action de masse avec scipy.integrate.solve_ivp.

    Paramètres
    ----------
    c0 : array-like (3,)
        Concentrations initiales [A]₀, [B]₀, [C]₀ (µm⁻²).
    k : float
        Constante de vitesse (µm²/s).
    t : ndarray (n,)
        Instants (s) auxquels on veut la solution, croissants.

    Retour
    ------
    ndarray (n, 3)
        Concentrations [A], [B], [C] (µm⁻²) à chaque instant de t.
    """
    solution = solve_ivp(mass_action_rhs, (t[0], t[-1]), c0, t_eval=t, args=(k,),
                         method="RK45", rtol=1e-8, atol=1e-12)
    if not solution.success:
        raise RuntimeError(f"solve_ivp a échoué : {solution.message}")
    return solution.y.T  # solve_ivp renvoie un tableau (3, n), on le transpose en (n, 3)


def mass_action_exact(c0, k, t):
    """
    Solution exacte de la loi d'action de masse, pour vérifier solve_mass_action.

    Comme [A] - [B] = Δ reste constant, d[A]/dt = -k·[A]·([A] - Δ) s'intègre :
    - si [A]₀ = [B]₀ :  [A](t) = [A]₀ / (1 + k·[A]₀·t)
    - sinon         :  [A](t) = Δ·[A]₀ / ([A]₀ - [B]₀·exp(-k·Δ·t)),  Δ = [A]₀ - [B]₀
    puis [B] = [A] - Δ et [C] = [C]₀ + ([A]₀ - [A]).

    Paramètres
    ----------
    c0 : array-like (3,)
        Concentrations initiales [A]₀, [B]₀, [C]₀ (µm⁻²).
    k : float
        Constante de vitesse (µm²/s).
    t : ndarray (n,)
        Instants (s).

    Retour
    ------
    ndarray (n, 3)
        Concentrations [A], [B], [C] (µm⁻²) à chaque instant de t.
    """
    c_a0, c_b0, c_c0 = c0
    delta = c_a0 - c_b0
    if np.isclose(delta, 0.0):
        c_a = c_a0 / (1 + k * c_a0 * t)
    else:
        c_a = delta * c_a0 / (c_a0 - c_b0 * np.exp(-k * delta * t))
    return np.column_stack((c_a, c_a - delta, c_c0 + c_a0 - c_a))


if __name__ == "__main__":
    from config import L, N_A, N_B, DT, R_C, P

    k = rate_constant(P, R_C, DT)
    print(f"k = p·π·r_c²/dt = {P}·π·{R_C:g}²/{DT:g} = {k:.4f} µm²/s")

    # Test 1 — solve_ivp comparé à la solution exacte, pour [A]₀ = [B]₀ puis [A]₀ ≠ [B]₀
    t = np.linspace(0.0, 50.0, 501)  # s
    for n_a, n_b in ((N_A, N_B), (N_A, N_B // 2)):
        c0 = np.array([n_a, n_b, 0]) / L ** 2  # µm⁻²
        numerique = solve_mass_action(c0, k, t)
        exacte = mass_action_exact(c0, k, t)
        ecart = np.max(np.abs(numerique - exacte)) / np.max(c0)
        print(f"Test 1 — N_A = {n_a}, N_B = {n_b} : écart relatif max solve_ivp / exacte = {ecart:.1e}")
        assert ecart < 1e-6

    # Test 2 — conservation : [A] + [C] et [B] + [C] constants dans la solution numérique
    c0 = np.array([N_A, N_B, 0]) / L ** 2
    c = solve_mass_action(c0, k, t)
    derive = max(np.max(np.abs(c[:, 0] + c[:, 2] - c0[0])), np.max(np.abs(c[:, 1] + c[:, 2] - c0[1])))
    print(f"Test 2 — conservation de la matière : dérive max = {derive:.1e} µm⁻²")
    assert derive < 1e-9

    # Ordre de grandeur : temps de demi-réaction pour [A]₀ = [B]₀, t_1/2 = 1 / (k·[A]₀)
    print(f"Temps de demi-réaction 1/(k·[A]₀) = {1 / (k * c0[0]):.1f} s")
    print("Tous les tests passent.")
