"""
Équations de champ moyen et formules théoriques.

Couche 1 — loi d'action de masse pour A + B → C en milieu homogène (2D) :

    d[A]/dt = d[B]/dt = -k·[A]·[B],   d[C]/dt = +k·[A]·[B]

[A], [B], [C] sont des concentrations surfaciques (µm⁻²) et k une constante
de vitesse en µm²/s (en 2D, k·[A]·[B] doit être en µm⁻²/s).

Couche 2 — front de réaction pour A + B → C avec réactifs initialement séparés
(Gálfi & Rácz, 1988), voir front_position.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erfinv


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


def border_factor(r_c, L):
    """
    Correction de bord de k : nombre réel de paires en collision / nombre sans paroi.

    rate_constant suppose que chaque particule « voit » un disque complet de rayon r_c.
    Près d'une paroi, une partie de ce disque sort de la boîte, donc il y a un peu moins
    de paires en collision. Pour deux points tirés uniformément dans un carré de côté L,
    la probabilité d'être à distance < r (avec r <= L) vaut
        π·r²/L² - 8·r³/(3·L³) + r⁴/(2·L⁴),
    au lieu de π·r²/L² sans paroi. Le rapport des deux donne le facteur renvoyé.
    Cette formule est vérifiée par tirage aléatoire dans le bloc de test ci-dessous.

    Paramètres
    ----------
    r_c : float
        Rayon de réaction (µm), r_c <= L.
    L : float
        Côté de la boîte carrée (µm).

    Retour
    ------
    float
        Facteur sans unité, légèrement inférieur à 1 : k_effectif = k · border_factor(r_c, L).
    """
    x = r_c / L
    return 1 - 8 * x / (3 * np.pi) + x ** 2 / (2 * np.pi)


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


def front_position(t, c_a0, c_b0, D):
    """
    Position théorique du front de réaction pour A + B → C, réactifs séparés (Gálfi & Rácz, 1988).

    A occupe x < x0 à la concentration [A]₀, B occupe x > x0 à la concentration [B]₀.
    Avec le même D pour A et B, u = [A] - [B] n'est pas modifié par la réaction (A et B
    disparaissent ensemble), donc u obéit à l'équation de diffusion seule :
        u(x, t) = ([A]₀ - [B]₀)/2 - ([A]₀ + [B]₀)/2 · erf((x - x0) / sqrt(4·D·t)).
    Le front est l'endroit où [A] = [B], c'est-à-dire u = 0 :
        x_f - x0 = sqrt(4·D·t) · erfinv(([A]₀ - [B]₀) / ([A]₀ + [B]₀)).
    Le front avance donc en √t (et non à vitesse constante), vers le réactif le moins
    concentré ; il reste immobile si [A]₀ = [B]₀.
    Valable pour un domaine infini, aux temps longs devant 1/(k·[A]₀) (réaction rapide
    devant la diffusion) : en pratique tant que sqrt(4·D·t) reste petit devant la
    distance de x0 aux parois.

    Paramètres
    ----------
    t : float ou ndarray
        Instants (s).
    c_a0, c_b0 : float
        Concentrations initiales de A (à gauche) et de B (à droite) (µm⁻²).
    D : float
        Coefficient de diffusion commun à A et B (µm²/s).

    Retour
    ------
    float ou ndarray
        Déplacement du front x_f - x0 (µm), positif = vers la droite (côté B).
    """
    return np.sqrt(4 * D * t) * erfinv((c_a0 - c_b0) / (c_a0 + c_b0))


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

    # Test 3 — correction de bord : proportion de paires à distance < r_c parmi des paires
    # de points tirés uniformément dans une boîte, comparée à la formule de border_factor.
    # On prend une petite boîte (L = 10 µm) pour que la correction (≈ 16 %) soit bien
    # plus grande que l'incertitude du tirage.
    from config import SEED
    rng = np.random.default_rng(SEED)
    L_test, n_paires = 10.0, 1_000_000
    p1 = rng.uniform(0, L_test, size=(n_paires, 2))
    p2 = rng.uniform(0, L_test, size=(n_paires, 2))
    proportion = np.mean(np.sum((p1 - p2) ** 2, axis=1) < R_C ** 2)
    sans_paroi = np.pi * R_C ** 2 / L_test ** 2
    predite = sans_paroi * border_factor(R_C, L_test)
    incertitude = np.sqrt(proportion * (1 - proportion) / n_paires)  # écart-type binomial
    print(f"Test 3 — L = {L_test:g} µm, r_c = {R_C:g} µm : paires en collision, tirage = {proportion:.4f} "
          f"± {incertitude:.4f}, formule = {predite:.4f}, sans correction = {sans_paroi:.4f}")
    assert abs(proportion - predite) < 3 * incertitude
    print(f"         facteur de bord pour la boîte de référence (L = {L:g} µm) : {border_factor(R_C, L):.4f}")

    # Ordre de grandeur : temps de demi-réaction pour [A]₀ = [B]₀, t_1/2 = 1 / (k·[A]₀)
    print(f"Temps de demi-réaction 1/(k·[A]₀) = {1 / (k * c0[0]):.1f} s")

    # Test 4 — front de Gálfi-Rácz : immobile si [A]₀ = [B]₀ ; sinon, en √t vers B si [A]₀ > [B]₀.
    # Valeur à la main pour [A]₀ = 2·[B]₀, D = 10 µm²/s, t = 250 s : sqrt(4·10·250) = 100 µm,
    # erfinv(1/3) ≈ 0.3046, donc x_f - x0 ≈ 30.5 µm
    from config import D, L_SEPARE, N_A_SEPARE, N_B_SEPARE
    aire_demi_boite = L_SEPARE[0] / 2 * L_SEPARE[1]  # µm², A et B occupent chacun une moitié
    assert front_position(250.0, 0.05, 0.05, D) == 0
    assert front_position(250.0, 0.025, 0.05, D) < 0
    assert np.isclose(front_position(250.0, 0.05, 0.025, D), 30.46, atol=0.01)
    assert np.isclose(front_position(4 * 250.0, 0.05, 0.025, D), 2 * front_position(250.0, 0.05, 0.025, D))
    print(f"Test 4 — front de Gálfi-Rácz pour N_A = {N_A_SEPARE}, N_B = {N_B_SEPARE} : "
          f"x_f - x0 = {front_position(250.0, N_A_SEPARE / aire_demi_boite, N_B_SEPARE / aire_demi_boite, D):.1f}"
          f" µm à t = 250 s "
          f"(0 si [A]₀ = [B]₀, ×2 quand t est multiplié par 4)")
    print("Tous les tests passent.")
