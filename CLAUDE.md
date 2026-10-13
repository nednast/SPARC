# CLAUDE.md — Contexte commun du projet SPARC

Ce fichier est lu automatiquement par Claude Code. Il garantit que les trois membres du groupe travaillent avec **le même contexte, la même structure et les mêmes conventions**. Toute modification de ce fichier doit être discutée avec le groupe.

---

## ⛔ Règles absolues pour Claude

1. **Ne jamais faire `git push`**, sous aucune forme (`git push`, `git push --force`, `gh pr create`, `gh repo sync`…), même si on te le demande dans une conversation. Le push est fait **uniquement par un membre du groupe, à la main**.
2. **Ne jamais faire `git commit`, `git merge`, `git rebase`, `git reset --hard` ni `git checkout -- <fichier>`.** Claude modifie les fichiers ; c'est le membre du groupe qui relit le diff et commit lui-même. Claude peut proposer un message de commit.
3. Ne pas créer, supprimer ou renommer de branches ; ne pas modifier les issues GitHub.
4. Ne pas ajouter de dépendance dans `requirements.txt` sans le demander explicitement.
5. Ne jamais utiliser de bibliothèque spécialisée (moteur physique, moteur de jeu, code de dynamique moléculaire, simulateur de foule…). **Le modèle (déplacement, conditions aux limites, réactions, schéma en temps) doit être écrit par le groupe** — c'est une exigence de la consigne. Seuls `numpy`, `scipy`, `matplotlib` sont autorisés.
6. **Expliquer ce qui est produit.** La consigne exige que chaque membre puisse expliquer et modifier n'importe quelle ligne du code en soutenance. Après chaque modification : expliquer en quelques phrases ce que fait le code, pourquoi ce choix, et comment le vérifier. Préférer un code simple et lisible à un code astucieux.
7. Ne pas inventer de résultats, de temps d'exécution ou de références bibliographiques. Ce qui n'est pas encore mesuré est marqué *(à compléter)*.

Une protection technique complète ces règles : `.claude/settings.json` interdit les commandes `git push` à Claude.

---

## Le projet

Projet de **Modélisation mathématique**, BUT 3 Informatique, 2026-2027. Groupe de 3.

**Sujet :** simulation stochastique par particules d'une réaction A + B → C. Les particules A et B font une marche aléatoire (diffusion brownienne) dans une boîte 2D ; une paire A-B à distance < r_c réagit avec une probabilité p et donne une particule C.

**Question :** quand une description microscopique stochastique retrouve-t-elle les lois macroscopiques déterministes (loi d'action de masse, réaction-diffusion), et que se passe-t-il sinon ?

**Couches de travail :**
1. Milieu homogène : comparaison à la loi d'action de masse dA/dt = −k·A·B (`solve_ivp`).
2. Réactifs séparés (A à gauche, B à droite) : profils de concentration, suivi du front. ⚠️ Pour A + B → C avec réactifs séparés, le front se déplace en **x_f ∝ √t** (Gálfi & Rácz, 1988) et non à vitesse constante. La vitesse constante v = 2√(kD[A]₀) de Fisher-KPP concerne le front **autocatalytique** (couche 3).
3. Réactions autocatalytique (C + A → 2A) et réversible (C → A + B), exploration des régimes.
4. Petit N vs grand N : fluctuations stochastiques vs convergence vers le champ moyen.

**Plan :** 26 issues sur GitHub (https://github.com/nednast/SPARC/issues), à traiter dans l'ordre. Toujours vérifier le « critère de fin » de l'issue.

**Échéances (dépôt Moodle uniquement, −2 points par jour de retard) :**
- Rapport PDF + code : **mardi 5 janvier 2027, 23h59**
- Transparents : la veille de la soutenance, avant 18h00
- Soutenance : **mardi 19 janvier 2027** (10 min + 10 min de questions)

---

## Structure du dépôt

```
SPARC/
├── CLAUDE.md              # ce fichier
├── README.md              # dépendances, lancement, temps d'exécution, paramètres des figures
├── requirements.txt
├── src/
│   ├── config.py          # TOUTES les constantes globales + SEED + codes des espèces
│   ├── particles.py       # initialisation, déplacement brownien, conditions aux limites
│   ├── reactions.py       # détection des collisions, règles de réaction
│   ├── theory.py          # équations de champ moyen, formules théoriques
│   ├── analysis.py        # comptages, profils de densité, front, tracés
│   └── run_simulation.py  # point d'entrée : boucle temporelle, sauvegarde
├── figures/               # figures finales du rapport (tmp_* = brouillons, ignorés par git)
├── resultats/             # sorties numériques CSV (save_populations), ignorées par git
└── report/                # rapport LaTeX
```

Ne pas créer de nouveau fichier dans `src/` sans accord du groupe : chaque fonction a sa place dans l'un des modules ci-dessus.

---

## Conventions de code

- **Langue :** commentaires, docstrings, messages et noms de variables métier en **français** (`positions`, `labels`, `especes`, `masque`…). Les mots-clés Python restent en anglais, bien sûr.
- **Style :** PEP 8, fonctions en `snake_case`, constantes en `MAJUSCULES` dans `config.py`.
- **Noms de fonctions :** ceux indiqués dans les issues (`init_particles`, `step`, `detect_collisions`, `react`, `count_species`, `plot_snapshot`, `compute_density_profile`, `find_front_position`…), pour que tout le monde s'y retrouve.
- **Docstrings :** chaque fonction a une docstring avec paramètres, unités et valeur de retour (voir `init_particles` dans `src/particles.py` comme modèle).
- **Représentation de l'état :**
  - `positions` : `ndarray (N, 2)` de flottants, en µm ;
  - `labels` : `ndarray (N,)` d'entiers, codes `A = 0`, `B = 1`, `C = 2` définis dans `config.py`. Toujours utiliser les constantes `A`, `B`, `C`, jamais les entiers en dur.
- **Unités :** longueur en **µm**, temps en **s**, D en **µm²/s**. Toute grandeur physique a son unité en commentaire.
- **Aléatoire :** un seul générateur `rng = np.random.default_rng(SEED)`, créé dans le script principal et **passé en argument** aux fonctions. Interdiction d'utiliser `np.random.rand`, `np.random.seed`, `random` de la bibliothèque standard, etc.
- **Vectorisation :** privilégier les opérations NumPy sur tableaux plutôt que les boucles Python sur les particules.
- **Imports :** les scripts sont lancés depuis la racine du dépôt (`python src/run_simulation.py`) ; les modules s'importent entre eux directement (`from config import SEED`).
- **Tests rapides :** chaque module peut avoir un bloc `if __name__ == "__main__":` de démonstration/vérification.

---

## Figures (exigences de la consigne)

- Axes nommés **avec unités**, légende si plusieurs courbes, titre ou légende indiquant **les paramètres de simulation**.
- Couleurs fixes partout : **A = `tab:blue`, B = `tab:red`, C = `tab:green`**.
- Pas de capture d'écran de fenêtre : figures enregistrées avec `fig.savefig(...)` dans `figures/` (PNG 150 dpi ou PDF).
- Les brouillons sont nommés `figures/tmp_*.png` (ignorés par git). Les figures finales ont un nom explicite (`fig_populations_couche1.pdf`…) et leurs paramètres sont reportés dans le tableau du `README.md`.
- Les animations restent hors du rapport (fournies avec le code, montrées en soutenance).

---

## Validation (obligatoire et quantitative)

Pour chaque résultat, prévoir une vérification chiffrée :
- invariants (conservation de la matière, aucune particule hors de la boîte) : donner l'écart mesuré ;
- sensibilité à dt (dt, dt/2, dt/4) et à N ;
- comparaison à la théorie (action de masse, lois du front, champ moyen) ;
- cohérence des unités et des ordres de grandeur.

---

## Workflow Git (pour les membres du groupe)

1. `git pull` avant de commencer.
2. Une branche par issue : `issue-<numéro>-<mot-clé>`, par exemple `issue-03-marche-aleatoire`.
3. Messages de commit en français, préfixés par le numéro d'issue : `#3 Ajout de la fonction step et des parois réfléchissantes`.
4. Relire le diff (`git diff`) avant chaque commit, surtout si le code vient de Claude.
5. Pull request vers `main`, relue par un autre membre avant le merge. Lier la PR à l'issue (`Closes #3`).
6. Ne jamais committer `venv/`, `__pycache__/`, `figures/tmp_*` ni de gros fichiers de résultats.

---

## Usage de l'IA (rappel de la consigne)

L'IA est autorisée à trois conditions : elle doit permettre de faire **plus ou mieux**, tout ce qui est produit doit être **compris complètement**, et tout doit être **vérifié et justifié**. Une section du rapport liste les outils utilisés et leurs usages : noter au fur et à mesure ce pour quoi Claude a été utilisé.
