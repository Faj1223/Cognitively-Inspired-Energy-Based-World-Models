# World Models : de la prédiction à la planification dans un espace latent

Implémentation progressive d'un world model au sens de LeCun (2022), sur une tâche contrôlée :
une balle qui rebondit dans une boîte 2D. Le projet construit l'architecture brique par brique,
en validant chaque étape avant d'ajouter la suivante.

```
s(t) = Enc(x(t))              →  perception
ŝ(t+1) = Pred(s(t), a(t))     →  modèle du monde
a* = argmin_a  coût(ŝ)        →  planification
```

## Motivation

Un agent qui apprend uniquement par essais et erreurs a besoin d'énormément d'interactions avec son
environnement : c'est lent, coûteux, parfois dangereux. Un humain sait qu'une chute d'une falaise
serait fatale sans avoir besoin d'essayer. Un world model permet d'apprendre la dynamique d'un
environnement, puis de planifier « en imagination » avant d'agir.

## Structure

| Notebook | Contenu | Encodeur | Action | Planification |
|---|---|---|---|---|
| `01_balle_passive.ipynb` | Prédire la dynamique à partir des positions | Fixé à la main | ✗ | ✗ |
| `02_balle_pilotable.ipynb` | Ajouter les actions, planifier vers une cible | Fixé à la main | ✓ | ✓ |
| `03_jepa.ipynb` | Apprendre la représentation à partir d'images | **Appris (JEPA)** | ✓ | En cours |

---

## Étape 1 — Prédire la dynamique

Un MLP prédit le déplacement de la balle à partir de ses 3 dernières positions. Les rebonds sont la
seule non-linéarité du problème. La baseline est l'extrapolation linéaire (« la balle refait son
dernier déplacement »), exacte loin des murs.

**Protocole** : 500 trajectoires, séparation *par trajectoire* (80/10/10), prédiction du déplacement
Δp plutôt que de la position, évaluation séparée loin des murs et autour des rebonds.

**Résultats** (erreur moyenne à un pas, sur le test) :

| Situation | Réseau | Extrapolation linéaire |
|---|---|---|
| Loin des rebonds | 2,0e-04 | 3,1e-18 *(exacte)* |
| Autour des rebonds | **6,7e-04** | 1,6e-02 |

Le réseau est **24 fois plus précis que la baseline autour des rebonds**. En rollout, ses trajectoires
imaginées suivent la vraie trajectoire sur plus de 100 pas en traversant plusieurs rebonds
(erreur de 0,11 à 100 pas, contre 3,09 pour la baseline, qui ignore les murs).

---

## Étape 2 — Agir et planifier

La balle reçoit à chaque pas une poussée qui modifie sa vitesse. Le modèle apprend l'effet des actions,
puis un agent l'utilise pour amener la balle vers une cible : il imagine 500 suites de 10 actions,
retient la meilleure, exécute sa première action, et replanifie au pas suivant
(*random shooting*, schéma MPC).

**Résultats du modèle** (erreur moyenne à un pas, sur le test) :

| Situation | Part | Réseau | Baseline (déplacement + action) |
|---|---|---|---|
| Mouvement libre | 81,5 % | 2,0e-04 | 4,1e-17 *(exacte)* |
| Vitesse saturée | 6,1 % | 1,4e-03 | 2,0e-03 |
| Autour des rebonds | 12,5 % | **3,3e-03** | 2,1e-02 |

**Résultats de la planification** (100 épisodes, cible atteinte si la balle passe à moins de 0,05) :

| Agent | Réussite | Pas moyens |
|---|---|---|
| Aléatoire | 36 % | 52,9 |
| Glouton (pousse vers la cible) | 54 % | 26,2 |
| **World model** | **100 %** | **17,1** |
| Oracle (vraie physique) | 100 % | 18,0 |

Le world model **planifie aussi bien que la vraie physique**. L'agent glouton échoue une fois sur
deux : la balle a de l'élan et dépasse la cible, ce qui montre la nécessité d'anticiper. Les rollouts
du modèle divergent à long terme, mais la replanification à chaque pas depuis la position réelle
empêche les erreurs de s'accumuler.

---

## Étape 3 — JEPA : apprendre la représentation (en cours)

L'agent ne reçoit plus de coordonnées, mais des **images** 32×32. Un encodeur convolutif apprend
seul une représentation latente (dimension 32), et le prédicteur travaille entièrement dans cet
espace. L'entraînement combine une énergie de prédiction latente et la régularisation
**VICReg** (variance + covariance) contre le collapse.

**Pas de collapse.** L'écart-type des latents reste stable autour de 1,06, et le rang effectif de leur
covariance atteint **32,0 / 32** : aucune direction du latent n'est inutilisée.

**Le latent contient l'état physique de la balle**, mesuré par des sondes entraînées *après coup*
sur un encodeur figé (les vraies coordonnées ne sont jamais vues par la JEPA) :

| Sonde | R² position (x, y) | R² vitesse (vx, vy) | Erreur de position |
|---|---|---|---|
| Linéaire — encodeur aléatoire | 0,41 / 0,50 | 0,00 / 0,00 | 0,260 |
| Linéaire — encodeur JEPA | 0,95 / 0,99 | 0,06 / 0,01 | 0,063 |
| **MLP — encodeur JEPA** | **1,00 / 1,00** | **0,91 / 0,90** | **0,014** |

L'encodeur retrouve la position à **moins d'un demi-pixel** et la vitesse avec un R² de 0,91, à partir
des seules images. La position est **linéairement** accessible, la vitesse ne l'est pas : rien
n'impose à l'encodeur de lui réserver une direction du latent, puisque le prédicteur, non linéaire,
peut l'en extraire. Une sonde linéaire qui échoue mesure donc l'accessibilité d'une information,
pas sa présence.

**Prochaines étapes** : rollouts dans l'espace latent, puis planification à partir d'une *image* de
la cible, l'agent n'ayant jamais accès aux coordonnées.

---

## Choix méthodologiques

- **Séparation par trajectoire** (jamais par pas), pour que les trajectoires de test soient réellement
  inédites.
- **Baselines systématiques** : extrapolation linéaire, agent glouton, oracle disposant de la vraie
  physique, encodeur aléatoire. Un résultat isolé ne se juge pas.
- **Prédiction des variations** (Δp, Δs) plutôt que des états : sur de petits déplacements, prédire
  l'état permet d'obtenir une loss faible en recopiant simplement l'entrée.
- **Statistiques VICReg sur batchs mélangés** : calculées par batch, les termes de variance et de
  covariance sont biaisés si un batch contient des transitions consécutives d'une même trajectoire.
  Ce point, découvert en cours de route, expliquait un écart de loss trompeur en validation.

## Lancer le projet

```bash
pip install torch numpy matplotlib scikit-learn
```

Les notebooks 1 et 2 s'exécutent sur CPU en quelques minutes. Le notebook 3 nécessite un GPU
(testé sur Colab T4, environ 5 minutes d'entraînement).

## Références

- LeCun, *A Path Towards Autonomous Machine Intelligence*, 2022
- Bardes, Ponce, LeCun, *VICReg: Variance-Invariance-Covariance Regularization*, ICLR 2022
- Assran et al., *Self-Supervised Learning from Images with a JEPA (I-JEPA)*, CVPR 2023
- Ha, Schmidhuber, *World Models*, 2018