# Cognitively-Inspired-Energy-Based-World-Models
Ce dépôt contient une lecture/explication détaillée et une réimplémentation en PyTorch ainsi qu'une reproduction des résultats du papier de recherche :   **"Cognitively Inspired Energy-Based World Models"** ([arXiv:2406.08862](https://arxiv.org/abs/2406.08862)).
## Présentation
Les modèles autorégressifs traditionnels (Transformers, RNNs) prédisent l'état suivant dans l'espace de sortie sans mécanisme d'évaluation de la plausibilité ni allocation dynamique de calcul.De plus la prédiction de l'état suivant n'est pas pris en compte dans la dynamique interne du processus. Ce projet explore les **Energy-Based World Models (EBWM)** et l'architecture **Energy-Based Transformer (EBT)** pour surmonter ces limites grâce à un objectif de reconstruction et une optimisation de l'énergie de l'état latent.

## Compétences
Ce mini projet évoque des notions qui peuvent à priori paraître ( sans vraiment l'être ), complexes ou nouvelles telles que : le NLP, les LLMs, la CV ( Computer Vision/Vision par ordinateur ), les Tranformers, la notion d'attention cusale, les modèles autoregressifs,les World models, les Energy-Based Models (EBM),les Energy-Based World Models (EBWM). La plupart de ces notions seront définies et expliqués dans ce projet dont le but est aussi de poser les bases mathématiques et théoriques autour des modèles du monde. Le projet est donc tout aussi adapté à un publique d'ébutant qu'à un publique expérimenté.

## Objectifs du Projet
* Implémenter l'architecture **Energy-Based Transformer (EBT)**.
* Reproduire les expériences comparatives contre les baselines autorégressives (Transformers / RNNs).
* Évaluer les performances sur la gestion de l'incertitude et le calcul dynamique.

## A propos de l'auteur
Je suis un jeune étudiant en mathématiques appliquées passionné de technologie. J'ai fait une licence en mathématique après des CPGE, un cycle ingénieur en mathématiques appliquées à l'Institut Galilée, et un master de recherche en apprentissage statistique à l'université Paris cité. Mes domaines d'intérêts couvrent les concepts d'apprentissage statistique, de vision par ordinateur et de modèles du monde. J'envisage de faire une thèse cifre sur les EBWM.
Je vous propose de consulter mon site web ici :


## 🛠️ Structure du Projet

```text
models/          # Architectures (EBT, Baselines Transformers/RNNs)
data/            # Scripts de chargement et prétraitement des données
training/        # Boucles d'entraînement et minimisation d'énergie
experiments/     # Scripts d'évaluation et reproduction des résultats
notebooks/
├── 01_intro_energy_based_models.ipynb    # C'est quoi un EBM, contrastive vs regularized
├── 02_EBT_architecture.ipynb              # L'architecture EBT, la matrice n×(n+1)
├── 03_MCMC_langevin.ipynb                 # La boucle MCMC, α, Langevin dynamics, avec visualisations
└── 04_results_ablations.ipynb             # Reproduction du Table 2, avec tes propres résultats

└── README.md

