# 🎮 Steam Recommender System

Système de recommandation de jeux vidéo Steam, construit à partir de vraies données de bibliothèques de joueurs (temps de jeu) récupérées via l'API Steam. Ce projet est actuellement en cours de développement. Il implémente les fondations du **Filtrage Collaboratif** classique (**User-Based** et **Item-Based**) et s'apprête à accueillir des modèles plus avancés (SVD, Hybride).

---

## Fonctionnalités actuelles

### 1. Extraction des données (`src/etl/extractor.py`)
- Parcours en largeur (BFS) du graphe social Steam à partir d'un `SteamID` de départ, en suivant les listes d'amis.
- Récupération des bibliothèques de jeux (`GetOwnedGames`) et du réseau social (`GetFriendList`) via l'API Steam Web.
- Filtrage des profils privés et des jeux jamais lancés (bruit statistique).
- Export en CSV (`data/steam_raw_data.csv`) : `user_id`, `game_id`, `playtime_minutes`.

### 2. Préparation des données (`src/models/data_utils.py`)
- Conversion du temps de jeu (minutes → heures) en note de **1 à 5** selon des seuils d'engagement (testé rapidement / fini le jeu / hardcore fan...).
- Construction de la matrice d'interaction Utilisateur × Jeu (pivot table).

### 3. Moteurs de recommandation (`src/models/`)
| Moteur | Fichier | Principe |
|---|---|---|
| **User-Based** | `user_based.py` | Recommande les jeux appréciés par des joueurs au profil similaire au tien. |
| **Item-Based** | `item_based.py` | Recommande des jeux mathématiquement proches de ceux que tu as déjà aimés. |

Deux métriques de similarité disponibles pour User-Based et Item-Based : **cosinus** et **corrélation de Pearson**.

---

## Installation

```bash
# Cloner le repo
git clone <url-du-repo>
cd steam-recommender-system

# Installer les dépendances (via uv)
uv sync

# Configurer la clé API
cp .env.example .env   # puis renseigner STEAM_API_KEY
```

Une clé API Steam gratuite est disponible sur [steamcommunity.com/dev/apikey](https://steamcommunity.com/dev/apikey).

⚠️ Le crawler ne fonctionne que sur des profils Steam **publics**.

---

## Utilisation

Pour l'instant, les scripts fonctionnent de manière autonome.

```bash
# 1. Extraire les données (bibliothèques + réseau d'amis)
uv run -m src.etl.extractor.py

# 2. Tester un modèle de recommandation individuellement
uv run -m src.models.user_based.py
uv run -m src.models.item_based.py
```

---

## Structure du projet

```
.
├── src/
│   ├── etl/   
│   │   └── extractor.py            # Pipeline d'extraction (API Steam, BFS)
│   ├── config.py                   # Constantes & variables d'environnement
│   └── models/
│       ├── data_utils.py           # Préparation de la matrice d'interaction
│       ├── user_based.py           # Moteur User-Based
│       └── item_based.py           # Moteur Item-Based
└── data/                           # Données extraites (ignoré par git)
```

---

## En cours de développement (Roadmap)

Le projet évolue activement ! Voici ce qui tourne déjà en local et sera pushé très prochainement :

- [x] **Moteur Hybride :** Combinaison User-Based et Item-Based via une moyenne pondérée normalisée (Min-Max).

- [x] **Moteur SVD (Facteurs Latents) :** Factorisation de matrice (`TruncatedSVD`) façon Netflix Prize.

- [x] **Interface CLI unifiée (`main.py`) :** Un menu interactif pour choisir facilement son moteur et sa métrique.

- [x] **Système de Cache :** Mise en cache (`joblib`) intelligente des matrices de similarité avec empreinte MD5 des données pour éviter les recalculs.

- [x] **Gestion du Cold Start :** Recommandations à la volée pour un `SteamID` non présent dans la base d'entraînement en récupérant ses jeux en temps réel.

- [x] **Pipeline d'évaluation (Leave-k-out) :** Calcul de Precision@K, Recall@K, et Coverage pour comparer la pertinence mathématique des différents moteurs.

- [ ] Interface web (`app.py`) avec Streamlit ou FastAPI.

- [ ] Tests unitaires (`pytest`).

---

## Stack technique

Python 3.14 · pandas · scikit-learn · scipy · requests · uv (gestion des dépendances)

---

## Licence

MIT
