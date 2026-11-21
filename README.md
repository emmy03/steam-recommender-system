# 🎮 Steam Recommender System

Système de recommandation de jeux vidéo Steam, construit à partir de vraies données de bibliothèques de joueurs (temps de jeu) récupérées via l'API Steam. Ce projet est actuellement en cours de développement. Il implémente les fondations du **Filtrage Collaboratif** classique (**User-Based** et **Item-Based**) et s'apprête à accueillir des modèles plus avancés (SVD, Hybride).

---

## Fonctionnalités actuelles

### 1. Extraction des données (`src/etl/extractor.py`)
- Parcours en largeur (BFS) du graphe social Steam à partir d'un `SteamID` de départ, en suivant les listes d'amis.
- Récupération des bibliothèques de jeux (`GetOwnedGames`) et du réseau social (`GetFriendList`) via l'API Steam Web.
- Conservation des jeux jamais lancés pour identifier la bibliothèque complète.
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
uv run python -m src.etl.extractor

# 2. Tester un modèle de recommandation individuellement
uv run python main.py
uv run python -m unittest discover -s tests -v
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

- [x] **Moteur SVD (Facteurs Latents) :** Factorisation de matrice (`TruncatedSVD`) sur la matrice d’engagement.

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


## Premier lot de fiabilisation

Le point d'entrée recommandé est `uv run python main.py`. Il récupère la
bibliothèque actuelle du joueur avant de proposer des jeux non possédés.
Les jeux possédés à zéro minute sont exclus eux aussi. Une clé Steam valide
et une bibliothèque accessible sont nécessaires ; une erreur API interrompt
la recommandation au lieu de supposer que le joueur ne possède aucun jeu.
Les seuls jeux pouvant être exclus sont ceux accessibles via l'API.

L'ancien CSV reste utilisable : il sert aux préférences, tandis que la liste
d'exclusion vient de l'API. La prochaine extraction conservera aussi les jeux
jamais lancés. L'extraction remplace `data/steam_raw_data.csv` : sauvegarder
l'ancien fichier avant de relancer la collecte si l'on veut le conserver.

Les fonctions de modèles acceptent `owned_games`, une collection d'AppID à
exclure avant le classement final. Les appels directs sans ce paramètre
excluent seulement les jeux joués : ils servent notamment à l'évaluation.
L'évaluation existante n'a pas encore été adaptée à la possession complète ;
ne pas interpréter ses résultats comme une mesure d'achat futur.

Tests hors ligne : `uv run python -m unittest discover -s tests -v`.
Les appels réseau sont simulés. Le premier lot a été vérifié sur Python 3.12
avec les bibliothèques disponibles dans l'environnement de revue, pas avec
l'environnement verrouillé Python 3.14 du projet. L'API réelle reste à tester
avec votre clé. Aucun secret n'est fourni.
