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
L'évaluation retire les jeux masqués du filtre de possession. Ses résultats
ne constituent pas une mesure d'achat futur.

Tests hors ligne : `uv run python -m unittest discover -s tests -v`.
Les appels réseau sont simulés. Le premier lot a été vérifié sur Python 3.12
avec les bibliothèques disponibles dans l'environnement de revue, pas avec
l'environnement verrouillé Python 3.14 du projet. L'API réelle reste à tester
avec votre clé. Aucun secret n'est fourni.


## Évaluation reproductible (deuxième lot)

Depuis la racine du projet :

```bash
# Vérification rapide : cinq utilisateurs évalués par découpage,
# tout le dataset restant utilisé pour l'entraînement après masquage.
uv run python -m src.evaluate --max-users 5 --output-dir reports/evaluation_rapide

# Comparaison complète : tous les utilisateurs éligibles, trois découpages.
uv run python -m src.evaluate

# Un seul découpage, si la comparaison complète est trop longue.
uv run python -m src.evaluate --seeds 42 --output-dir reports/evaluation_seed42

# Tests sans API et sans téléchargement de données.
uv run python -m unittest discover -s tests -v
```

Les cinq méthodes (Popularité, User-Based, Item-Based, Hybride, SVD) utilisent
exactement le même entraînement et les mêmes utilisateurs de test pour une
seed donnée. Les seeds par défaut sont 42, 43 et 44. On masque 20 % des jeux
joués des utilisateurs ayant au moins cinq jeux, en gardant au moins un jeu
joué dans l'entraînement. Ce découpage est aléatoire, pas chronologique.
Les utilisateurs non éligibles participent uniquement à l'entraînement.

- Precision@10 : nombre de jeux masqués retrouvés / 10, même si moins de
  dix jeux sont proposés.
- Recall@10 : proportion des jeux masqués retrouvés.
- NDCG@10 : qualité de l'ordre, avec pertinence binaire. Retrouver un jeu
  masqué en première position rapporte davantage qu'en dixième position.
- Coverage : proportion du catalogue initial recommandée au moins une fois.
- Taux d'utilisateurs servis : proportion recevant au moins une proposition.

Les utilisateurs sans recommandation comptent avec des scores nuls. Les
résultats individuels sont moyennés par utilisateur, sans pondération par la
taille de sa bibliothèque. Les doublons dans un top-K sont rejetés.

La popularité compte les joueurs ayant joué à un jeu dans l'entraînement,
jamais dans le test. Tous les moteurs excluent les jeux possédés non masqués
connus du CSV, dont ceux à zéro minute. L'ancien CSV ne contient pas ces
derniers : l'évaluation ne peut pas reconstruire cette information manquante.
Elle n'appelle pas l'API. Après une nouvelle collecte, ce filtre sera plus
complet et les résultats pourront changer.

Un jeu sans interaction positive restante dans l'entraînement n'est candidat
pour aucun moteur. S'il était masqué, il reste dans le dénominateur du rappel
et du NDCG : il est compté comme un échec. Les nombres de jeux sans signal et
d'interactions de test concernées sont exportés pour rendre cette difficulté
visible. Les scores SVD sont calculés une seule fois par découpage, avec la
même fonction que la CLI. Le cache de production n'est pas utilisé.

Trois fichiers sont écrits dans `reports/evaluation/` par défaut :

- `resultats_par_decoupage.csv` : résultats par moteur et seed, effectifs et
  nombres d'interactions sans signal d'entraînement ;
- `synthese.csv` : moyenne et écart-type échantillonnal entre découpages
  (écart-type vide avec une seule seed) ;
- `parametres.json` : paramètres, dimensions, versions et empreinte SHA-256 du CSV.

Le dossier choisi est réutilisé et ces trois fichiers sont remplacés à chaque
exécution. Utiliser un autre `--output-dir` pour conserver une expérience.
L'ancien `data/evaluation_resultats.csv` est conservé mais correspond à
l'ancien protocole : ne pas le confondre avec les nouveaux résultats.

Le mode rapide vérifie le fonctionnement et ne suffit pas à désigner le
meilleur moteur. Les boucles de scoring et la matrice dense jeu-jeu rendent
encore l'évaluation complète coûteuse. Leurs optimisations restent à faire.
Les paramètres des moteurs sont fixes ici ; avant de les ajuster, prévoir
un jeu de validation distinct et conserver un test final non utilisé pour
les choisir. Les écarts-types entre seeds ne sont pas des intervalles de
confiance sur l'ensemble de la population Steam.
