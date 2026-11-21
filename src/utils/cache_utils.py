"""
Cache disque pour les matrices de similarité (coûteuses à recalculer).

La clé de cache inclut un hash du fichier de données source : si le CSV
change (nouvelle extraction), le cache est automatiquement invalidé et la
matrice recalculée.
"""

import os
import hashlib
import joblib

CACHE_DIR = "cache"


def _hash_file(file_path):
    """Empreinte MD5 du fichier de données, utilisée comme clé de cache."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:10]


def get_or_compute(cache_name, data_path, cache_metric, compute_fn, *args, **kwargs):
    """
    Charge une matrice depuis le cache si elle existe et correspond aux
    données actuelles, sinon la calcule via compute_fn et la sauvegarde.

    Args:
        cache_name (str): Nom logique de la matrice ("sim_users", "sim_items"...).
        data_path (str): Chemin du CSV source (sert à générer la clé de cache).
        cache_metric (str): Métrique utilisée ("cosine"/"pearson"), incluse dans la clé.
        compute_fn (callable): Fonction à appeler si le cache est absent/invalide.
        *args, **kwargs: Passés tels quels à compute_fn.

    Returns:
        pd.DataFrame: La matrice, depuis le cache ou fraîchement calculée.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    data_hash = _hash_file(data_path)
    cache_path = os.path.join(
        CACHE_DIR, f"{cache_name}_{cache_metric}_{data_hash}.joblib"
    )

    if os.path.exists(cache_path):
        print(f"Chargement de '{cache_name}' depuis le cache...")
        return joblib.load(cache_path)

    print(f"Calcul de '{cache_name}' (pas de cache valide trouvé)...")
    resultat = compute_fn(*args, **kwargs)
    joblib.dump(resultat, cache_path)
    return resultat
