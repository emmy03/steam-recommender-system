import pandas as pd

def prepare_data(file_path):
    """
    Charge les données brutes et construit la matrice d'intéraction Utilisateur-Jeu.

    Args:
        file_path (str): Le chemin vers le fichier CSV contenant les données brutes.

    Returns:
        matrice_interaction (pd.DataFrame): Une matrice pivot (lignes = utilisateurs, colonnes = jeux, valeurs = heures jouées).
    """
    df = pd.read_csv(file_path)
    # Transformation des minutes en heures pour avoir des scores plus lisibles
    df["playtime_hours"] = df["playtime_minutes"] / 60

    matrice_interaction = df.pivot_table(
        index="user_id", columns="game_id", values="playtime_hours", fill_value=0
    )
    return matrice_interaction
