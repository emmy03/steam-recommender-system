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
    df["rating"] = df["playtime_hours"].apply(convert_playtime_to_rating)

    matrice_interaction = df.pivot_table(
        index="user_id", columns="game_id", values="rating", fill_value=0
    )
    return matrice_interaction

def convert_playtime_to_rating(hours):
    """
    Transforme le temps de jeu en une note de 1 à 5.
    (Les seuils sont donnés à titre d'exemple, tu pourras les ajuster !)
    """
    if hours <= 0:
        return 0
    elif hours < 2:   # Testé rapidement
        return 1
    elif hours < 10:  # Un peu joué
        return 2
    elif hours < 30:  # Fini la trame principale
        return 3
    elif hours < 100: # Beaucoup investi
        return 4
    else:             # Hardcore fan (> 100h)
        return 5
