import os
from dotenv import load_dotenv

# Charge les variables cachées
load_dotenv()

# --- SECRETS ---
STEAM_API_KEY = os.getenv("STEAM_API_KEY")

# --- PARAMETRES UTILISATEUR ---
MON_STEAM_ID = 76561198305752421
MAX_USERS = 100
TOP_N = 5

# --- CHEMINS DE FICHIERS ---
DATA_PATH = "data/steam_raw_data.csv"
