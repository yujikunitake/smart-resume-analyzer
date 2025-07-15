# app/config.py
"""Configurações do projeto carregadas de variáveis de ambiente"""

import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI: str = os.getenv("MONGO_URI")
