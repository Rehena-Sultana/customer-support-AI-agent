import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

BASE_MODEL = os.getenv("SUPPORT_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_DIR = os.getenv("SUPPORT_ADAPTER_DIR", str(ROOT_DIR / "customer_support_qwen25_15b_lora"))
DB_PATH = os.getenv("SUPPORT_DB_PATH", str(ROOT_DIR / "data" / "support.db"))
KNOWLEDGE_DIR = os.getenv("SUPPORT_KNOWLEDGE_DIR", str(ROOT_DIR / "data" / "knowledge"))
CHROMA_DIR = os.getenv("SUPPORT_CHROMA_DIR", str(ROOT_DIR / "data" / "chroma_db"))
EMBEDDING_MODEL = os.getenv("SUPPORT_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
USE_ADAPTER = os.getenv("SUPPORT_USE_ADAPTER", "1").strip().lower() in {"1", "true", "yes"}
DISABLE_MODEL = os.getenv("SUPPORT_DISABLE_MODEL", "0").strip().lower() in {"1", "true", "yes"}
