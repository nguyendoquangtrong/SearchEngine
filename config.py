import os
from dotenv import load_dotenv

# Tự động tìm và load các biến từ file .env vào hệ thống
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Lấy cấu hình ChromaDB từ file .env (nếu không có thì mặc định là localhost:8000)
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))

# Cấu hình nguồn dữ liệu
MOVIE_FOLDERS = os.path.join(BASE_DIR, "DataMovie")
RAW_JSON_PATH = os.path.join(BASE_DIR, "movies_data.json")
CLEAN_EN_JSON_PATH = os.path.join(BASE_DIR, "movies_data_english_clean.json")
BM25_PATH = os.path.join(BASE_DIR, "bm25_index.pkl")