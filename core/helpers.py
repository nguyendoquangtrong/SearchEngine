import re

def make_folder_name(title):
    """Lọc bỏ ký tự đặc biệt để map với tên Folder trong Win/Mac"""
    clean_title = re.sub(r'[^\w\s-]', '', str(title))
    return re.sub(r'[-\s]+', '_', clean_title).strip('_')

def normalize_name(name):
    """Viết thường và xóa mọi khoảng trắng để làm ID"""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())

def tokenize(text):
    """Tokenize locally for BM25; does not download NLTK resources at runtime."""
    stop_words = {"i", "am", "is", "are", "the", "a", "an", "of", "to", "in", "and", "you", "it", "that", "for", "on", "with"}
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    return [t for t in tokens if t.isalnum() and t not in stop_words]
