import re
import nltk

# Tải bộ băm từ (Chỉ chạy ngầm 1 lần)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def make_folder_name(title):
    """Lọc bỏ ký tự đặc biệt để map với tên Folder trong Win/Mac"""
    clean_title = re.sub(r'[^\w\s-]', '', str(title))
    return re.sub(r'[-\s]+', '_', clean_title).strip('_')

def normalize_name(name):
    """Viết thường và xóa mọi khoảng trắng để làm ID"""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())

def tokenize(text):
    """Băm câu thành từ khóa cho thuật toán BM25"""
    stop_words = {"i", "am", "is", "are", "the", "a", "an", "of", "to", "in", "and", "you", "it", "that", "for", "on", "with"}
    tokens = nltk.word_tokenize(str(text).lower())
    return [t for t in tokens if t.isalnum() and t not in stop_words]