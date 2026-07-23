import sys
from src.v3.search_engine import MovieSearchEngine

engine = MovieSearchEngine()
query = "a computer hacker learning from mysterious rebels about the true nature of his reality"

old_predict = engine.rerank_model.predict
def debug_predict(pair):
    score = old_predict(pair)
    if "The Matrix" in pair[1] or "Inception" in pair[1]:
        print(f"DEBUG RERANK: Movie={pair[1][:80]}... Score={score}")
    return score
engine.rerank_model.predict = debug_predict

res = engine.search(query, system_type="PT3", top_n=5)
print("PT3 RESULT:", res)
