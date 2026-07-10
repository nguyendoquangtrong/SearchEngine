import sys
import os

from src.v2.search_engine import MovieSearchEngine

if __name__ == "__main__":
    engine = MovieSearchEngine()
    print("\n--- Testing V2 ---")
    query = "a glowing mechanical suit flying in the sky"
    print(f"Query: {query}")
    results = engine.search(query)
    print("Results:", results)
