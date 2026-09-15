"""Regression checks for data loss guards and BM25 movie alignment."""
import contextlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from src.v1 import db_builder as builder
from src.v1.extract_data import interval, timestamp

class Encoder:
    def __init__(self,*args,**kwargs): pass
    def encode(self, docs, **kwargs): return np.ones((len(docs),4))

class Collection:
    def __init__(self): self.ids=[]
    def add(self, **kwargs): self.ids.extend(kwargs['ids'])

class Client:
    def __init__(self): self.deleted=[]; self.collections={}
    def delete_collection(self,name): self.deleted.append(name)
    def create_collection(self,name):
        self.collections[name]=Collection()
        return self.collections[name]

class RebuildTests(unittest.TestCase):
    def test_incomplete_media_cannot_delete_database(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(builder,'MOVIE_FOLDERS',tmp), patch.object(builder.chromadb,'HttpClient') as client:
            with self.assertRaisesRegex(RuntimeError,'incomplete'):
                builder.DatabaseBuilder().build_vector_db([{'title':'Missing'}])
            client.assert_not_called()

    def test_missing_folder_cannot_overwrite_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata=Path(tmp)/'metadata.json'
            metadata.write_text('[{"title":"Keep"}]')
            with patch.object(builder,'MOVIE_FOLDERS',str(Path(tmp)/'missing')),patch.object(builder,'CLEAN_EN_JSON_PATH',str(metadata)):
                with self.assertRaises(RuntimeError): builder.DatabaseBuilder().clean_and_translate()
            self.assertEqual(json.loads(metadata.read_text()),[{'title':'Keep'}])

    def test_dialogue_bm25_keeps_correct_movie_with_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            movies=[{'title':f'Movie_{i}','content':f'Summary for movie {i}'} for i in range(3)]
            keywords=['spaceship','jury','robot']
            for movie,word in zip(movies,keywords):
                folder=Path(tmp)/movie['title']; (folder/'picture').mkdir(parents=True); (folder/'script').mkdir()
                (folder/'manifest.json').write_text('{"status":"complete"}')
                (folder/'script/dialogue.txt').write_text(f'[00:00:00 -> 00:00:02]: {word} unique dialogue\n')
                Image.new('RGB',(10,10)).save(folder/'picture/frame_000001.jpg')
            client=Client(); output=str(Path(tmp)/'bm25.pkl')
            with patch.object(builder,'MOVIE_FOLDERS',tmp),patch.object(builder,'BM25_PATH',output),patch.object(builder,'SentenceTransformer',Encoder),patch.object(builder.chromadb,'HttpClient',return_value=client),patch.object(builder,'tokenize',lambda text:text.lower().split()),contextlib.redirect_stdout(io.StringIO()):
                builder.DatabaseBuilder().build_vector_db(movies)
            with open(output,'rb') as f: bm25,metas,docs=pickle.load(f)
            self.assertEqual((bm25.corpus_size,len(metas),len(docs)),(3,3,3))
            for i,word in enumerate(keywords):
                index=int(np.argmax(bm25.get_scores([word])))
                self.assertEqual(metas[index]['movie_name'],movies[i]['title'])
            self.assertEqual(len(client.collections['text_sbert_collection'].ids),6)

    def test_sampling_boundaries(self):
        self.assertEqual([interval(x) for x in [1799,1800,3599,3600,7200,7201]],[30,60,60,120,120,180])
        self.assertEqual(timestamp(3661.25),'01:01:01.250')

if __name__ == '__main__': unittest.main()
