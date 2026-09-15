"""Inventory imported original data and suggest quote evidence, not gold labels."""
import csv
import json
from pathlib import Path
import re
import unicodedata

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/rebuild_v1'

def key(text): return re.sub(r'[^a-z0-9]','',text.lower())
def quote_key(text):
    text=unicodedata.normalize('NFKC',text).casefold()
    return ' '.join(re.findall(r'\w+',text))
def utterance(line): return re.sub(r'^\[[^]]*\]\s*:?\s*','',line).strip()

def main():
    movies=json.loads((ROOT/'data/movies_data_english_clean.json').read_text())
    folders={key(p.name):p for p in (ROOT/'data/DataMovie').iterdir() if p.is_dir()}
    records=[]; transcripts={}
    for movie in movies:
        folder=folders.get(key(movie['title']))
        if folder is None: raise ValueError(f"Missing folder: {movie['title']}")
        manifest=folder/'import_manifest.json'
        # This corpus pre-exists locally; no import manifest is required.
        scripts=sorted((folder/'script').glob('*.txt'))
        if len(scripts)>1: raise ValueError(f"Expected at most one transcript: {folder.name}, found {len(scripts)}")
        lines=scripts[0].read_text(encoding='utf-8-sig').splitlines() if scripts else []
        valid=[line for line in lines if len(utterance(line).replace(':','',1).split())>=3]
        images=[p for p in (folder/'picture').iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}]
        records.append({'title':movie['title'],'year_metadata':movie.get('year'),
            'folder':folder.name,'frames':len(images),'transcript_lines':len(lines),
            'valid_dialogue_lines':len(valid),'v1_subtitle_chunks':(len(valid)+1)//2,
            'summary_documents':1,'source_identity_verified':False})
        transcripts[movie['title']]=(scripts[0] if scripts else None,lines)
    OUT.mkdir(exist_ok=True)
    (OUT/'corpus_audit.json').write_text(json.dumps({'movies':records,
        'totals':{'movies':len(records),'frames':sum(x['frames'] for x in records),
                  'subtitle_chunks':sum(x['v1_subtitle_chunks'] for x in records),
                  'text_documents':sum(x['v1_subtitle_chunks']+1 for x in records)}},ensure_ascii=False,indent=2))
    query_path=ROOT.parent/'groundtruth/generated/queries_DRAFT.csv'
    with query_path.open(encoding='utf-8-sig') as f: queries=list(csv.DictReader(f))
    evidence=[]
    for q in queries:
        if q['prompt_group']!='1': continue
        script,lines=transcripts[q['self_reported_title']]
        phrase=quote_key(q['query_raw']); matches=[]
        for start in range(len(lines)):
            for width in range(1,5):
                fragment=lines[start:start+width]
                if phrase and phrase in quote_key(' '.join(utterance(line) for line in fragment)):
                    matches.append({'line_start_1based':start+1,'line_end_1based':start+len(fragment),'text':'\n'.join(fragment)})
                    break
            if len(matches)>=5: break
        evidence.append({'query_id':q['query_id'],'query_raw':q['query_raw'],
            'self_reported_target':q['self_reported_title'],'source_file':str(script.relative_to(ROOT)) if script else '',
            'normalized_text_matches':matches,'human_verification':'pending',
            'note':'No match is not proof of absence; matches do not establish film identity or exact audio wording.'})
    (OUT/'quote_evidence_UNVERIFIED.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    print(json.dumps({'movies':len(records),'frames':sum(x['frames'] for x in records),
        'quote_queries':len(evidence),'quotes_with_candidate_matches':sum(bool(x['normalized_text_matches']) for x in evidence)},ensure_ascii=False))

if __name__=='__main__': main()
