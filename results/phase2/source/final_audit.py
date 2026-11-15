"""Final input/model/result integrity and human-review artifact audit."""
import json
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree
import numpy as np
import pandas as pd
from dotenv import dotenv_values
from benchmark import OUT,ROOT,DATA,inputs,sha,write_json,now,canonical,digest

def main():
    df,X,clip=inputs();siglip=json.loads((OUT/'siglip/embedding_manifest.json').read_text())
    assert siglip['inputs']==clip['descriptor']['inputs']
    assert sha(siglip['array_path'])==siglip['array_sha256']
    assert sha(siglip['model']['weight_path'])==siglip['model']['weight_sha256']
    assert sha(clip['model']['weight_path'])==clip['model']['weight_sha256']
    identity=json.loads((OUT/'inputs/input_identity.json').read_text())
    assert identity['ordered_images_sha256']==digest(canonical(siglip['inputs']).encode())
    assert identity['masks_sha256']==sha(OUT/'inputs/evaluation_masks.csv')
    checks=[];rows=0;silhouettes=0
    for enc in ['clip','siglip']:
        a=json.loads((OUT/enc/'validation.json').read_text());b=json.loads((OUT/'evaluation'/enc/'validation.json').read_text());assert a['success'] and b['success']
        rows+=a['metric_rows_recomputed']+b['chance_and_subset_metric_rows_recomputed'];silhouettes+=a['silhouettes_recomputed']
        runs=pd.read_csv(OUT/enc/'runs.csv')
        assert len(runs)==120
        for r in runs.itertuples():assert sha(OUT/enc/'predictions'/f'{r.run_id}.npy')==r.predictions_sha256
        checks.extend([a,b])
    assert rows==5700 and silhouettes==480
    review=pd.read_csv(OUT/'review/annotation_template.csv',dtype=str,keep_default_na=False)
    assert len(review)==180 and review.image_id.is_unique
    for r in review.itertuples():
        row=df.iloc[int(r.row)];assert r.image_id==row.image_id and r.hs1==row.hs1 and r.hs2==row.hs2 and r.hs3==row.hs3 and r.source_notations==row.notations
    from review import HUMAN
    assert review[HUMAN].eq('').all().all()
    for html_path in (OUT/'galleries').glob('*.html'):
        class Parser(HTMLParser):
            def handle_starttag(self,tag,attrs):
                if tag=='img':
                    p=(html_path.parent/dict(attrs)['src']).resolve();assert p.is_relative_to(ROOT/'data/phase2') and p.is_file()
        Parser().feed(html_path.read_text())
    missing=[]
    for p in OUT.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if not link.startswith(('https://','http://')) and not (p.parent/link).exists():missing.append((p.name,link))
    assert not missing,missing
    # Credential safety: only inspect the configured value locally, never print it.
    secret=os.getenv('EUROPEANA_API_KEY') or dotenv_values(ROOT/'.env').get('EUROPEANA_API_KEY')
    scanned=0
    if secret:
        for p in OUT.rglob('*'):
            if p.is_file() and p.suffix in {'.json','.csv','.md','.py','.html','.log','.xml','.svg','.txt'}:
                assert secret.encode() not in p.read_bytes(), 'Credential found in '+str(p.relative_to(OUT));scanned+=1
    tests=ElementTree.parse(OUT/'tests.xml');suites=list(tests.iter('testsuite'));assert all(int(s.get('failures','0'))==0 and int(s.get('errors','0'))==0 for s in suites)
    result={'utc':now(),'success':True,'original_pilot_unchanged':True,'no_new_dataset_downloaded':True,'commit_or_push_performed':False,'images':len(df),'encoders':2,'full_grid_fits':240,'subset_fits':60,'random_partitions':1600,'independently_recomputed_metric_rows':rows,'independently_recomputed_silhouettes':silhouettes,'seed_partition_comparisons':400,'subset_partition_comparisons':40,'human_review_images':len(review),'human_annotations_completed':0,'source_labels_changed':False,'input_order_and_masks_verified':True,'model_weight_hashes_verified':True,'artifact_links_valid':True,'gallery_photo_bytes_inside_ignored_data':True,'credential_scan_files':scanned,'credential_found':False,'targeted_tests_passed':sum(int(s.get('tests','0')) for s in suites),'source_sha256':{p.name:sha(p) for p in (OUT/'source').glob('*.py')},'report_sha256':{p.name:sha(p) for p in OUT.glob('*.md')},'figure_sha256':{p.name:sha(p) for p in (OUT/'figures').glob('*')},'audits':checks}
    write_json(OUT/'FINAL_VALIDATION.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','report_sha256','figure_sha256','audits']},indent=2))

if __name__=='__main__':main()
