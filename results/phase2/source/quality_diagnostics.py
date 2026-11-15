"""Measured source-quality diagnostics and separately recorded visual observations."""
import json
import re
import pandas as pd
from PIL import Image
from benchmark import OUT,ROOT,inputs,csv,write_json,now

def main():
    d,_,_=inputs();q=pd.read_csv(OUT/'review/image_quality_candidates.csv');m=pd.read_csv(OUT/'review/museum_label_coverage.csv')
    observations=[]
    for row,record,note in [(237,'/09102/_IMIT_JAD_4807','Generic missing-image icon visible instead of an instrument; source HS3 is 421.'),(1834,'/09102/_MINIM_UK_10345','Photograph shows piano maker branding/detail rather than a complete square piano; source HS3 is 314.'),(1955,'/09102/_NL_AsdRM_lido_4691','Image is a painting of a terrace scene rather than a museum-object photograph; HS labels missing.')]:
        r=d.iloc[row];assert r.id==record
        observations.append({'row':row,'image_id':r.image_id,'record_id':r.id,'observation':note,'status':'assistant visual observation; human confirmation pending','source_labels_changed':False,'benchmark_excluded':False})
    csv(OUT/'review/visual_observations.csv',observations)
    conflicts={}
    for depth in [1,2,3]:
        n=0
        for value in d.notations:
            prefixes=[re.split(r'[-+]',c)[0].replace('.','')[:depth] for c in json.loads(value) if c]
            n+=len(set(prefixes))>1
        conflicts['conflicting_prefix_hs'+str(depth)]=n
    write_json(OUT/'review/data_quality_summary.json',{'utc':now(),'hs1_labelled':int(d.hs1.ne('').sum()),'hs3_labelled':int(d.hs3.ne('').sum()),'hs1_majority_fraction_among_labeled':float(d.hs1[d.hs1.ne('')].value_counts(normalize=True).iloc[0]),'hs3_majority_fraction_among_labeled':float(d.hs3[d.hs3.ne('')].value_counts(normalize=True).iloc[0]),'museums_with_zero_hs3':int(m.hs3_labeled.eq(0).sum()),'museums_with_all_hs3':int(m.hs3_missing_pct.eq(0).sum()),'parts_title_flagged':int(q.title_parts_flag.sum()),'hs3_missing_fraction_parts_flagged':float(d.loc[q.title_parts_flag,'hs3'].eq('').mean()),'hs3_missing_fraction_not_flagged':float(d.loc[~q.title_parts_flag,'hs3'].eq('').mean()),**conflicts,'visual_observations':3,'review_annotations_filled':0,'no_label_corrections':True})
    p=ROOT/'data/phase2/gallery'
    with Image.open(p/'human_review_180.jpg') as im:
        im.crop((0,0,1200,1350)).save(p/'review_pairs_excerpt.jpg')
        im.crop((0,5400,1200,6750)).save(p/'review_screening_excerpt.jpg')

if __name__=='__main__':main()
