"""Reproducible targeted review sample; flags do not alter benchmark labels."""
import html
import json
import os
import re
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image,ImageOps,ImageStat,ImageDraw
from benchmark import OUT,ROOT,DATA,inputs,csv,write_json,now,run_id

REVIEW_SEED=20261102
PARTS=r'(?i)\b(?:bow|bows|mouthpiece|beater|reed|archet|embouchure|baguette|part|chanter)\b'
REPR=r'(?i)\b(?:drawing|engraving|photograph|painting|gravure|dessin|carte|portrait|plan|reproduction|miniature)\b'
PLACEHOLDER=r'(?i)(?:no image|not available|missing image|sans image|indisponible|placeholder)'
HUMAN=['reviewer','reviewed_at','is_complete_instrument','is_part','is_representation','is_placeholder','source_hs_plausible','same_object_as_image_id','recommended_exclude','review_notes']

def protect_annotations(path):
    if path.exists():
        old=pd.read_csv(path,dtype=str,keep_default_na=False)
        if any(col in old and old[col].str.strip().ne('').any() for col in HUMAN):
            raise RuntimeError('Existing human annotations found; preserve them and create a new review version instead of overwriting.')

def image_stats(df):
    out=[];hashes=[]
    for i,r in enumerate(df.itertuples()):
        with Image.open(DATA/r.file) as im:
            im=ImageOps.exif_transpose(im).convert('RGB');g=im.convert('L').resize((9,8));a=np.asarray(g);bits=(a[:,1:]>a[:,:-1]).ravel();dh=int(''.join('1' if x else '0' for x in bits),2)
            small=im.resize((64,64));gray=small.convert('L');hist=np.asarray(gray.histogram(),float);p=hist/hist.sum();entropy=float(-(p[p>0]*np.log2(p[p>0])).sum());sat=float(np.asarray(small.convert('HSV'))[:,:,1].mean())
        hashes.append(dh);out.append({'row':i,'image_id':r.image_id,'dhash_hex':f'{dh:016x}','grayscale_entropy':entropy,'mean_saturation':sat,'title_parts_flag':bool(re.search(PARTS,r.title)),'title_representation_flag':bool(re.search(REPR,r.title)),'title_placeholder_flag':bool(re.search(PLACEHOLDER,r.title)),'multiple_hs_uris':len(json.loads(r.hs_uris))>1})
    return pd.DataFrame(out),hashes

def thumbnail(row):
    path=ROOT/'data/phase2/gallery/thumbnails'/(row.image_id+'.jpg');path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        with Image.open(DATA/row.file) as im:
            im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((240,200));im.save(path,quality=88)
    return path

def gallery(name,title,groups,df,caption):
    out=OUT/'galleries'/f'{name}.html';out.parent.mkdir(parents=True,exist_ok=True)
    e=html.escape;sections=[];manifest=[]
    for group,ids in groups:
        cards=[]
        for i in ids:
            r=df.iloc[int(i)];src=os.path.relpath(thumbnail(r),out.parent)
            cards.append(f'<figure><img loading="lazy" src="{e(src)}" alt="Source instrument image"><figcaption>Row {i}: {e(r.title)}<br>{e(r.museum)}<br>Source HS: {e(r.notations)}<br>{e(r.id)}<br><a href="{e(r.image_url)}">Source image</a><br>{e(r.rights)}</figcaption></figure>')
            manifest.append({'gallery':name,'group':group,'row':int(i),'image_id':r.image_id})
        sections.append(f'<section><h2>{e(group)}</h2><div>{"".join(cards)}</div></section>')
    out.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>'+e(title)+'</title><style>body{font:16px system-ui;background:#f6f5f1;margin:30px auto;max-width:1500px;padding:20px}section>div{display:flex;flex-wrap:wrap;gap:10px}figure{width:245px;background:white;padding:12px;margin:0}img{width:240px;height:200px;object-fit:contain}figcaption{font-size:12px;overflow-wrap:anywhere}section{margin:30px 0}</style><h1>'+e(title)+'</h1><p>'+e(caption)+'</p>'+''.join(sections)+'</html>')
    csv(OUT/'galleries'/f'{name}_image_order.csv',manifest)
    # Photography stays inside ignored data/, including presentation contact sheets.
    flat=[(g,int(i)) for g,ids in groups for i in ids];cols=5;cw,ch=240,225
    sheet=Image.new('RGB',(cols*cw,((len(flat)+cols-1)//cols)*ch),'white');draw=ImageDraw.Draw(sheet)
    for j,(g,i) in enumerate(flat):
        with Image.open(thumbnail(df.iloc[i])) as im:
            im=ImageOps.contain(im.convert('RGB'),(225,185));x=(j%cols)*cw;y=(j//cols)*ch;sheet.paste(im,(x,y+25));draw.text((x+3,y+3),g[:32]+' / '+str(i),fill='black')
    contact=ROOT/'data/phase2/gallery'/f'{name}.jpg';sheet.save(contact,quality=90)
    return out

def main():
    protect_annotations(OUT/'review/annotation_template.csv')
    df,X,_=inputs();q,hashes=image_stats(df);csv(OUT/'review/image_quality_candidates.csv',q)
    cosine=np.clip(X@X.T,-1,1);pairs=[];titles=df.title.str.casefold().str.replace(r'\W+',' ',regex=True).str.strip().tolist()
    for i in range(len(df)):
        for j in range(i):
            h=(hashes[i]^hashes[j]).bit_count();same=(df.iloc[i].museum==df.iloc[j].museum and titles[i]==titles[j])
            sim=float(cosine[i,j])
            if (same and sim>=.88) or (h<=3 and sim>=.90):pairs.append({'row_a':j,'row_b':i,'image_id_a':df.iloc[j].image_id,'image_id_b':df.iloc[i].image_id,'dhash_hamming':h,'clip_cosine':sim,'same_museum_and_title':same,'interpretation':'possible same object/view or visually similar type; unconfirmed'})
    pairs.sort(key=lambda p:(-p['clip_cosine'],p['dhash_hamming'],p['row_a'],p['row_b']))
    csv(OUT/'review/possible_duplicate_pairs.csv',pairs)
    rng=np.random.default_rng(REVIEW_SEED);selected=[];seen=set();partner={};flags={}
    def add(ids,stratum,n):
        count=0
        for i in ids:
            i=int(i)
            if i not in seen:
                selected.append((i,stratum));seen.add(i);count+=1
                if count>=n:break
        if count<n:
            for i in rng.permutation(len(df)):
                if i not in seen:selected.append((int(i),stratum+' (random backfill)'));seen.add(int(i));count+=1
                if count>=n:break
    pair_ids=[]
    for p in pairs:
        a,b=p['row_a'],p['row_b']
        if a not in pair_ids and b not in pair_ids:
            pair_ids.extend([a,b]);partner[a]=df.iloc[b].image_id;partner[b]=df.iloc[a].image_id
            if len(pair_ids)==30:break
    add(pair_ids,'possible duplicate/view pair',30)
    museum_order=rng.permutation(df.museum.unique());museum_rows=[]
    for museum in museum_order:
        museum_rows.extend(rng.permutation(np.flatnonzero(df.museum.eq(museum))))
    # Initial pass is one candidate per museum, avoiding large-museum dominance.
    firsts=[int(rng.choice(np.flatnonzero(df.museum.eq(m)))) for m in museum_order]
    add(firsts+museum_rows,'random across museums',30)
    pools=[list(rng.permutation(np.flatnonzero(df.hs1.eq(c)))) for c in sorted(set(df.hs1)-{''})]
    class_rows=[]
    while any(pools):
        for pool in pools:
            if pool:class_rows.append(pool.pop())
    add(class_rows,'random across HS1 classes (review only)',30)
    pred=np.load(OUT/'clip/predictions'/(run_id('clip','l2','average_cosine',5,0)+'.npy'));counts=np.bincount(pred)
    cluster_order=sorted(range(len(counts)),key=lambda c:(counts[c],c));extremes=[]
    for c in cluster_order:
        extremes.extend(rng.permutation(np.flatnonzero(pred==c))[:8])
    add(extremes,'large and small failure-case clusters',30)
    suspicious=list(rng.permutation(np.flatnonzero(q.title_parts_flag|q.title_representation_flag|q.title_placeholder_flag)))
    # Low entropy / low saturation are weak screening signals, never semantic labels.
    weak=list(q.sort_values(['grayscale_entropy','row']).row.head(12))+list(q.sort_values(['mean_saturation','row']).row.head(12))
    interleaved=[]
    for i in range(max(len(suspicious),len(weak))):
        if i<len(suspicious):interleaved.append(suspicious[i])
        if i<len(weak):interleaved.append(weak[i])
    add(interleaved,'potential parts / representations / placeholder screen',30)
    multiple=list(rng.permutation(np.flatnonzero(q.multiple_hs_uris)))
    missing=list(rng.permutation(np.flatnonzero(df.hs3.eq(''))))
    special=[]
    for a,b in zip(multiple,missing):special.extend([a,b])
    add(special+multiple+missing,'missing / multiple or conflicting HS',30)
    assert len(selected)==180 and len(seen)==180
    annotations=[]
    human=HUMAN
    for i,stratum in selected:
        r=df.iloc[i];s=q.iloc[i];f=[]
        for col in ['title_parts_flag','title_representation_flag','title_placeholder_flag','multiple_hs_uris']:
            if s[col]:f.append(col)
        if s.grayscale_entropy<=q.grayscale_entropy.quantile(.1):f.append('low_entropy_screen')
        if s.mean_saturation<=q.mean_saturation.quantile(.1):f.append('low_saturation_screen')
        annotations.append({'row':i,'image_id':r.image_id,'record_id':r.id,'selection_stratum':stratum,'candidate_flags':'; '.join(f),'possible_pair_image_id':partner.get(i,''),'source_title':r.title,'museum':r.museum,'hs1':r.hs1,'hs2':r.hs2,'hs3':r.hs3,'source_notations':r.notations,'source_hs_uris':r.hs_uris,'image_file':str(DATA/r.file),'source_url':r.image_url,**{col:'' for col in human}})
    csv(OUT/'review/annotation_template.csv',annotations)
    groups=[(s,[i for i,g in selected if g==s]) for s in dict.fromkeys(s for _,s in selected)]
    gallery('human_review_180','MIMO: 180-image human review',groups,df,'Unreviewed candidates. Six targeted strata; seed 20261102. Screening flags are not annotations. Source labels remain unchanged. Fill review/annotation_template.csv using the codebook. This targeted sample cannot estimate population error prevalence.')
    (OUT/'review/ANNOTATION_CODEBOOK.md').write_text('# Annotation codebook\n\nAll human fields start blank. Use yes/no/uncertain for complete instrument, part, representation, placeholder, plausible source HS, and recommended exclusion. A flag is only a sampling rule; titles can match innocently (e.g. an entire flute mentioning embouchure). Low entropy or saturation does not prove a placeholder or drawing. Multiple HS concepts can be compatible. Assess images and source evidence; do not infer corrections from titles alone.\n\nIf possible same-object views are confirmed, enter the partner image ID in same_object_as_image_id; similar instrument types are not necessarily the same specimen. Record reviewer, date and notes. Do not edit source columns. No annotations are automatically applied to labels, exclusions or scores. Maintain a reviewed copy and an explicit new dataset version for any later curation.\n\nSampling: 30 possible-pair images, 30 museum-random, 30 HS1-stratified, 30 large/small-cluster, 30 lexical/low-entropy/low-saturation candidates, 30 missing/multiple-HS. Disjoint selection in that order, seeded 20261102. Prior selections can change stratum quotas; actual coverage is saved. This enriched review set is not a representative error-rate sample.\n')
    good=np.load(OUT/'clip/predictions'/(run_id('clip','l2','kmeans',5,0)+'.npy'));Z=np.load(OUT/'clip/representations/l2.npy');g=[]
    for c in range(5):
        ids=np.flatnonzero(good==c);mean=Z[ids].mean(0);nearest=ids[np.argsort(np.linalg.norm(Z[ids]-mean,axis=1),kind='stable')[:8]];g.append((f'Cluster {c} ({len(ids)} images)',nearest))
    gallery('meaningful_grouping','Illustrative grouping: CLIP K-Means k=5',g,df,'Fixed raw-L2 CLIP, K-Means k=5, seed 0; eight images nearest each cluster mean, chosen without HS AMI. Illustrates shape grouping, not universally correct taxonomy.')
    failure=[];rr=np.random.default_rng(REVIEW_SEED)
    for c in np.argsort(-counts):failure.append((f'Cluster {c} ({counts[c]} images)',rr.choice(np.flatnonzero(pred==c),min(20,counts[c]),replace=False)))
    gallery('failure_case','Failure case: raw-L2 cosine average linkage k=5',failure,df,'Fixed raw-L2 CLIP average linkage k=5. Largest cluster: 1993/2000. Twenty seeded random images from the giant cluster plus every image in each small cluster; source HS is shown only for interpretation.')
    museum=df.groupby('museum').agg(images=('id','size'),hs3_labeled=('hs3',lambda x:x.ne('').sum()));museum['hs3_missing_pct']=100*(1-museum.hs3_labeled/museum.images);csv(OUT/'review/museum_label_coverage.csv',museum.reset_index())
    coverage=[]
    runs=pd.read_csv(OUT/'clip/runs.csv')
    for r in runs[runs.seed.eq(0)].itertuples():
        p=np.load(OUT/'clip/predictions'/f'{r.run_id}.npy')
        for c in set(p):
            mask=p==c;n=int(mask.sum());known=int(df.loc[mask,'hs3'].ne('').sum());coverage.append({'run_id':r.run_id,'cluster':int(c),'images':n,'hs3_labeled':known,'hs3_missing_pct':100*(1-known/n)})
    csv(OUT/'review/cluster_label_coverage.csv',coverage)
    write_json(OUT/'review/summary.json',{'utc':now(),'review_seed':REVIEW_SEED,'review_images':len(selected),'review_museums':df.iloc[list(seen)].museum.nunique(),'strata':dict(Counter(g for _,g in selected)),'candidate_pair_count':len(pairs),'selected_pair_images':len(pair_ids),'title_parts_candidates':int(q.title_parts_flag.sum()),'title_representation_candidates':int(q.title_representation_flag.sum()),'explicit_placeholder_title_candidates':int(q.title_placeholder_flag.sum()),'multiple_hs_records':int(q.multiple_hs_uris.sum()),'human_annotations_completed':0,'labels_or_exclusions_changed':False})
    print('180-image review and two illustrative galleries saved',flush=True)

if __name__=='__main__':main()
