"""Chance partitions and crossed seed/subset sensitivity on fixed inputs."""
import argparse
import itertools
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits
from benchmark import (OUT, SEEDS, LABELS, inputs, array, csv, write_json, now, sha,
                       transform, estimator, metrics, size_stats, run_id)

def evaluate(encoder, X, df):
    out=OUT/'evaluation'/encoder; out.mkdir(parents=True,exist_ok=True)
    runs=pd.read_csv(OUT/encoder/'runs.csv')
    seed_rows=[]
    for (rep,method,k),group in runs.groupby(['representation','method','requested_k']):
        if method!='kmeans': continue
        for a,b in itertools.combinations(group.itertuples(),2):
            pa=np.load(OUT/encoder/'predictions'/f'{a.run_id}.npy');pb=np.load(OUT/encoder/'predictions'/f'{b.run_id}.npy')
            seed_rows.append({'encoder':encoder,'representation':rep,'method':method,'k':k,'seed_a':a.seed,'seed_b':b.seed,'partition_ARI':adjusted_rand_score(pa,pb),'partition_NMI':normalized_mutual_info_score(pa,pb)})
    csv(out/'seed_partition_stability.csv',seed_rows)
    chance=[]
    for r in runs[runs.seed.eq(0)].itertuples():
        pred=np.load(OUT/encoder/'predictions'/f'{r.run_id}.npy')
        shuffled=np.stack([np.random.default_rng(seed).permutation(pred) for seed in range(1000,1020)])
        path=out/'chance_predictions'/f'{r.run_id}.npy';array(path,shuffled)
        assert all(np.array_equal(np.bincount(x),np.bincount(pred)) for x in shuffled)
        for j,p in enumerate(shuffled):
            for label in LABELS:
                mask=df[label].ne('').to_numpy()
                chance.append({'encoder':encoder,'representation':r.representation,'method':r.method,'k':r.requested_k,'source_run_id':r.run_id,'baseline_seed':1000+j,'prediction_row':j,'predictions_sha256':sha(path),'label':label,'n_evaluated':int(mask.sum()),**metrics(df.loc[mask,label],p[mask])})
        csv(out/'chance_scores.csv',chance)
    print(encoder,'chance baseline complete',len(chance),'metric rows',flush=True)
    scores=[];overlap=[];subset_predictions={}
    for subset_seed in range(2026,2031):
        idx=np.sort(np.random.default_rng(subset_seed).choice(len(df),1600,replace=False))
        sub=df.iloc[idx].reset_index(drop=True)
        folder=out/'subsets'/str(subset_seed)
        array(folder/'rows.npy',idx)
        csv(folder/'image_order_and_masks.csv',[{'original_row':int(i),'image_id':df.iloc[i].image_id,**{label:df.iloc[i][label] for label in LABELS}} for i in idx])
        Z,params=transform(X[idx],'centered_l2',folder/'centered_l2.npy')
        for method in ['kmeans','average_cosine']:
            full=np.load(OUT/encoder/'predictions'/(run_id(encoder,'centered_l2',method,10,0)+'.npy'))
            for seed in SEEDS if method=='kmeans' else [0]:
                started=time.perf_counter();pred=estimator(method,10,seed).fit_predict(Z);seconds=time.perf_counter()-started
                path=folder/f'{method}_s{seed}.npy';array(path,pred)
                if seed==0: subset_predictions[(subset_seed,method)]=(idx,pred)
                for label in LABELS:
                    mask=sub[label].ne('').to_numpy()
                    scores.append({'encoder':encoder,'subset_seed':subset_seed,'seed':seed,'method':method,'representation':'centered_l2','k':10,'n_images':1600,'label':label,'n_evaluated':int(mask.sum()),'cluster_seconds':seconds,'predictions_sha256':sha(path),'full_vs_subset_partition_ARI':adjusted_rand_score(full[idx],pred),**size_stats(pred),**metrics(sub.loc[mask,label],pred[mask])})
        csv(out/'subset_scores.csv',scores)
        print(encoder,'subset',subset_seed,'complete',flush=True)
    for method in ['kmeans','average_cosine']:
        for a,b in itertools.combinations(range(2026,2031),2):
            ia,pa=subset_predictions[(a,method)];ib,pb=subset_predictions[(b,method)]
            common,ca,cb=np.intersect1d(ia,ib,return_indices=True)
            overlap.append({'encoder':encoder,'method':method,'subset_a':a,'subset_b':b,'seed':0,'common_images':len(common),'partition_ARI':adjusted_rand_score(pa[ca],pb[cb]),'partition_NMI':normalized_mutual_info_score(pa[ca],pb[cb])})
    csv(out/'subset_overlap_stability.csv',overlap)
    s=pd.DataFrame(scores);summary=[]
    for (method,label),g in s.groupby(['method','label']):
        by=g.groupby('subset_seed').AMI.agg(['mean','std'])
        summary.append({'encoder':encoder,'method':method,'label':label,'mean_AMI':g.AMI.mean(),'mean_within_subset_seed_sd':by['std'].mean(),'sd_of_subset_seed_means':by['mean'].std(),'subsets':5,'images_per_subset':1600,'note':'Conditional overlapping subset sensitivity, not a population confidence interval; seed SD is initialization variability.'})
    csv(out/'uncertainty_summary.csv',summary)
    write_json(out/'complete.json',{'utc':now(),'success':True,'chance_metric_rows':len(chance),'subset_metric_rows':len(scores),'seed_pair_comparisons':len(seed_rows),'subset_pair_comparisons':len(overlap)})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('encoder',choices=['clip','siglip']);a=ap.parse_args()
    df,X,m=inputs()
    if a.encoder=='siglip':
        m=json.loads((OUT/'siglip/embedding_manifest.json').read_text());assert sha(m['array_path'])==m['array_sha256'];X=np.load(m['array_path'])
    with threadpool_limits(limits=4): evaluate(a.encoder,X,df)
