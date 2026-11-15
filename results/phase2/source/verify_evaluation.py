"""Recompute random-baseline, subset and stability scores from persisted arrays."""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
from benchmark import OUT,inputs,sha,write_json,now,run_id

FS={'AMI':adjusted_mutual_info_score,'ARI':adjusted_rand_score,'NMI':normalized_mutual_info_score}

def main(encoder):
    df,X,m=inputs()
    if encoder=='siglip':
        m=json.loads((OUT/'siglip/embedding_manifest.json').read_text());assert sha(m['array_path'])==m['array_sha256'];X=np.load(m['array_path'])
    out=OUT/'evaluation'/encoder;chance=pd.read_csv(out/'chance_scores.csv');subsets=pd.read_csv(out/'subset_scores.csv');checked=0
    for source,g in chance.groupby('source_run_id'):
        path=out/'chance_predictions'/f'{source}.npy';preds=np.load(path);full=np.load(OUT/encoder/'predictions'/f'{source}.npy')
        assert sha(path)==g.predictions_sha256.iloc[0]
        for j,p in enumerate(preds):
            assert np.array_equal(np.bincount(p),np.bincount(full))
            assert np.array_equal(p,np.random.default_rng(1000+j).permutation(full))
        for r in g.itertuples():
            mask=df[r.label].ne('').to_numpy();p=preds[r.prediction_row]
            assert mask.sum()==r.n_evaluated
            for name,f in FS.items():assert np.isclose(f(df.loc[mask,r.label],p[mask]),getattr(r,name),atol=1e-12)
            checked+=1
    for subset_seed,g in subsets.groupby('subset_seed'):
        folder=out/'subsets'/str(subset_seed);idx=np.load(folder/'rows.npy')
        assert np.array_equal(idx,np.sort(np.random.default_rng(subset_seed).choice(2000,1600,replace=False)))
        sub=df.iloc[idx].reset_index(drop=True);order=pd.read_csv(folder/'image_order_and_masks.csv',dtype=str,keep_default_na=False)
        assert sub.image_id.tolist()==order.image_id.tolist()
        z=np.load(folder/'centered_l2.npy');expected=normalize(X[idx]-X[idx].mean(0));assert np.allclose(z,expected,atol=1e-7)
        for (method,seed),gg in g.groupby(['method','seed']):
            path=folder/f'{method}_s{seed}.npy';p=np.load(path);assert sha(path)==gg.predictions_sha256.iloc[0]
            full=np.load(OUT/encoder/'predictions'/(run_id(encoder,'centered_l2',method,10,0)+'.npy'))
            for r in gg.itertuples():
                mask=sub[r.label].ne('').to_numpy();assert mask.sum()==r.n_evaluated
                for name,f in FS.items():assert np.isclose(f(sub.loc[mask,r.label],p[mask]),getattr(r,name),atol=1e-12)
                assert np.isclose(adjusted_rand_score(full[idx],p),r.full_vs_subset_partition_ARI,atol=1e-12)
                assert np.isclose(np.bincount(p).max()/1600*100,r.largest_cluster_pct)
                checked+=1
    seed_pairs=pd.read_csv(out/'seed_partition_stability.csv')
    for r in seed_pairs.itertuples():
        a=np.load(OUT/encoder/'predictions'/(run_id(encoder,r.representation,r.method,r.k,r.seed_a)+'.npy'));b=np.load(OUT/encoder/'predictions'/(run_id(encoder,r.representation,r.method,r.k,r.seed_b)+'.npy'))
        assert np.isclose(adjusted_rand_score(a,b),r.partition_ARI,atol=1e-12)
        assert np.isclose(normalized_mutual_info_score(a,b),r.partition_NMI,atol=1e-12)
    pairs=pd.read_csv(out/'subset_overlap_stability.csv')
    for r in pairs.itertuples():
        fa=out/'subsets'/str(r.subset_a);fb=out/'subsets'/str(r.subset_b);ia=np.load(fa/'rows.npy');ib=np.load(fb/'rows.npy');common,a,b=np.intersect1d(ia,ib,return_indices=True)
        pa=np.load(fa/f'{r.method}_s0.npy');pb=np.load(fb/f'{r.method}_s0.npy')
        assert len(common)==r.common_images
        assert np.isclose(adjusted_rand_score(pa[a],pb[b]),r.partition_ARI,atol=1e-12)
        assert np.isclose(normalized_mutual_info_score(pa[a],pb[b]),r.partition_NMI,atol=1e-12)
    write_json(out/'validation.json',{'utc':now(),'success':True,'chance_and_subset_metric_rows_recomputed':checked,'seed_pair_rows_recomputed':len(seed_pairs),'subset_pair_rows_recomputed':len(pairs),'subsets_refit_centering_verified':True})
    print(encoder,'auxiliary audit passed',checked,'metric rows',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('encoder',choices=['clip','siglip']);main(ap.parse_args().encoder)
