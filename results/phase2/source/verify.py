"""Independent metric audit: reads arrays and masks, never imports fit functions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from threadpoolctl import threadpool_limits
from benchmark import OUT, ROOT, DATA, sha, write_json, csv, now

FUNCTIONS = {'AMI': adjusted_mutual_info_score, 'ARI': adjusted_rand_score, 'NMI': normalized_mutual_info_score}

def verify(encoder):
    sample = pd.read_csv(DATA/'sample_pilot.csv',dtype=str,keep_default_na=False)
    order = pd.read_csv(OUT/'inputs/image_order.csv',dtype=str,keep_default_na=False)
    masks = pd.read_csv(OUT/'inputs/evaluation_masks.csv',dtype=str,keep_default_na=False)
    assert order.image_id.tolist() == sample.image_id.tolist() == masks.image_id.tolist()
    assert order.sha256.tolist() == sample.sha256.tolist()
    for r in sample.itertuples(): assert sha(DATA/r.file) == r.sha256
    out=OUT/encoder; scores=pd.read_csv(out/'scores.csv'); runs=pd.read_csv(out/'runs.csv'); sizes=pd.read_csv(out/'cluster_sizes.csv')
    checked=0
    with threadpool_limits(limits=4):
        for rep, group in runs.groupby('representation'):
            path=out/'representations'/f'{rep}.npy'; meta=json.loads(path.with_suffix('.json').read_text())
            assert sha(path)==meta['array_sha256']
            Z=np.load(path,allow_pickle=False)
            for r in group.itertuples():
                path=out/'predictions'/f'{r.run_id}.npy'; assert sha(path)==r.predictions_sha256
                pred=np.load(path,allow_pickle=False); counts=np.bincount(pred)
                assert len(pred)==2000 and len(counts)==r.requested_k and (counts>0).all()
                s=sizes[sizes.run_id.eq(r.run_id)].sort_values('cluster')
                assert np.array_equal(s.images.to_numpy(),counts)
                assert np.isclose(100*counts.max()/2000,r.largest_cluster_pct)
                assert np.isclose(silhouette_score(Z,pred,metric='cosine'),r.silhouette_cosine,atol=2e-6)
                assert np.isclose(silhouette_score(Z,pred,metric='euclidean'),r.silhouette_euclidean,atol=2e-6)
                for row in scores[scores.run_id.eq(r.run_id)].itertuples():
                    mask=sample[row.label].ne('').to_numpy()
                    assert np.array_equal(mask,masks[row.label+'_valid'].eq('True').to_numpy())
                    assert mask.sum()==row.n_evaluated
                    for name,f in FUNCTIONS.items():
                        assert np.isclose(f(sample.loc[mask,row.label],pred[mask]),getattr(row,name),atol=1e-12)
                    checked+=1
    parity=[]
    if encoder=='clip':
        old=pd.read_csv(ROOT/'results/pilot/scores.csv')
        for r in old.itertuples():
            new=scores[(scores.representation==r.representation)&(scores.method==r.method)&(scores.requested_k==r.requested_k)&(scores.seed==r.seed)&(scores.label==r.label)].iloc[0]
            parity.append({'original_run_id':r.run_id,'label':r.label,'AMI_delta':new.AMI-r.AMI,'ARI_delta':new.ARI-r.ARI,'NMI_delta':new.NMI-r.NMI})
        csv(out/'original_pilot_parity.csv',parity)
        assert max(abs(r['AMI_delta']) for r in parity)<1e-10
    old=json.loads((OUT/'inputs/original_snapshot.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in old['files'].items())
    write_json(out/'validation.json',{'utc':now(),'success':True,'images_rehashed':2000,'runs':len(runs),'metric_rows_recomputed':checked,'silhouettes_recomputed':2*len(runs),'original_pilot_unchanged':True,'original_parity_rows':len(parity)})
    print(encoder,'independent audit passed:',checked,'metric rows,',2*len(runs),'silhouettes',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('encoder',choices=['clip','siglip']);verify(ap.parse_args().encoder)
