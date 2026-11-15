"""Presentation figures and paired summaries; no fitting or method selection."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark import OUT,REPS,KS,csv,run_id
from sklearn.metrics import adjusted_rand_score

NAMES={'l2':'Original L2','centered_l2':'Centered full + L2','pca50':'PCA50, no post-L2','pca50_l2':'PCA50 + L2','pca100_l2':'PCA100 + L2'}
COLORS={'clip':'#246a9a','siglip':'#d06b2e'}

def save(fig,name):
    path=OUT/'figures';path.mkdir(parents=True,exist_ok=True);fig.savefig(path/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(path/(name+'.svg'),bbox_inches='tight');plt.close(fig)

def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    summaries={e:pd.read_csv(OUT/e/'summary.csv') for e in ['clip','siglip']}
    hs={e:d[d.label.eq('hs3')] for e,d in summaries.items()}
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,method in zip(axes,['kmeans','average_cosine']):
        for enc,d in hs.items():
            g=d[d.representation.eq('centered_l2')&d.method.eq(method)].sort_values('requested_k');sd=g.AMI_seed_sd.fillna(0)
            ax.errorbar(g.requested_k,g.AMI_mean,yerr=sd,marker='o',capsize=3,color=COLORS[enc],label='CLIP B/32' if enc=='clip' else 'SigLIP B/16')
        ax.set(title='K-Means' if method=='kmeans' else 'Average linkage · cosine',xlabel='Number of clusters',xticks=KS);ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('HS3 adjusted mutual information')
    fig.suptitle('Fixed comparison: centered full-dimensional embeddings + L2')
    fig.text(.5,-.03,'Same 2,000 images; 1,549 evaluated labels. K-Means: mean ± seed SD (5 seeds); average linkage: one deterministic fit.\nDescriptive fixed-grid comparison, not held-out generalization.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'clip_siglip_comparison')
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharey=True)
    for ax,method in zip(axes,['kmeans','average_cosine']):
        for j,rep in enumerate(REPS):
            g=hs['clip'][hs['clip'].representation.eq(rep)&hs['clip'].method.eq(method)].sort_values('requested_k')
            ax.errorbar(g.requested_k,g.AMI_mean,yerr=g.AMI_seed_sd.fillna(0),marker='o',ls='--' if rep=='pca50' else '-',capsize=2,label=NAMES[rep],alpha=.9)
        ax.set(title='K-Means' if method=='kmeans' else 'Average linkage · cosine',xlabel='Number of clusters',xticks=KS);ax.grid(alpha=.2)
    axes[0].set_ylabel('HS3 adjusted mutual information');axes[1].legend(loc='lower right',fontsize=8)
    fig.suptitle('CLIP preprocessing ablation: centering versus dimensionality')
    fig.text(.5,-.03,'Cosine-average-link PCA50 curves with/without post-L2 coincide exactly. Error bars show initialization SD, not sampling uncertainty.',ha='center',fontsize=9)
    fig.tight_layout();save(fig,'preprocessing_ablation')
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,enc in zip(axes,['clip','siglip']):
        for rep in REPS:
            g=hs[enc][hs[enc].representation.eq(rep)&hs[enc].method.eq('average_cosine')].sort_values('requested_k')
            ax.plot(g.requested_k,g.largest_cluster_pct,marker='o',ls='--' if rep=='pca50' else '-',label=NAMES[rep])
        ax.plot(KS,[100/k for k in KS],':',color='gray',label='Equal-size reference 100/k')
        ax.set(title='CLIP B/32' if enc=='clip' else 'SigLIP B/16',xlabel='Number of clusters',xticks=KS,ylim=(0,103));ax.grid(alpha=.2)
    axes[0].set_ylabel('Images in largest cluster (%)');axes[1].legend(fontsize=8)
    fig.suptitle('Cluster-size imbalance · average linkage with cosine distance');fig.tight_layout();save(fig,'cluster_size_imbalance')
    paired=hs['clip'].merge(hs['siglip'],on=['representation','method','requested_k','label'],suffixes=('_clip','_siglip'))
    for metric in ['AMI','ARI','NMI']:paired[metric+'_delta_siglip_minus_clip']=paired[metric+'_mean_siglip']-paired[metric+'_mean_clip']
    csv(OUT/'paired_encoder_comparison.csv',paired)
    grid=[]
    for method,g in paired.groupby('method'):
        for nonredundant in [False,True]:
            sub=g[~g.representation.eq('pca50')] if nonredundant and method=='average_cosine' else g
            grid.append({'method':method,'remove_cosine_scale_duplicate':nonredundant,'cells':len(sub),'CLIP_mean_AMI':sub.AMI_mean_clip.mean(),'SigLIP_mean_AMI':sub.AMI_mean_siglip.mean(),'mean_paired_AMI_delta':sub.AMI_delta_siglip_minus_clip.mean(),'positive_cells':int(sub.AMI_delta_siglip_minus_clip.gt(0).sum()),'interpretation':'Equal-weight descriptive grid average; cells share images and are not independent replicates.'})
    csv(OUT/'paired_grid_summary.csv',grid)
    equivalence=[]
    for enc in ['clip','siglip']:
        for k in KS:
            a=np.load(OUT/enc/'predictions'/(run_id(enc,'pca50','average_cosine',k,0)+'.npy'));b=np.load(OUT/enc/'predictions'/(run_id(enc,'pca50_l2','average_cosine',k,0)+'.npy'))
            equivalence.append({'encoder':enc,'k':k,'partition_ARI':adjusted_rand_score(a,b),'arrays_identical':np.array_equal(a,b)})
    csv(OUT/'cosine_normalization_equivalence.csv',equivalence)
    # Chance baseline at the predeclared anchor; same-size permutations.
    fig,ax=plt.subplots(figsize=(9,4));labels=[];observed=[];null=[];std=[]
    for enc in ['clip','siglip']:
        chance=pd.read_csv(OUT/'evaluation'/enc/'chance_scores.csv')
        for method in ['kmeans','average_cosine']:
            g=chance[chance.label.eq('hs3')&chance.representation.eq('centered_l2')&chance.method.eq(method)&chance.k.eq(10)]
            score=pd.read_csv(OUT/enc/'scores.csv');v=score[score.label.eq('hs3')&score.representation.eq('centered_l2')&score.method.eq(method)&score.requested_k.eq(10)&score.seed.eq(0)].AMI.iloc[0]
            labels.append(enc.upper()+'\n'+('K-Means' if method=='kmeans' else 'Average cosine'));observed.append(v);null.append(g.AMI.mean());std.append(g.AMI.std())
    x=np.arange(4);ax.bar(x-.18,observed,.36,label='Observed seed-0 partition');ax.bar(x+.18,null,.36,yerr=std,capsize=3,label='20 size-preserving random partitions')
    ax.set(xticks=x,xticklabels=labels,ylabel='HS3 AMI',title='Chance control: fixed centered full-dimensional + L2, k=10');ax.axhline(0,color='gray',lw=.7);ax.legend(fontsize=9);fig.tight_layout();save(fig,'chance_baseline')
    print('Charts and paired encoder summaries saved',flush=True)

if __name__=='__main__':main()
