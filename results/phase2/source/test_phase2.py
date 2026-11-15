"""Safeguards for the scientific invariants, not implementation-mirroring tests."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, adjusted_rand_score
from sklearn.preprocessing import normalize
sys.path.insert(0,str(Path(__file__).resolve().parent))
from review import protect_annotations

def test_positive_row_scaling_does_not_change_cosine_clustering():
    x=np.random.default_rng(0).normal(size=(90,12));scaled=x*np.exp(np.random.default_rng(1).normal(size=(90,1)))
    assert np.allclose(pairwise_distances(x,metric='cosine'),pairwise_distances(scaled,metric='cosine'),atol=1e-12)
    a=AgglomerativeClustering(n_clusters=5,metric='cosine',linkage='average').fit_predict(x)
    b=AgglomerativeClustering(n_clusters=5,metric='cosine',linkage='average').fit_predict(scaled)
    assert adjusted_rand_score(a,b)==1

def test_full_pca_rotation_agrees_with_centered_cosine_geometry():
    x=np.random.default_rng(0).normal(size=(90,12))+3
    a=normalize(x-x.mean(0));b=normalize(PCA(12,svd_solver='full').fit_transform(x))
    assert np.allclose(pairwise_distances(a,metric='cosine'),pairwise_distances(b,metric='cosine'),atol=1e-12)

def test_review_rerun_cannot_overwrite_human_annotations(tmp_path):
    p=tmp_path/'review.csv';pd.DataFrame([{'image_id':'a','reviewer':'','is_part':''}]).to_csv(p,index=False);protect_annotations(p)
    pd.DataFrame([{'image_id':'a','reviewer':'Expert reviewer','is_part':'yes'}]).to_csv(p,index=False)
    with pytest.raises(RuntimeError):protect_annotations(p)
