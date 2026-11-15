"""Week 6 / P3: declare P3_EXPERIMENT_PROTOCOL.json BEFORE any P3 fit. Refuses to overwrite."""
import json, sys, platform, importlib.metadata, hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/week6'

P = {
    'declared_utc': None,
    'status': 'declared before any P3 fit. Development-set sensitivity analyses on the already-explored 2,000-image pilot; not held-out validation; no significance tests.',
    'inherits': 'reports/week6/P1_EXPERIMENT_PROTOCOL.json (dataset, masks, metrics, method hyperparameters, label-blind staging)',
    'task_3_1_robustness': {
        'subsets': 'phase-2 subsets reused exactly: seeds 2026-2030, 1,600 rows each without replacement, sorted (results/phase2/evaluation/<enc>/subsets/<seed>/rows.npy)',
        'representation': 'centered_l2 refit inside each subset (phase-2 saved subset arrays reused after hash/recompute check)',
        'encoders': ['clip', 'siglip'],
        'methods': ['kmeans (seeds 0-4, n_init 10)', 'agglomerative_average_cosine (deterministic)', 'spectral_knn (15-NN, seeds 0-4)'],
        'k_grid': [2, 3, 5, 8, 10, 15, 20],
        'principal_k': 10,
        'blind_k': {'silhouette': 'max seed-mean cosine silhouette over k_grid, ties -> smaller k, per subset x encoder x method',
                    'eigengap': 'largest gap of the symmetric normalized Laplacian of the subset 15-NN graph, restricted to k_grid (spectral only)',
                    'evaluation': 'HS3 AMI at the selected k computed only after selection files are written'},
        'stability': ['seed-0 subset partition vs full-pilot partition restricted to subset rows (ARI)', 'pairwise subset partitions on common rows (ARI)'],
        'parity_check': 'refit k-means/agglomerative k=10 on subsets must equal phase-2 saved subset predictions'},
    'task_3_2_ami_vs_ari': {
        'configuration': 'SigLIP centered_l2 k=10: spectral_knn seed 0 vs agglomerative_average_cosine vs kmeans seed 0 (P2 saved assignments); CLIP repeated',
        'analyses': ['cluster sizes', 'cluster x HS3 contingency', 'purity (descriptive)', 'dominant class per cluster', 'class split across clusters', 'class merges within clusters', 'pair-counting precision/recall', 'per-cluster contribution to mutual information'],
        'examples': 'mixed clusters chosen by a descriptive rule fixed here: the 2 clusters with the most HS3-labelled rows whose top HS3 class is < 60% of labelled members, for SigLIP spectral seed 0; reported with saved metadata (title, HS3) and local thumbnails'},
    'task_3_3_preprocessing': {
        'representations': ['l2', 'centered_l2', 'pca50', 'pca50_l2', 'pca100_l2'],
        'representation_source': 'results/phase2/<enc>/representations/*.npy (hash-checked)',
        'per_museum_centering': 'NOT run: museums contribute 1-14 images in the pilot (a 1-image museum centers to the zero vector); distinct from global centering; deferred to the larger dataset',
        'methods': ['kmeans (seeds 0-4)', 'agglomerative_average_cosine', 'spectral_knn (seeds 0-4)'],
        'ks': [5, 10, 20], 'principal_k': 10,
        'parity_check': 'k-means/agglomerative must reproduce phase-2 predictions; spectral centered_l2/pca50_l2 must reproduce P2 week-6 predictions'},
    'task_3_4_rq3': 'readiness audit only; no download of model weights',
    'metrics': ['HS3 AMI/NMI/ARI (also HS1/HS2)', 'cosine silhouette', 'largest cluster %', 'actual k', 'fit seconds'],
    'dtype': 'native float32 representations (P1 amendment A1)',
    'threads': 4,
    'output_dir': 'reports/week6/p3_experiments/',
}


def main():
    path = OUT / 'P3_EXPERIMENT_PROTOCOL.json'
    if path.exists():
        print('already declared; not overwriting'); return
    P['declared_utc'] = datetime.now(timezone.utc).isoformat()
    P['environment'] = {'python': sys.version.split()[0], 'platform': platform.platform(),
                        'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'scikit-learn', 'pandas', 'matplotlib', 'threadpoolctl']}}
    P['protocol_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path.write_text(json.dumps(P, indent=2)); print('declared', P['declared_utc'])


if __name__ == '__main__':
    main()
