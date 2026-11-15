"""Week 6: write P1_EXPERIMENT_PROTOCOL.json BEFORE any P2 fit. Refuses to overwrite."""
import json, sys, platform, importlib.metadata, hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/week6'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


PROTOCOL = {
    'declared_utc': None,
    'status': 'declared before any P2 clustering fit; HS labels had been seen in phase 1/2 on this same pilot, so this is a development-set protocol, not a preregistered held-out test',
    'dataset': {
        'id': 'mimo_pilot_2000',
        'sample_csv': 'data/mimo_pilot/sample_pilot.csv',
        'ordered_image_ids': 'results/phase2/inputs/image_order.csv',
        'ordered_images_sha256': '07550be6c2d32c5b8e517069e1f2cc2536fffb49554d9b2a71a57ac3339565a8',
        'n_rows_clustered': 2000,
        'rows_clustered': 'all 2,000 rows in fixed order, including rows with missing HS labels',
        'larger_benchmark': 'the 59,661-photo cleaned set / 27,348 scored objects are NOT available locally and are NOT used',
        'held_out': 'no reserved held-out split exists locally; none is accessed. All results are descriptive development-set results.'},
    'evaluation': {
        'levels': ['hs1', 'hs2', 'hs3'],
        'masks': 'results/phase2/inputs/evaluation_masks.csv (sha256 d333e523...), nonmissing label per level; identical for every method/encoder',
        'n_evaluated': {'hs1': 1565, 'hs2': 1563, 'hs3': 1549},
        'class_frequency_filter': 'none',
        'metrics': ['AMI (arithmetic, sklearn default)', 'NMI (arithmetic)', 'ARI'],
        'internal_metrics': ['silhouette, cosine distance, exact on all clustered rows of the representation the method was fit on', 'silhouette, euclidean (recorded)'],
        'labels_usage': 'HS labels are read only by the scoring function after all fits and k-selection are complete. They never enter features, hyperparameters or k selection.'},
    'encoders': {
        'clip': {'model': 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K (OpenCLIP)', 'dim': 512, 'cache': 'data/mimo_pilot/embeddings/a0fbb271....npy'},
        'siglip': {'model': 'timm/ViT-B-16-SigLIP rev 41f575766f40e752fdd1383e9565b7f02388c1c4', 'dim': 768, 'cache': 'results/phase2/siglip/embeddings.npy'},
        'dinov2': {'status': 'not available locally; not downloaded without approval; excluded from P2'}},
    'representations': {
        'centered_l2': 'primary; phase-2 predeclared anchor representation (reused saved array results/phase2/<enc>/representations/centered_l2.npy)',
        'pca50_l2': 'secondary; reused saved array (non-whitened full-SVD PCA50 then L2)'},
    'methods': {
        'kmeans': {'n_init': 10, 'max_iter': 300, 'tol': 1e-4, 'algorithm': 'lloyd', 'seeds': [0, 1, 2, 3, 4]},
        'agglomerative_average_cosine': {'linkage': 'average', 'metric': 'cosine', 'deterministic': True, 'seeds': [0]},
        'gmm_diag': {'covariance_type': 'diag', 'n_init': 3, 'max_iter': 200, 'tol': 1e-3, 'reg_covar': 1e-6, 'init_params': 'kmeans', 'seeds': [0, 1, 2, 3, 4]},
        'spectral_knn': {'affinity': 'nearest_neighbors', 'n_neighbors': 15, 'assign_labels': 'kmeans', 'eigen_solver': 'arpack', 'n_init': 10, 'seeds': [0, 1, 2, 3, 4], 'note': 'kNN graph on euclidean distance of unit vectors (rank-equivalent to cosine)'},
        'hdbscan': {'min_cluster_size': [20, 50], 'min_samples': 'None (= min_cluster_size)', 'metric': 'euclidean', 'cluster_selection_method': 'eom', 'deterministic': True, 'implementation': 'sklearn.cluster.HDBSCAN',
                    'noise_scoring': 'two variants reported: (a) noise_as_cluster: noise label -1 is its own cluster, all masked rows scored; (b) assigned_only: only non-noise masked rows scored, with coverage reported. (a) is the primary, conservative number.'}},
    'k_grid': [2, 3, 5, 8, 10, 15, 20],
    'fixed_k_baseline': {'k': 10, 'reason': 'phase-2 predeclared anchor (centered_l2, k=10), chosen before phase-2 scores'},
    'blind_k': {
        'primary_criterion': 'maximum cosine silhouette over k_grid; for stochastic methods, maximum of the seed-mean silhouette; ties -> smaller k',
        'secondary_criteria': {'elbow_kmeans': 'Kneedle-style maximum distance to the chord of normalized (k, seed-mean inertia) over k_grid',
                               'bic_gmm': 'minimum seed-mean BIC of the diagonal GMM over k_grid',
                               'eigengap': 'largest gap lambda_{k+1}-lambda_k of the symmetric normalized Laplacian of the same symmetrized 15-NN connectivity graph, restricted to k in k_grid'},
        'after_selection': 'evaluate the partition(s) at the selected k against HS1/HS2/HS3; report mean and SD over seeds; compare to fixed k=10 and to the post-hoc best-AMI k (labelled as oracle, not blind)'},
    'seeds': {'clustering': [0, 1, 2, 3, 4], 'aggregation': 'mean and SD across seeds; no best-seed selection'},
    'runtime': 'wall-clock seconds of fit_predict only (time.perf_counter), threadpool_limits(4); silhouette time recorded separately; machine shared with other processes, so times are indicative only',
    'output_dir': 'reports/week6/experiments/ (runs.csv, scores.csv, cluster_sizes.csv, blind_k_curves.csv, assignments/*.npy, environment.json)',
    'preserve': 'no file outside reports/week6/ is written',
}


def main():
    path = OUT / 'P1_EXPERIMENT_PROTOCOL.json'
    if path.exists():
        print('protocol already declared; not overwriting'); return
    PROTOCOL['declared_utc'] = datetime.now(timezone.utc).isoformat()
    PROTOCOL['environment'] = {'python': sys.version.split()[0], 'platform': platform.platform(),
                               'compute': 'CPU only (12 logical cores, ~15 GB RAM, no CUDA); 4 BLAS threads',
                               'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'scikit-learn', 'pandas', 'matplotlib', 'threadpoolctl', 'torch', 'open_clip_torch', 'timm']}}
    PROTOCOL['protocol_source_sha256'] = sha(__file__)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(PROTOCOL, indent=2))
    print('declared', PROTOCOL['declared_utc'])


if __name__ == '__main__':
    main()
