# Task 1: CLIP transformation ablation

Same frozen 2,000 images; 1,549 evaluated HS3 labels. K-Means: seeds 0–4, n_init=10. Average linkage: cosine, deterministic. Fixed k=5/10/20/40. Descriptive evaluation on the existing pilot, not held-out estimation.

| Representation | Method | k | HS3 AMI mean | seed SD | ARI | NMI | largest cluster % | cosine silhouette |
|---|---|---|---|---|---|---|---|---|
| centered_l2 | average_cosine | 5 | 0.3793 | nan | 0.2379 | 0.3883 | 37.10 | 0.0929 |
| centered_l2 | average_cosine | 10 | 0.4591 | nan | 0.3662 | 0.4729 | 24.35 | 0.1186 |
| centered_l2 | average_cosine | 20 | 0.4597 | nan | 0.3752 | 0.4818 | 16.70 | 0.1073 |
| centered_l2 | average_cosine | 40 | 0.4661 | nan | 0.3656 | 0.5004 | 12.15 | 0.0994 |
| centered_l2 | kmeans | 5 | 0.4001 | 0.0040 | 0.2515 | 0.4087 | 36.55 | 0.1336 |
| centered_l2 | kmeans | 10 | 0.4400 | 0.0066 | 0.3063 | 0.4542 | 14.57 | 0.1345 |
| centered_l2 | kmeans | 20 | 0.4476 | 0.0128 | 0.2660 | 0.4709 | 8.07 | 0.1199 |
| centered_l2 | kmeans | 40 | 0.4406 | 0.0043 | 0.2111 | 0.4791 | 5.30 | 0.1151 |
| l2 | average_cosine | 5 | 0.0035 | nan | 0.0007 | 0.0087 | 99.65 | 0.2124 |
| l2 | average_cosine | 10 | 0.0322 | nan | 0.0056 | 0.0481 | 98.65 | 0.1183 |
| l2 | average_cosine | 20 | 0.0403 | nan | 0.0083 | 0.0709 | 97.05 | 0.0432 |
| l2 | average_cosine | 40 | 0.2302 | nan | 0.0805 | 0.2745 | 72.15 | 0.0169 |
| l2 | kmeans | 5 | 0.3933 | 0.0031 | 0.2546 | 0.4021 | 34.11 | 0.1175 |
| l2 | kmeans | 10 | 0.4492 | 0.0057 | 0.3103 | 0.4632 | 17.09 | 0.1187 |
| l2 | kmeans | 20 | 0.4439 | 0.0048 | 0.2650 | 0.4674 | 7.57 | 0.1036 |
| l2 | kmeans | 40 | 0.4422 | 0.0030 | 0.1978 | 0.4805 | 4.82 | 0.0876 |
| pca100_l2 | average_cosine | 5 | 0.3507 | nan | 0.2348 | 0.3601 | 30.60 | 0.1026 |
| pca100_l2 | average_cosine | 10 | 0.4621 | nan | 0.3648 | 0.4758 | 22.95 | 0.1380 |
| pca100_l2 | average_cosine | 20 | 0.4657 | nan | 0.3924 | 0.4876 | 20.15 | 0.1142 |
| pca100_l2 | average_cosine | 40 | 0.4524 | nan | 0.3581 | 0.4885 | 12.35 | 0.1175 |
| pca100_l2 | kmeans | 5 | 0.4016 | 0.0018 | 0.2598 | 0.4103 | 35.37 | 0.1594 |
| pca100_l2 | kmeans | 10 | 0.4391 | 0.0030 | 0.3139 | 0.4533 | 13.53 | 0.1635 |
| pca100_l2 | kmeans | 20 | 0.4461 | 0.0095 | 0.2669 | 0.4694 | 7.34 | 0.1477 |
| pca100_l2 | kmeans | 40 | 0.4392 | 0.0063 | 0.2014 | 0.4777 | 5.29 | 0.1401 |
| pca50 | average_cosine | 5 | 0.3942 | nan | 0.2499 | 0.4029 | 32.40 | 0.1404 |
| pca50 | average_cosine | 10 | 0.4610 | nan | 0.3663 | 0.4747 | 21.55 | 0.1559 |
| pca50 | average_cosine | 20 | 0.4601 | nan | 0.3645 | 0.4823 | 17.95 | 0.1417 |
| pca50 | average_cosine | 40 | 0.4821 | nan | 0.3832 | 0.5167 | 13.75 | 0.1351 |
| pca50 | kmeans | 5 | 0.3924 | 0.0032 | 0.2529 | 0.4012 | 34.71 | 0.1841 |
| pca50 | kmeans | 10 | 0.4458 | 0.0069 | 0.3118 | 0.4598 | 16.06 | 0.1877 |
| pca50 | kmeans | 20 | 0.4415 | 0.0084 | 0.2728 | 0.4651 | 7.69 | 0.1765 |
| pca50 | kmeans | 40 | 0.4373 | 0.0091 | 0.2048 | 0.4760 | 5.30 | 0.1569 |
| pca50_l2 | average_cosine | 5 | 0.3942 | nan | 0.2499 | 0.4029 | 32.40 | 0.1404 |
| pca50_l2 | average_cosine | 10 | 0.4610 | nan | 0.3663 | 0.4747 | 21.55 | 0.1559 |
| pca50_l2 | average_cosine | 20 | 0.4601 | nan | 0.3645 | 0.4823 | 17.95 | 0.1417 |
| pca50_l2 | average_cosine | 40 | 0.4821 | nan | 0.3832 | 0.5167 | 13.75 | 0.1351 |
| pca50_l2 | kmeans | 5 | 0.4017 | 0.0024 | 0.2678 | 0.4103 | 33.69 | 0.1841 |
| pca50_l2 | kmeans | 10 | 0.4409 | 0.0078 | 0.3217 | 0.4551 | 13.69 | 0.1927 |
| pca50_l2 | kmeans | 20 | 0.4469 | 0.0074 | 0.2726 | 0.4702 | 7.47 | 0.1809 |
| pca50_l2 | kmeans | 40 | 0.4453 | 0.0086 | 0.2101 | 0.4834 | 5.22 | 0.1707 |

Post-PCA positive row scaling cannot change cosine distances mathematically. Check saved partition agreement in `clip/cosine_normalization_check.csv`; any disagreement would be a numerical tie effect, not a normalization benefit.

Centering is isolated by raw L2 vs centered full-dimensional L2. Dimensionality is examined by centered full-dimensional vs PCA50+L2 and PCA100+L2. Post-PCA normalization is isolated by PCA50 vs PCA50+L2. For K-Means it can change Euclidean geometry; for average cosine it should not.

Full per-seed metrics/runtimes, exact silhouettes on all images, representations, PCA parameters, cluster-size distributions and predictions are saved under clip/.

## Interpretation from completed ablations

Centering alone removes most of the average-linkage collapse. At k=5, raw L2 gives HS3 AMI 0.0035 and a 99.65% largest cluster; centered full-dimensional L2 gives AMI 0.3793 and 37.10%; PCA50+L2 gives AMI 0.3942 and 32.40%. This rescue therefore does not require dimensionality reduction. At k=10, centered full-dimensional AMI 0.4591 is close to PCA50 AMI 0.4610. At k=40, PCA50 reaches 0.4821 versus centered 0.4662, but PCA100 reaches 0.4524: no consistent or monotonic dimensionality benefit is established.

PCA50 with and without row normalization yields exactly identical cosine-average-link partitions at all four k values (partition ARI=1). Post-PCA L2 cannot explain this method's improvement. K-Means is sensitive to row normalization, but no representation consistently improves HS3 AMI across k; raw L2 at k=10 has five-seed mean AMI 0.4492 versus centered 0.4400. These are conditional descriptive results, not an unbiased best-method estimate.

Independent audit passed: 360 metric rows, 240 silhouettes, 2,000 image hashes, and all 96 overlapping original-pilot rows. The original reports/results are unchanged.
