"""Unsupervised fault detection in transformer DGA data.

Public modules:
    data           - load & clean the raw Excel into a tidy DataFrame
    preprocessing  - log-transform, scaling, outlier handling, feature matrix
    conventional   - Duval Triangle 1 + IEC 60599 / Rogers ratio baselines
    models          - PyTorch Autoencoder and Variational Autoencoder
    clustering     - KMeans / GMM / HDBSCAN on the latent space
    anomaly        - reconstruction-error & classical anomaly detectors
    evaluation     - internal + external clustering metrics
    viz            - paper-quality plotting helpers
"""
__version__ = "0.1.0"
