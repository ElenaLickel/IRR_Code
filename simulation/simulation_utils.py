import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, log_loss
from scipy.stats import multivariate_normal
import numpy as np
from scipy.stats import multivariate_normal

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

np.random.seed(42)

import numpy as np

def generate_data(n_per_class, eta, K, means, shared_cov, class_covs):
    
    X_list, y_list = [], [] # Use _list suffix to clearly indicate they are lists of arrays

    for i in range(K):
        # Calculate the blended covariance matrix based on eta
        cov = (1 - eta) * shared_cov + eta * class_covs[i]
        
        # Add a small epsilon to the diagonal for numerical stability
        # This helps prevent LinAlgError due to singular or non-positive definite matrices
        #epsilon = 1e-6 
        #cov = cov + np.eye(cov.shape[0]) * epsilon
        
        # Generate multivariate normal samples for the current class
        X_list.append(np.random.multivariate_normal(means[i], cov, size=n_per_class))
        y_list.append(np.full(n_per_class, i))
    
    # Vertically stack all class feature arrays into a single data matrix
    X = np.vstack(X_list)
    # Horizontally stack all class label arrays into a single label vector
    y = np.hstack(y_list)
    
    # Permute the data to ensure samples from different classes are interleaved
    # This is important for robust splitting and training
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(len(y))

    X = X[perm]
    y = y[perm]
    
    return X, y

from sklearn.utils import shuffle

def setup_simulation(K, eta, n_train, print_covariances=False):
    np.random.seed(1)

    # Arrange class means in a grid pattern
    means = [np.array([i * 2, j * 2])
             for i in range(int(np.ceil(K**0.5)))
             for j in range(int(np.ceil(K**0.5)))]
    means = means[:K]

    shared_cov = np.array([[1, 0], [0, 1]])  # Identity
    n_per_class = n_train // K

    # Generate distinct elliptical and rotated covariances
    class_covs = []
    for k in range(K):
        angle = (np.pi / K) * k  # unique angle per class
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        scales = np.diag([1.0 + 0.5 * k, 0.3 + 0.2 * (K - k)])  # different eigenvalues
        cov_k = rotation @ scales @ rotation.T
        cov_k = (cov_k + cov_k.T) / 2 + 1e-6 * np.eye(2)  # ensure symmetry & stability
        class_covs.append(cov_k)

    # Generate data
    X_train, y_train = generate_data(n_per_class=n_per_class, eta=eta, K=K,
                                     means=means, shared_cov=shared_cov, class_covs=class_covs)

    X_train, y_train = shuffle(X_train, y_train, random_state=1)
    # Optionally print empirical covariances
    if print_covariances:
        print(f"\nEmpirical covariance matrices (η = {eta}):")
        for i in range(K):
            X_class = X_train[y_train == i]
            if len(X_class) > 1:
                cov = np.cov(X_class, rowvar=False)
                print(f"Class {i}:\n{cov}\n")
            else:
                print(f"Class {i}: Not enough samples to compute empirical covariance.\n")

    return X_train, y_train, means, shared_cov, class_covs

def setup_simulation_new(K, eta, n_train, spread=6.0, print_covariances=False):
    """
    Generates well-separated classes with clear covariance differences:
    - eta = 0: All classes identical covariance → LDA optimal, QDA overfits
    - eta → 1: Classes have very different shapes → QDA better, LDA suboptimal
    """
    np.random.seed(42)
    
    # Create moderately separated, clean centroid arrangement
    means = []
    if K == 2:
        means = [np.array([-2.5, 0.0]), np.array([2.5, 0.0])]
    elif K == 3:
        # Triangle arrangement for 3 classes - closer together
        means = [
            np.array([0.0, 2.5]),      # Top
            np.array([-2.2, -1.3]),   # Bottom left  
            np.array([2.2, -1.3])     # Bottom right
        ]
    else:
        # Regular polygon for K > 3 - smaller radius
        angles = np.linspace(0, 2*np.pi, K+1)[:-1]
        radius = 3.0
        for angle in angles:
            means.append(np.array([radius * np.cos(angle), radius * np.sin(angle)]))
    
    # Shared covariance: this will be the ONLY covariance at eta=0
    shared_cov = np.array([[1.2, 0.4],
                          [0.4, 1.0]])
    
    # Class-specific covariances: extremely distinct shapes to hurt LDA more
    class_covs = []
    
    if K >= 1:
        # Class 0: Very strong horizontal elongation
        class_covs.append(np.array([[4.5, 0.0],
                                   [0.0, 0.3]]))
    
    if K >= 2:
        # Class 1: Very strong vertical elongation
        class_covs.append(np.array([[0.3, 0.0],
                                   [0.0, 4.5]]))
    
    if K >= 3:
        # Class 2: Strong positive diagonal correlation with large spread
        class_covs.append(np.array([[2.8, 2.5],
                                   [2.5, 2.8]]))
    
    if K >= 4:
        # Class 3: Very strong negative diagonal correlation
        class_covs.append(np.array([[2.8, -2.5],
                                   [-2.5, 2.8]]))
    
    if K >= 5:
        # Class 4: Extremely small, tight circular - very different scale
        class_covs.append(np.array([[0.15, 0.0],
                                   [0.0, 0.15]]))
    
    # For additional classes beyond 5, create systematic patterns
    while len(class_covs) < K:
        k = len(class_covs)
        angle = k * np.pi / 4
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos_a, -sin_a],
                           [sin_a, cos_a]])
        
        # Alternate between very elongated and very compact
        if k % 2 == 0:
            base_cov = np.array([[4.0, 0.0], [0.0, 0.25]])
        else:
            base_cov = np.array([[0.2, 0.0], [0.0, 0.2]])
            
        rotated_cov = rotation @ base_cov @ rotation.T
        class_covs.append(rotated_cov)
    
    # Ensure all covariances are positive definite
    for i in range(len(class_covs)):
        class_covs[i] = (class_covs[i] + class_covs[i].T) / 2 + 1e-6 * np.eye(2)
    
    if print_covariances:
        print(f"Eta = {eta}")
        print(f"Shared covariance:\n{shared_cov}\n")
        for k in range(K):
            actual_cov = (1 - eta) * shared_cov + eta * class_covs[k]
            print(f"Class {k} centroid: {means[k]}")
            print(f"Actual covariance matrix:\n{actual_cov}")
            eigvals = np.linalg.eigvals(actual_cov)
            print(f"Eigenvalues: {eigvals}")
            print(f"Condition number: {np.max(eigvals)/np.min(eigvals):.2f}\n")
    
    # Generate data
    n_per_class = max(1, n_train // K)
    X_list, y_list = [], []
    for k in range(K):
        actual_cov = (1 - eta) * shared_cov + eta * class_covs[k]
        X_k = np.random.multivariate_normal(means[k], actual_cov, size=n_per_class)
        y_k = np.full(n_per_class, k)
        X_list.append(X_k)
        y_list.append(y_k)
    
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    
    # Shuffle
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    
    return X, y, means, shared_cov, class_covs


def generate_data_high(n_per_class, eta, K, means, shared_cov, class_covs, seed=42):
    """
    Generate synthetic data with blended covariances in higher dimensions.
    """

    X_list, y_list = [], []

    rng = np.random.default_rng(seed)

    for i in range(K):
        # Blend covariance matrix
        cov = (1 - eta) * shared_cov + eta * class_covs[i]
        cov = (cov + cov.T) / 2  # Ensure symmetry (important in high dimensions)

        # Defensive adjustment for numerical stability
        cov += 1e-6 * np.eye(cov.shape[0])

        # Generate samples
        X_c = rng.multivariate_normal(mean=means[i], cov=cov, size=n_per_class)
        y_c = np.full(n_per_class, i)

        X_list.append(X_c)
        y_list.append(y_c)

    # Stack and shuffle
    X = np.vstack(X_list)
    y = np.hstack(y_list)

    perm = rng.permutation(len(y))

    return X[perm], y[perm]


from sklearn.datasets import make_spd_matrix

def setup_simulation_high(K=7, eta=0.5, n_train=1000, D=20, spread=6.0, print_covariances=False):
    """
    Generate challenging high-dimensional data:
    - K: number of classes
    - eta: covariance blending parameter
    - n_train: total training samples
    - D: feature dimension
    """

    np.random.seed(42)

    # Class means: placed on a hypersphere
    angles = np.linspace(0, 2 * np.pi, K + 1)[:-1]
    means = []

    for angle in angles:
        direction = np.random.normal(size=D)
        direction /= np.linalg.norm(direction)  # Project to sphere
        means.append(direction * spread)

    # Shared covariance (η = 0): identity scaled to spread
    shared_cov = make_spd_matrix(D, random_state=0)
    shared_cov = (shared_cov + shared_cov.T) / 2
    shared_cov += 0.5 * np.eye(D)

    # Class-specific covariances (η = 1): highly distinct, random
    class_covs = []
    for k in range(K):
        cov_k = make_spd_matrix(D, random_state=100 + k)
        cov_k = (cov_k + cov_k.T) / 2
        cov_k += 0.1 * np.eye(D)  # Ensure full rank, avoid singularity
        class_covs.append(cov_k)

    # Optional inspection
    if print_covariances:
        print(f"η = {eta:.2f}")
        print(f"Shared Covariance (first 5x5 block):\n{shared_cov[:5, :5]}")
        for k in range(K):
            actual_cov = (1 - eta) * shared_cov + eta * class_covs[k]
            eigvals = np.linalg.eigvalsh(actual_cov)
            print(f"\nClass {k} Mean (first 5 dims): {means[k][:5]}")
            print(f"Class {k} Covariance (first 5x5 block):\n{actual_cov[:5, :5]}")
            print(f"Eigenvalue range: {eigvals.min():.4f} to {eigvals.max():.4f}")

    # Generate data
    n_per_class = max(1, n_train // K)
    X_list, y_list = [], []

    for k in range(K):
        cov_k = (1 - eta) * shared_cov + eta * class_covs[k]
        cov_k = (cov_k + cov_k.T) / 2
        cov_k += 1e-6 * np.eye(D)

        X_k = np.random.multivariate_normal(means[k], cov_k, size=n_per_class)
        y_k = np.full(n_per_class, k)

        X_list.append(X_k)
        y_list.append(y_k)

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    # Shuffle
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    return X, y, means, shared_cov, class_covs


def compute_bayes_predictions(X, means, shared_cov, class_covs, eta):
    K = len(means)
    n = X.shape[0]
    probs = np.zeros((n, K))
    priors = np.full(K, 1.0 / K)  # uniform priors


    for k in range(K):
        sigma_k = (1 - eta) * shared_cov + eta * class_covs[k]
        sigma_k = (sigma_k + sigma_k.T) / 2  # enforce symmetry
        sigma_k += np.eye(sigma_k.shape[0]) * 1e-6  # add epsilon for numerical stability

        rv = multivariate_normal(mean=means[k], cov=sigma_k)
        probs[:, k] = rv.pdf(X) * priors[k]

    probs /= probs.sum(axis=1, keepdims=True)
    predictions = np.argmax(probs, axis=1)
    return predictions, probs



def compute_lda_predictions(X_train, y_train, X_test):
    model = LinearDiscriminantAnalysis().fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    return preds, probs, model


def compute_qda_predictions(X_train, y_train, X_test):
    model = QuadraticDiscriminantAnalysis().fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    return preds, probs, model