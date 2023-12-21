import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def plot_histogram(positive_distances: np.ndarray, negative_distances: np.ndarray):
    assert positive_distances.ndim == 1
    assert negative_distances.ndim == 1
    plt.hist(positive_distances, bins=100, label='Positive')
    plt.hist(negative_distances, bins=100, label='Negative', alpha=0.7)
    plt.legend()
    plt.show()

def plot_roc(positive_distances: np.ndarray, negative_distances: np.ndarray) -> float:
    assert positive_distances.ndim == 1
    assert negative_distances.ndim == 1
    positive_labels = np.ones(shape=positive_distances.shape)
    negative_labels = np.zeros(shape=negative_distances.shape)
    distances = np.append(positive_distances, negative_distances)
    labels = np.append(positive_labels, negative_labels)
    lr_fpr, lr_tpr, _ = roc_curve(labels, distances)
    plt.plot([0,1], [0,1], linestyle='--', label='No Skill', color='gray')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Predictions')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

def get_roc_auc(positive_distances: np.ndarray, negative_distances: np.ndarray) -> float:
    assert positive_distances.ndim == 1
    assert negative_distances.ndim == 1
    positive_labels = np.ones(shape=positive_distances.shape)
    negative_labels = np.zeros(shape=negative_distances.shape)
    distances = np.append(positive_distances, negative_distances)
    labels = np.append(positive_labels, negative_labels)
    return roc_auc_score(labels, distances)