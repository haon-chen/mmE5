import numpy as np


def get_pred(qry_t, tgt_t, normalization=False):
    """
    Use L2 norms.
    """
    if normalization:
        qry_t_norm = np.linalg.norm(qry_t)
        tgt_t_norms = np.linalg.norm(tgt_t, axis=1)
        scores = np.dot(tgt_t, qry_t) / (tgt_t_norms * qry_t_norm)
    else:
        scores = np.dot(tgt_t, qry_t)
    pred = np.argmax(scores)
    return scores, pred

def precision_at_k(scores, relevant_indices, k):
    """
    Calculate Precision@k.

    Args:
        qry_t (np.ndarray): Query vector of shape (d,).
        tgt_t (np.ndarray): Target matrix of shape (N, d).
        relevant_indices (set): Indices of relevant targets.
        k (int): The rank position.
        normalization (bool): If True, apply L2 normalization to query and targets.

    Returns:
        float: Precision@k value.
    """
    top_k_indices = np.argsort(scores)[-k:][::-1]
    hits = sum(1 for idx in top_k_indices if idx in relevant_indices)
    return hits / k

def recall_at_k(scores, relevant_indices, k):
    """
    Calculate Recall@k.

    Args:
        qry_t (np.ndarray): Query vector of shape (d,).
        tgt_t (np.ndarray): Target matrix of shape (N, d).
        relevant_indices (set): Indices of relevant targets.
        k (int): The rank position.
        normalization (bool): If True, apply L2 normalization to query and targets.

    Returns:
        float: Recall@k value.
    """
    top_k_indices = np.argsort(scores)[-k:][::-1]
    hits = sum(1 for idx in top_k_indices if idx in relevant_indices)
    return hits / len(relevant_indices)
