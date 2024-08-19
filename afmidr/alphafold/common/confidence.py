import numpy as np
import scipy.special
from typing import Optional


def _calculate_bin_centers(breaks: np.ndarray):
    """Gets the bin centers from the bin edges.

    Args:
      breaks: [num_bins - 1] the error bin edges.

    Returns:
      bin_centers: [num_bins] the error bin centers.
    """
    step = breaks[1] - breaks[0]

    # Add half-step to get the center
    bin_centers = breaks + step / 2
    # Add a catch-all bin at the end.
    bin_centers = np.concatenate([bin_centers, [bin_centers[-1] + step]], axis=0)
    return bin_centers


def predicted_tm_score(
    probs: np.ndarray,
    breaks: np.ndarray,
    residue_weights: Optional[np.ndarray] = None,
    asym_id: Optional[np.ndarray] = None,
    interface: bool = False,
    receptor_mask: Optional[np.ndarray] = None,
    ligand_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    # residue_weights has to be in [0, 1], but can be floating-point, i.e. the
    # exp. resolved head's probability.
    if residue_weights is None:
        residue_weights = np.ones(probs.shape[0])

    bin_centers = _calculate_bin_centers(breaks)

    num_res = int(probs.shape[0])
    # Clip num_res to avoid negative/undefined d0.
    clipped_num_res = max(num_res, 19)

    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
    # "Scoring function for automated assessment of protein structure template
    # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

    # TM-Score term for every bin.
    tm_per_bin = 1.0 / (1 + np.square(bin_centers) / np.square(d0))
    # E_distances tm(distance).
    predicted_tm_term = np.sum(probs * tm_per_bin, axis=-1)

    if ligand_mask is not None and receptor_mask is not None:
        pair_mask = receptor_mask[:, None] * ligand_mask[None, :]
    else:
        pair_mask = np.ones(shape=(num_res, num_res), dtype=bool)
        if interface:
            pair_mask *= asym_id[:, None] != asym_id[None, :]

    predicted_tm_term *= pair_mask

    pair_residue_weights = pair_mask * (
        residue_weights[None, :] * residue_weights[:, None]
    )
    normed_residue_mask = pair_residue_weights / (
        1e-8 + np.sum(pair_residue_weights, axis=-1, keepdims=True)
    )
    per_alignment = np.sum(predicted_tm_term * normed_residue_mask, axis=-1)
    return np.asarray(per_alignment[(per_alignment * residue_weights).argmax()])


def predicted_per_residue_tm_score(
    probs: np.ndarray,
    breaks: np.ndarray,
    residue_weights: Optional[np.ndarray] = None,
    asym_id: Optional[np.ndarray] = None,
    interface: bool = False,
    receptor_asym_id: Optional[str] = None,
    ligand_asym_id: Optional[str] = None,
) -> np.ndarray:
    # residue_weights has to be in [0, 1], but can be floating-point, i.e. the
    # exp. resolved head's probability.
    if residue_weights is None:
        residue_weights = np.ones(probs.shape[0])

    bin_centers = _calculate_bin_centers(breaks)

    num_res = int(probs.shape[0])
    # Clip num_res to avoid negative/undefined d0.
    clipped_num_res = max(num_res, 19)

    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
    # "Scoring function for automated assessment of protein structure template
    # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

    # TM-Score term for every bin.
    tm_per_bin = 1.0 / (1 + np.square(bin_centers) / np.square(d0))
    # E_distances tm(distance).
    predicted_tm_term = np.sum(probs * tm_per_bin, axis=-1)

    if ligand_asym_id is not None and receptor_asym_id is not None:
        receptor_mask = asym_id == receptor_asym_id
        ligand_mask = asym_id == ligand_asym_id
        pair_mask = receptor_mask[:, None] * ligand_mask[None, :]
    else:
        pair_mask = np.ones(shape=(num_res, num_res), dtype=bool)
        if interface:
            pair_mask *= asym_id[:, None] != asym_id[None, :]

    predicted_tm_term *= pair_mask

    pair_residue_weights = pair_mask * (
        residue_weights[None, :] * residue_weights[:, None]
    )
    normed_residue_mask = pair_residue_weights / (
        1e-8 + np.sum(pair_residue_weights, axis=-1, keepdims=True)
    )
    per_alignment = np.sum(predicted_tm_term * normed_residue_mask, axis=-1)
    # return np.asarray(per_alignment[(per_alignment * residue_weights).argmax()])
    return per_alignment
