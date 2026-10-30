"""Dataset-specific triplet transition matrices for HOI anticipation.

Matrices use T[i, j] = P(current triplet i | future triplet j). Their row and
column order must match the selected dataset's text-label dictionary exactly.
Unavailable variants fall back to identity as a pipeline-only baseline.
"""

import numpy as np

from datasets.ag_text_label import ag_text_label
from datasets.vidhoi_text_label import vidhoi_text_label

# Fill these in with the generated matrices, e.g. {1: [[...], ...], 3: ..., 5: ..., 7: ...}
HORIZONS = (1, 3, 5, 7)
NUM_HOI_CLASSES = {
    'ag': len(ag_text_label),
    'vidhoi': len(vidhoi_text_label),
}

def _validate_matrices(matrices, dataset_file):
    expected_size = NUM_HOI_CLASSES[dataset_file]
    if set(matrices) != set(HORIZONS):
        raise ValueError(
            f'{dataset_file} transition matrices must provide horizons {HORIZONS}, '
            f'got {tuple(matrices)}'
        )
    for horizon, matrix in matrices.items():
        rows = len(matrix)
        cols = len(matrix[0]) if rows else 0
        valid_rows = rows == expected_size and all(
            len(row) == expected_size for row in matrix
        )
        if not valid_rows:
            raise ValueError(
                f'{dataset_file} horizon {horizon} matrix is {rows}x{cols}, '
                f'expected {expected_size}x{expected_size}'
            )


def _identity_matrices(dataset_file):
    identity = np.eye(NUM_HOI_CLASSES[dataset_file], dtype=np.float32)
    return {h: identity.copy() for h in HORIZONS}


def _load_detant_matrices(dataset_file):
    # Import only the selected dataset's large constants to avoid unnecessary RAM use.
    if dataset_file == 'ag':
        from datasets.priors.transition_matrix_triplet_ag import (
            TRIPLET_TRANSITION_MATRICES_DETANT,
        )
        return TRIPLET_TRANSITION_MATRICES_DETANT
    else:
        from datasets.priors.transition_matrix_triplet_vidhoi import (
            TRIPLET_TRANSITION_MATRICES_DETANT_VAL,
        )
    return TRIPLET_TRANSITION_MATRICES_DETANT_VAL


def extract_transition_matrix(mode, dataset_file='vidhoi'):
    """Return {horizon: matrix} for the requested mode.

    Mirrors the verb-space API so the eval code is identical across repos.
    """
    if dataset_file not in NUM_HOI_CLASSES:
        raise ValueError(f"Transition-matrix evaluation does not support '{dataset_file}'")

    normalized_mode = mode.lower()
    variant = None
    if 'orig' in normalized_mode:
        variant = 'original'
    elif 'detant' in normalized_mode and 'v2' in normalized_mode:
        variant = 'detant_v2'
    elif 'detant' in normalized_mode:
        variant = 'detant'

    matrices = _load_detant_matrices(dataset_file) if variant == 'detant' else {}
    if matrices:
        _validate_matrices(matrices, dataset_file)
        print(f'Loading {dataset_file} {variant} triplet transition matrices.')
        return matrices

    size = NUM_HOI_CLASSES[dataset_file]
    print(f"No {dataset_file} triplet transition matrix available for mode '{mode}'. "
          f'Falling back to identity ({size}x{size}).')
    return _identity_matrices(dataset_file)
