import numpy as np
from typing import List, Tuple


def find_nonzero_segments(y: np.ndarray) -> List[Tuple[int, int, int]]:
    """Find contiguous nonzero label segments in a 1D label sequence.

    Args:
        y (np.ndarray): 1D array of integer labels.
            - 0 = background (ignored)
            - Nonzero values = active segments to extract.

    Returns:
        List[Tuple[int, int, int]]: A list of segments, where each segment is a tuple:
            (label, start_index, end_index).
    """

    segments = []
    current_label = y[0]
    start = 0

    for i in range(1, len(y)):
        if y[i] != current_label:
            if current_label != 0:
                segments.append([start, i - 1, int(current_label), ])
            current_label = y[i]
            start = i

    # Append last segment if nonzero
    if current_label != 0:
        segments.append([start, len(y) - 1, int(current_label)])

    return segments


def fill_label_gaps(preds: np.ndarray, max_gap: int = 1) -> np.ndarray:
    """Fill small gaps (runs of zeros) in a label sequence if surrounded by the same label.

    Example:
        preds = [1, 1, 0, 1, 1] with max_gap=1 → [1, 1, 1, 1, 1]

    Args:
        preds (np.ndarray): 1D array of integer labels.
            - 0 = gap/no label
            - Nonzero values = labels
        max_gap (int, optional): Maximum allowed gap size (in samples) to fill. Defaults to 1.

    Returns:
        np.ndarray: A new array with gaps filled, same shape as preds.
    """

    preds = np.array(preds)  # ensure numpy array
    filled = preds.copy()

    i = 0
    while i < len(preds):
        if preds[i] == 0:  # gap starts
            start = i
            while i < len(preds) and preds[i] == 0:
                i += 1
            end = i  # first non-zero after gap

            # Check if surrounded by same label
            if start > 0 and end < len(preds) and (end - start) <= max_gap:
                if preds[start - 1] == preds[end] and preds[start - 1] != 0:
                    filled[start:end] = preds[start - 1]
        else:
            i += 1

    return filled
