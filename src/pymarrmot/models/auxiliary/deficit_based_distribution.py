import numpy as np

def deficit_based_distribution(s1, s1max, s2, s2max):
    """
    Calculates a fractional split for two stores,
    based on the relative deficit in each.

    Parameters
    ----------
    s1 : float
        Current storage of the first store.
    s1max : float
        Maximum storage of the first store.
    s2 : float
        Current storage of the second store.
    s2max : float
        Maximum storage of the second store.

    Returns
    -------
    f1 : float
        Fractional split for the first store.
    f2 : float
        Fractional split for the second store.
    """
    # Convert inputs to numpy arrays for computational efficiency
    s1, s1max, s2, s2max = map(np.array, [s1, s1max, s2, s2max])

    # Calculate relative deficits
    rd1 = (s1 - s1max) / s1max
    rd2 = (s2 - s2max) / s2max

    # Calculate fractional split
    if np.any(rd1 + rd2 != 0):
        # Deficit exists and can be used to compute the split
        f1 = rd1 / (rd1 + rd2)
        f2 = rd2 / (rd1 + rd2)
    else:
        # Both deficits are zero, and we revert to distribution based on
        # relative maximum store size
        f1 = s1max / (s1max + s2max)
        f2 = s2max / (s1max + s2max)

    return f1, f2