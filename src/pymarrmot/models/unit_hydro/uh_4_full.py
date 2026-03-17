import numpy as np
from scipy.integrate import quad

def uh_4_full(d_base: float, delta_t: float) -> np.ndarray:
    """
    Generates a unit hydrograph with a triangular shape (linear).

    Parameters
    ----------
    d_base : float
        Time base of routing delay [d].
    delta_t : float
        Time step size [d].

    Returns
    -------
    np.ndarray
        Unit hydrograph [nx2], where the first row contains coefficients 
        to split flow at each of n timesteps forward, and the second row 
        contains zeros (still-to-flow values).
    """
    # Time step size
    delay = d_base / delta_t
    if delay == 0:
        delay = 1  # any value below t = 1 means no delay, zero leads to problems
    tt = np.arange(1, int(np.ceil(delay)) + 1)  # time series for which we need UH ordinates [days]

    # Fraction of flow step size
    ff = 0.5 / (0.5 * (0.5 * delay) ** 2)
    d50 = 0.5 * delay

    # Triangle function
    def tri(t):
        return max(ff * (t - d50) * np.sign(d50 - t) + ff * d50, 0)

    # Unit hydrograph
    UH = np.zeros(len(tt))
    for t in range(len(tt)):
        UH[t], _ = quad(tri, t, t + 1)  # integrate the triangle function over each time step

    # Ensure UH sums to 1
    tmp_diff = 1 - np.sum(UH)
    tmp_weight = UH / np.sum(UH)
    UH = UH + tmp_weight * tmp_diff

    UH_2 = np.zeros_like(UH)
    UH = np.vstack((UH, UH_2))

    return UH