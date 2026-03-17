import numpy as np

def uh_2_full(d_base: float, delta_t: float) -> np.ndarray:
    """
    Creates a Unit Hydrograph (UH) with a full bell curve based on the GR4J model.

    Parameters
    ----------
    d_base : float
        Time base of routing delay [days].
    delta_t : float
        Time step size [days].

    Returns
    -------
    np.ndarray
        Unit hydrograph [nx2]. The first row contains coefficients to split 
        flow at each of n timesteps forward, the second row contains zeros 
        (still-to-flow values).
    """
    delay = d_base / delta_t

    #bug fix: 15July2024 - SAS - original code had 2*np.ceil(delay)+1
    tt = np.arange(1, np.ceil(delay) + 1)

    SH = np.zeros(len(tt) + 1)
    UH = np.zeros((2, len(tt)))

    for t in range(len(tt)):
        if t + 1 <= delay:
            SH[t + 1] = 0.5 * ((t + 1) / delay) ** (5 / 2)
        elif delay < t + 1 < 2 * delay:
            SH[t + 1] = 1 - 0.5 * (2 - (t + 1) / delay) ** (5 / 2)
        elif t + 1 >= 2 * delay:
            SH[t + 1] = 1

        UH[0, t] = SH[t + 1] - SH[t]

    UH[1, :] = 0
    
    return UH

