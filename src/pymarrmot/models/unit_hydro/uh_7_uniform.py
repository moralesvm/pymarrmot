import numpy as np

def uh_7_uniform(d_base: float, delta_t: float) -> np.ndarray:
    """
    Unit Hydrograph with uniform spread.

    Parameters
    ----------
    d_base : float
        Time base of routing delay [d].
    delta_t : float
        Time step size [d].

    Returns
    -------
    UH : np.ndarray
        Unit hydrograph [nx2]. The first row contains coefficients to split 
        flow at each of n timesteps forward, the second row contains zeros, 
        which are the still-to-flow values.

    Examples
    --------
    For d_base = 3.8 [days] and delta_t = 1:
    UH[0, 0] = 0.26  [% of inflow]
    UH[0, 1] = 0.26
    UH[0, 2] = 0.26
    UH[0, 3] = 0.22
    """
    # TIME STEP SIZE
    delay = d_base / delta_t
    tt = np.arange(1, int(np.ceil(delay)) + 1)  # time series for which we need UH ordinates [days]

    # EMPTIES
    UH = np.full((2, len(tt)), np.nan)

    # FRACTION FLOW
    ff = 1 / delay  # fraction of flow per time step

    # UNIT HYDROGRAPH
    for t in range(1, int(np.ceil(delay)) + 1):
        if t < delay:
            UH[0, t-1] = ff
        else:
            UH[0, t-1] = (delay % (t-1)) * ff
    
    UH[1, :] = 0  # second row contains zeros

    return UH
