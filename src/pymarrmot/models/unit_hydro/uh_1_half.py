import numpy as np

def uh_1_half(d_base, delta_t):
    """
    uh_1_half Unit Hydrograph [days] with half a bell curve. GR4J-based

    Args:
    - d_base (float): time base of routing delay [d]
    - delta_t (float): time step size [d]

    Returns:
    - UH (ndarray): unit hydrograph [nx2]
                    uh's first row contains coefficients to split flow at each
                    of n timesteps forward, the second row contains zeros now,
                    these are the still-to-flow values.
    """

    # TIME STEP SIZE
    delay = d_base / delta_t
    if delay == 0:
        delay = 1  # any value below t = 1 means no delay,
                   # but zero leads to problems
    tt = np.arange(1, np.ceil(delay) + 1)  # Time series for which we need UH
                                           # ordinates [days]

    # EMPTIES
    SH = np.zeros(len(tt) + 1)
    SH[0] = 0
    UH = np.zeros(len(tt))

    # UNIT HYDROGRAPH
    for i, t in enumerate(tt, start=1):
        if t < delay:
            SH[i] = (t / delay) ** (5 / 2)
        elif t >= delay:
            SH[i] = 1

        UH[i - 1] = SH[i] - SH[i - 1]

    ##UH = np.column_stack((UH, np.zeros_like(UH)))
    zeros = np.zeros_like(UH)
    UH = np.vstack([UH, zeros])

    return UH
