import numpy as np

def uh_8_delay(t_delay, delta_t):

    """

    Unit Hydrograph [days] with a pure delay (no transformation).



    Parameters

    ----------

    t_delay : float

        Flow delay [d].

    delta_t : float

        Time step size [d].



    Returns

    -------

    UH : numpy.ndarray

        Unit hydrograph [nx2].

        uh's first row contains coefficients to split flow at each

        of n timesteps forward, the second row contains zeros now,

        these are the still-to-flow values.

    """

    delay = t_delay / delta_t

    ord1 = 1 - t_delay + np.floor(t_delay)

    ord2 = t_delay - np.floor(t_delay)

    t_start = np.floor(delay)

    UH = np.zeros((2, int(t_delay) + 2))

    UH[0, int(t_start)] = ord1

    UH[0, int(t_start) + 1] = ord2

    ##UH[1, :] = np.zeros(UH.shape)


    return UH
