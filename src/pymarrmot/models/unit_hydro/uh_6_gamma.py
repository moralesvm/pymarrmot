import numpy as np
from scipy.integrate import quad
from scipy.special import gamma

def uh_6_gamma(n: float, k: float, delta_t: float) -> np.ndarray:
    """
    Unit Hydrograph from gamma function.

    Parameters
    ----------
    n : float
        Shape parameter [-].
    k : float
        Time delay for flow reduction by a factor e [d].
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
    For n = 1, k = 3.8 [days], delta_t = 1:
    UH[0, 0] = 0.928  [% of inflow]
    UH[0, 1] = 0.067
    UH[0, 2] = 0.005
    UH[0, 3] = 0.000
    """
    UH = []

    t = 1
    while True:
        # Calculate the pdf of the gamma distribution at this timestep
        integral_value, _ = quad(
            lambda x: 1.0 / (k * gamma(n)) * (x / k) ** (n - 1) * np.exp(-x / k),
            (t - 1) * delta_t,
            t * delta_t
        )
        UH.append(integral_value)
        
        # If the new value of the UH is less than 0.1% of the peak, end the loop
        if UH[-1] < (max(UH) * 0.001):
            break
        
        # Go to the next step
        t += 1

    UH = np.array(UH)
    
    # Account for the truncated part of the UH
    tmp_excess = 1 - np.sum(UH)
    tmp_weight = UH / np.sum(UH)
    UH += tmp_weight * tmp_excess
    
    # Create the final UH array with a second row of zeros
    UH_full = np.vstack([UH, np.zeros_like(UH)])
    
    return UH_full