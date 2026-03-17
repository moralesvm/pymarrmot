
import numpy as np

def uh_3_half(d_base: float, delta_t: float) -> np.ndarray:
    """
    Unit Hydrograph with half a triangle (linear).

    Parameters
    ----------
    d_base : float
        Time base of routing delay [days].
    delta_t : float
        Time step size [days].
    
    Returns
    -------
    np.ndarray
        Unit hydrograph as a 2-row array:
        - First row contains coefficients to split flow at each of `n` timesteps forward.
        - Second row contains zeros (still-to-flow values).
    """
    # Calculate the delay
    delay = d_base / delta_t
    if delay == 0:
        delay = 1  # Prevent division by zero, any value below t = 1 means no delay
    
    # Time series for which we need UH ordinates
    tt = np.arange(1, int(np.ceil(delay)) + 1)
    
    # Calculate fraction of flow per time step
    ff = 1 / (0.5 * delay**2)
    
    # Initialize the Unit Hydrograph array
    UH = np.zeros(len(tt))
    
    # Populate the Unit Hydrograph
    for t in range(1, len(tt) + 1):
        if t <= delay:
            UH[t - 1] = ff * (0.5 * t**2 - 0.5 * (t - 1)**2)
        else:
            UH[t - 1] = ff * (0.5 * delay**2 - 0.5 * (t - 1)**2)
    
    # Add the second row of zeros to represent "still-to-flow" values
    UH_full = np.vstack((UH, np.zeros_like(UH)))

    return UH_full