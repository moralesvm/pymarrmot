import numpy as np

def route(flux_in: float, uh: np.ndarray) -> float:
    """
    Calculates the output of a unit hydrograph at the current timestep after routing a flux through it.

    Parameters
    ----------
    flux_in : float
        Input flux.
    uh : np.ndarray
        Unit hydrograph of shape (2, n). The first row contains coefficients to split flow at each of n timesteps forward, 
        the second row contains still-to-flow values.

    Returns
    -------
    float
        Flux routed through the unit hydrograph at this step.
    """
    if uh.shape[0] != 2:
        raise ValueError("Unit hydrograph (uh) must have two rows.")
        
    flux_out = uh[0, 0] * flux_in + uh[1, 0]
    return flux_out