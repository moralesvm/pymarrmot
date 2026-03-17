import numpy as np

def update_uh(uh: np.ndarray, flux_in: float) -> np.ndarray:
    """
    Calculates new still-to-flow values of a unit hydrograph based on a flux routed through it at this timestep.

    Parameters
    ----------
    uh : np.ndarray
        Unit hydrograph of shape (2, n). The first row contains coefficients to split flow at each of n timesteps forward,
        the second row contains still-to-flow values.
    flux_in : float
        Input flux.

    Returns
    -------
    np.ndarray
        Updated unit hydrograph of shape (2, n) with updated still-to-flow values.
    """
    if uh.shape[0] != 2:
        raise ValueError("Unit hydrograph (uh) must have two rows.")
    
    # Update the still-to-flow values
    uh[1, :] = (uh[0, :] * flux_in) + uh[1, :]
    uh[1, :] = np.roll(uh[1, :], -1)
    uh[1, -1] = 0
    
    return uh
