import numpy as np

def NewtonRaphson(fun, x0, options=None):
    """
    Solve set of non-linear equations using Newton-Raphson method.
    [X, RESNORM, F, EXITFLAG] = NEWTONRAPHSON(FUN, X0, OPTIONS)
    FUN is a function handle that returns a vector of residuals equations, F,
    and takes a vector, x, as its only argument. When the equations are
    solved by x, then F(x) == zeros(size(F(:), 1)).

    Parameters:
    ----------
    fun : function handle
        Function that returns a vector of residuals equations and takes a vector, x, as its only argument.
    x0 : numpy array
        Vector of initial guesses.
    options : dict, optional
        Solver options.

    Returns:
    ----------
    x : numpy array
        Solution that solves the set of equations within the given tolerance.
    F : numpy array
        Residuals evaluated at x.
    exitflag : int
        An integer that corresponds to the output conditions.
        -1 = Jacobian matrix contains NaN or Inf values.
        0 = Maximum number of iterations reached without convergence.
        1 = Convergence achieved.
        2 = May have converged, but X is too close to XOLD

    References:
    ----------
    https://en.wikipedia.org/wiki/Newton's_method
    
    https://en.wikipedia.org/wiki/Newton's_method_in_optimization
    
    9.7 Globally Convergent Methods for Nonlinear Systems of Equations 383,
    Numerical Recipes in C, Second Edition (1992)
    """

    # Initialize
    x0 = np.array(x0).astype(float)
    x0 = np.array(x0).reshape(-1, 1)  # Ensure column vector
    defaultopt = {'TolX': 1e-12, 'TolFun': 1e-6, 'MaxIter': 1000}
    if options is None:
        options = defaultopt
    else:
        options = {**defaultopt, **options}

    # Get options
    TOLX = options.get('TolX', defaultopt['TolX'])
    TOLFUN = options.get('TolFun', defaultopt['TolFun'])
    MAXITER = options.get('MaxIter', defaultopt['MaxIter'])

    ALPHA = 1e-4
    MIN_LAMBDA = 0.1
    MAX_LAMBDA = 0.5

    # Check initial guess
    x = x0
    F = fun(x)
    nf = len(F)
    J = jacobian(fun, x, nf, F)
    exitflag = 1 if not (np.any(np.isnan(J)) or np.any(np.isinf(J))) else -1
    resnorm = np.linalg.norm(F, np.inf)
    resnorm0 = 100 * resnorm
    dx = np.zeros_like(x0)

    # Solver
    Niter = 0
    lambda_ = 1
    while (resnorm > TOLFUN or lambda_ < 1) and exitflag >= 0 and Niter <= MAXITER:
        if lambda_ == 1:
            Niter += 1
            if resnorm / resnorm0 > 0.2:
                J = jacobian(fun, x, nf, F)
                if np.any(np.isnan(J)) or np.any(np.isinf(J)):
                    exitflag = -1
                    break

            if np.linalg.cond(J) <= np.finfo(float).eps:
                dx = np.linalg.pinv(J) @ (-F)
            else:
                dx = -np.linalg.solve(J, F)
            dx = np.array(dx).reshape(-1, 1) # Ensure column vector

            g = F.T @ J
            slope = g @ dx
            fold = F.T @ F
            xold = x
            lambda_min = TOLX / np.max(np.abs(dx) / np.maximum(np.abs(xold), 1))

        if lambda_ < lambda_min:
            exitflag = 2
            break
        if np.any(np.isnan(dx)) or np.any(np.isinf(dx)):
            exitflag = -1
            break

        x = xold + dx * lambda_
        F = fun(x)
        f = F.T @ F

        lambda1 = lambda_
        if f > fold + ALPHA * lambda_ * slope:
            if lambda_ == 1:
                lambda_ = -slope / 2 / (f - fold - slope)
            else:

                lambda_ = float(lambda_)
                lambda2 = float(lambda2)
                lambda1 = float(lambda1)

                A = 1 / (lambda1 - lambda2)
                ## Modified for testing    
                B = np.array([[1 / lambda1**2, -1 / lambda2**2], [-lambda2 / lambda1**2, lambda1 / lambda2**2]])
                C = np.array([f - fold - lambda1 * slope, f2 - fold - lambda2 * slope])
                a, b = np.dot(A * B, C)
                if a == 0:
                    lambda_ = -slope / 2 / b
                else:
                    discriminant = b**2 - 3 * a * slope
                    if discriminant < 0:
                        lambda_ = MAX_LAMBDA * lambda1
                    elif b <= 0:
                        lambda_ = (-b + np.sqrt(discriminant)) / 3 / a
                    else:
                        lambda_ = -slope / (b + np.sqrt(discriminant))
                lambda_ = min(lambda_, MAX_LAMBDA * lambda1)

        elif np.isnan(f) or np.isinf(f):
            lambda_ = MAX_LAMBDA * lambda1
        else:
            lambda_ = 1

        if lambda_ < 1:
            lambda2 = lambda1
            f2 = f
            lambda_ = max(lambda_, MIN_LAMBDA * lambda1)
            continue

        resnorm0 = resnorm
        resnorm = np.linalg.norm(F, np.inf)

    if Niter >= MAXITER:
        exitflag = 0

    return x.flatten(), F.flatten(), exitflag

def jacobian(fun, x, nf, funx=None):
    """
    Estimate Jacobian matrix.

    Parameters:
    ----------
    fun : function handle
        Function that returns a vector of residuals equations and takes a vector, x, as its only argument.
    x : numpy array
        Vector at which to compute Jacobian.
    nf : int
        Number of functions.
    funx : numpy array, optional
        Function value at x.

    Returns:
    ----------
    J : numpy array
        Jacobian matrix.
    """
    dx = np.finfo(float).eps**(1/3)
    nx = len(x)
    if funx is None:
        funx = fun(x)
    J = np.zeros((nf, nx))
    for n in range(nx):
        delta = np.zeros(x.shape)
        # delta = np.zeros_like(x)
        delta[n,0] = dx
        dF = (fun(x + delta) - funx)
        J[:, n] = dF.flatten() / dx

    return J
