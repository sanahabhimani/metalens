import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import optimize as opt
from scipy import interpolate
import core_utils as cu
from core_utils import get_cut_parameters, get_spindle_offsets, make_cam_file
from pathlib import Path
from datetime import datetime



def lensfit(
    pathname,
    metrologyfilename,
    lensparams,
    afixed,
    bfixed,
    plot=True,
    return_full=False,
    verbose=True
):
    """
    Perform lens surface fitting using two techniques: direct fit and rotated fit.
    Matches the SawPy fitting behavior, excluding all file writing and cutting logic.

    Parameters
    ----------
    pathname : str
        Path to the directory containing metrology data.
    metrologyfilename : str
        Name of the metrology data file (CSV).
    lensparams : list
        Lens surface parameters [R, K, A, B, C, D, Tctr, Diam].
    afixed, bfixed : float
        Fixed rotation values used in both fitting techniques.
    plot : bool, optional
        Whether to generate contour plots of data, residuals, and model surface.
    return_full : bool, optional
        Whether to return full least-squares diagnostic output from the first fit.
    verbose : bool, optional
        Whether to print detailed least-squares diagnostics to stdout.

    Returns
    -------
    If return_full is False:
        p : ndarray
            Best-fit parameters from the F-based model.
        p2 : ndarray
            Best-fit parameters from the FuncNew-based model.
    If return_full is True:
        p, p2, cov, infodict, mesg, ier
            Full leastsq diagnostic output from the F-based model in addition.
    """
    # 1. Load metrology data
    pts = np.loadtxt(pathname + metrologyfilename, delimiter=',')
    xin, yin, zin, r = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    qin = zin + r

    # 2. Initial guess [x0, y0, z0, afixed, bfixed]
    p0 = [83.255, 365.741, -40, afixed, bfixed]

    # 3. Fit 1: Standard least-squares fit (F)
    p, cov, infodict, mesg, ier = opt.leastsq(
        F, p0, args=(xin, yin, qin, afixed, bfixed, lensparams),
        full_output=1, ftol=1e-14, xtol=1e-14, diag=(1, 1, 1, 10, 10)
    )
    resids = infodict['fvec']

    # 4. Fit 2: Rotated least-squares fit (FuncNew)
    p2, cov2, infodict2, mesg2, ier2 = opt.leastsq(
        FuncNew, p0, args=(xin, yin, qin, afixed, bfixed, lensparams),
        full_output=1, ftol=1e-14, xtol=1e-14, diag=(1, 1, 1, 10, 10)
    )

    # 5. Verbose diagnostic printout
    if verbose:
        print("\nP1 (F):", p)
        print("P2 (FuncNew):", p2)
        print("Diff:", p2 - p)

        print("\n--- Least Squares Fit Diagnostics ---")
        print("ier (exit code):", ier)
        print("mesg (termination message):", mesg)
        print("infodict keys:", list(infodict.keys()))
        print("Sample values:")
        for k in ['ipvt', 'qtf']:
            if k in infodict:
                print(f"{k}: {infodict[k]}")
        print("--------------------------------------\n")

    # 6. Evaluate model surface using the first fit (F)
    xouts, youts, zmodels, modelresids = [], [], [], []
    for xval, yval, qval in zip(xin, yin, qin):
        xout, yout, zmodel = Flrt(p, xval, yval, afixed, bfixed, lensparams)
        xouts.append(xout - xval + p[0])
        youts.append(yout - yval + p[1])
        zmodels.append(zmodel)
        modelresids.append(zmodel - qval)

    xouts, youts, zmodels, modelresids = map(np.array, (xouts, youts, zmodels, modelresids))

    # 7. Plotting
    if plot:
        fig = plt.figure(figsize=(12, 4))
        xgrid = np.linspace(xin.min(), xin.max(), 300)
        ygrid = np.linspace(yin.min(), yin.max(), 300)
        xoutgrid, youtgrid = np.meshgrid(xgrid, ygrid)

        qingrid = interpolate.griddata((xin, yin), qin, (xoutgrid, youtgrid), method='linear')
        residsgrid = interpolate.griddata((xin, yin), resids, (xoutgrid, youtgrid), method='linear')
        zmodelsgrid = interpolate.griddata((xin, yin), zmodels, (xoutgrid, youtgrid), method='linear')

        ax = fig.add_subplot(1, 3, 1, aspect='equal')
        cax = ax.contourf(xgrid, ygrid, qingrid, 40, cmap=cm.jet)
        fig.colorbar(cax)
        plt.scatter(xin, yin, color='k')
        ax.set_title('Data')

        ax = fig.add_subplot(1, 3, 2, aspect='equal')
        cax = ax.contourf(xgrid, ygrid, residsgrid, 40, cmap=cm.jet)
        fig.colorbar(cax)
        ax.set_title('Residuals')

        ax = fig.add_subplot(1, 3, 3, aspect='equal')
        cax = ax.contourf(xgrid, ygrid, zmodelsgrid, 40, cmap=cm.jet)
        fig.colorbar(cax)
        ax.set_title('Model Values')

        plt.tight_layout()
        plt.savefig(pathname + 'Residuals.png', bbox_inches='tight')
        plt.show()

    # 8. Return
    if return_full:
        return p, p2, cov, infodict, mesg, ier
    else:
        return p, p2


def Flens(rin, lensparams):
    """
    Computes the height of a rotationally symmetric aspheric lens surface
    as a function of input radial distance, including correction for a
    measurement ball probe.

    The surface is modeled using the standard conic asphere equation with
    higher-order polynomial terms. A measurement correction is applied to
    account for the radius and angular effect of a 1 mm measurement ball
    on the effective radial input.

    Parameters
    ----------
    rin : float
        Input radial distance in mm from the center of the lens.
    lensparams : array_like
        List or tuple of 8 parameters:
            - R : float
                Radius of curvature (mm).
            - K : float
                Conic constant.
            - A, B, C, D : float
                Higher-order aspheric coefficients.
            - Tctr : float
                Center thickness or vertical offset (mm).
            - rmax : float
                Maximum usable radius of the lens (mm).

    Returns
    -------
    z : float
        Height of the lens surface (z-coordinate) at the corrected radial distance.
        Returns a large sentinel value (9999999) if `rin` exceeds allowed radius.
    """
    R = lensparams[0]
    K = lensparams[1]
    A = lensparams[2]
    B = lensparams[3]
    C = lensparams[4]
    D = lensparams[5]
    Tctr = lensparams[6]
    rmax = lensparams[7] / 2.0  # Half of lens diameter in mm

    r = rin

    # Measurement ball correction (probe has 1.0 mm diameter, radius 0.5 mm)
    dr = dF(rin, lensparams)
    theta = np.arctan(dr)
    r = r - 0.500 * np.sin(theta)

    dr2 = dF(r, lensparams)
    rr = r + 0.500 * np.sin(np.arctan(dr2))

    if rin > rmax + 500:
        return 9999999  # Sentinel value for out-of-bounds input

    # Asphere formula with conic section and polynomial terms
    z = (r**2 / R) / (1 + np.sqrt(1 - (1 + K) * (r / R)**2)) + A * r**2 + B * r**4 + C * r**6 + D * r**8

    return float(z + Tctr + 0.500 * np.cos(theta))


def Flrt(p, x, y, afixed, bfixed, lensparams):
    """
    Computes the transformed coordinates and height of a point on a rotated lens surface model.

    This function models the lens surface by adjusting for a center offset, applying two fixed
    rotational angles (a and b), and iteratively refining the position and height through two
    iterations of rotation and re-evaluation using the lens profile function `Flens`.

    Parameters
    ----------
    p : array_like
        Optimization parameters. Expected to be of the form [x0, y0, z0, a, b], where:
            - x0, y0 : float
                Translation offsets for the lens center.
            - z0 : float
                Vertical offset for the lens surface.
            - a, b : float
                Rotation angles about the y-axis and x-axis, respectively (will be overwritten by afixed and bfixed).
    x : float
        Measured x-coordinate.
    y : float
        Measured y-coordinate.
    afixed : float
        Fixed rotation angle around the y-axis (overrides `a` in `p`).
    bfixed : float
        Fixed rotation angle around the x-axis (overrides `b` in `p`).
    lensparams : dict or tuple
        Parameters required by the lens profile function `Flens`, which models surface height as a function of radius.

    Returns
    -------
    xn4 : float
        Final x-coordinate after two iterations of rotation correction.
    yn4 : float
        Final y-coordinate after two iterations of rotation correction.
    zn4 : float
        Final z-coordinate (height) after two iterations of rotation correction.
    """
    # p vector is [x0, y0, z0, a, b]
    x0 = p[0]
    y0 = p[1]
    z0 = p[2]
    a = afixed  # override input a
    b = bfixed  # override input b

    xmod = x - x0
    ymod = y - y0

    # First iteration
    r1 = np.sqrt(xmod**2 + ymod**2)
    z1 = Flens(r1, lensparams) + z0

    A = np.asmatrix([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    B = np.asmatrix([[np.cos(b), 0, -np.sin(b)], [0, 1, 0], [np.sin(b), 0, np.cos(b)]])

    pt = np.asmatrix([[xmod], [ymod], [z1]])
    newpt = A @ B @ pt
    x2, y2, z2 = newpt

    dx1 = float(x2 - xmod)
    dy1 = float(y2 - ymod)
    xn1 = float(xmod - dx1)
    yn1 = float(ymod - dy1)
    rn1 = np.sqrt(xn1**2 + yn1**2)
    zn1 = Flens(rn1, lensparams) + z0

    ptn1 = np.asmatrix([[xn1], [yn1], [zn1]])
    newptn1 = A @ B @ ptn1
    xn2, yn2, zn2 = newptn1

    dx = float(xn2 - xmod)
    dy = float(yn2 - ymod)
    xn3 = float(xn1 - dx)
    yn3 = float(yn1 - dx)
    rn3 = np.sqrt(xn3**2 + yn3**2)
    zn3 = Flens(rn3, lensparams) + z0

    ptn3 = np.asmatrix([[xn3], [yn3], [zn3]])
    newptn3 = A @ B @ ptn3

    xn4, yn4, zn4 = newptn3

    return float(xn4), float(yn4), float(zn4)


def F(p, x, y, q, afixed, bfixed, lensparams):
    """
    Vectorized residual function used for lens surface fitting.

    This function evaluates the residuals between measured data points and a modeled lens surface.
    It uses the `Flrt` function to compute model predictions based on the current parameters `p`
    and subtracts these predictions from the observed values `q`.

    Parameters
    ----------
    p : array_like
        Optimization parameters, where `p[0]` and `p[1]` are adjustments to the x and y positions.
    x : array_like
        x-coordinates of the measured data points.
    y : array_like
        y-coordinates of the measured data points.
    q : array_like
        Observed z-values (e.g., measured surface heights).
    afixed : float
        Fixed parameter representing a known component of the optical surface model in x-direction.
    bfixed : float
        Fixed parameter representing a known component of the optical surface model in y-direction.
    lensparams : dict or tuple
        Additional parameters required by the `Flrt` function to model the lens surface.

    Returns
    -------
    outs : ndarray
        Residuals between observed z-values and model predictions.
    """
    outs = np.zeros(len(x))
    xouts = np.zeros(len(x))
    youts = np.zeros(len(x))
    for i, val in enumerate(x):
        xout, yout, z = Flrt(p, x[i], y[i], afixed, bfixed, lensparams)
        xouts[i] = xout - x[i] + p[0]
        youts[i] = yout - y[i] + p[1]
        outs[i] = q[i] - z
    return outs


################ Rotate Data Method Functions Below, Originally Described as a "New" Comparison Method ##############
def Fnew(p, x, y, q, afixed, bfixed, lensparams):
    """
    Residual function for the "new" fitting method, where data points are rotated 
    into the model frame instead of rotating the model.

    This method applies inverse rotations (negative angles) to the data point 
    (x, y, q), transforming it into the coordinate frame of the unrotated lens model.
    The function returns the residual between the rotated data height and the 
    predicted model height at that radial distance.

    Parameters
    ----------
    p : array_like
        Optimization parameters. Expected to be of the form [x0, y0, z0, a, b], where:
            - x0, y0 : float
                Translation offsets to align data with the model center.
            - z0 : float
                Vertical offset for the lens surface.
            - a, b : float
                Rotation angles about the y-axis and x-axis (applied with a negative sign to rotate data into model frame).
    x : float
        Measured x-coordinate of the data point.
    y : float
        Measured y-coordinate of the data point.
    q : float
        Measured surface height (z-coordinate) of the data point.
    afixed : float
        Unused in this method but included for interface consistency.
    bfixed : float
        Unused in this method but included for interface consistency.
    lensparams : dict or tuple
        Parameters passed to the `Flens` function to define the lens profile.

    Returns
    -------
    residual : float
        Difference between the rotated data height and the predicted model height
        at the corresponding radial distance from the model center.
    """
    x0 = p[0]
    y0 = p[1]
    z0 = p[2]
    a = -p[3]
    b = -p[4]

    A = np.asmatrix([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    B = np.asmatrix([[np.cos(b), 0, -np.sin(b)], [0, 1, 0], [np.sin(b), 0, np.cos(b)]])

    pt = np.asmatrix([[x - x0], [y - y0], [q]])  # Rotate then translate
    newpt = A * B * pt
    xr = float(newpt[0])
    yr = float(newpt[1])
    zr = float(newpt[2])

    r = np.sqrt(xr**2 + yr**2)
    zmod = Flens(r, lensparams) + z0

    return zr - zmod


def FuncNew(p, x, y, q, afixed, bfixed, lensparams):
    """
    Vectorized residual function for the "new" fitting method, where data points
    are rotated into the lens model frame before evaluating surface height residuals.

    This function applies the `Fnew` routine to a set of (x, y, q) measurements, computing
    the difference between each rotated data point's height and the corresponding model
    prediction from the aspheric lens surface.

    Parameters
    ----------
    p : array_like
        Optimization parameters. Expected to be of the form [x0, y0, z0, a, b], where:
            - x0, y0 : float
                Translation offsets.
            - z0 : float
                Vertical offset.
            - a, b : float
                Rotation angles (in radians) about the y- and x-axes, respectively.
    x : array_like
        x-coordinates of the measured data points.
    y : array_like
        y-coordinates of the measured data points.
    q : array_like
        z-values (measured heights) of the data points.
    afixed : float
        Unused in this method but included for interface consistency.
    bfixed : float
        Unused in this method but included for interface consistency.
    lensparams : dict or tuple
        Parameters passed to the `Flens` function to define the lens profile.

    Returns
    -------
    outs : ndarray
        Array of residuals between each rotated data height and model-predicted height.
    """
    outs = np.zeros(len(x))
    for i, val in enumerate(x):
        outs[i] = Fnew(p, x[i], y[i], q[i], afixed, bfixed, lensparams)
    return outs


def FlensNoball(rin, lensparams):
    """
    Compute the sag of a rotationally symmetric aspheric lens surface (excluding ball lens shape)
    at a given radial position from the center.

    Parameters
    ----------
    rin : float
        Radial distance from the center of the lens, in millimeters.
    lensparams : array-like
        Lens parameters in the following order:
            [0] R     : Radius of curvature (mm)
            [1] K     : Conic constant
            [2] A     : 2nd order aspheric coefficient (mm^-1)
            [3] B     : 4th order aspheric coefficient (mm^-3)
            [4] C     : 6th order aspheric coefficient (mm^-5)
            [5] D     : 8th order aspheric coefficient (mm^-7)
            [6] Tctr  : Center thickness of the lens (mm)
            [7] Diam  : Lens diameter (mm), used to compute max usable radius

    Returns
    -------
    float
        Sag (surface height) at radial position `rin` in mm.

    Notes
    -----
    - If `rin` exceeds `rmax + 500`, the function returns a large float (5,000,000) 
      to indicate invalid input or safeguard against unphysical values.
    - The surface equation follows the standard aspheric surface formula:
        z(r) = (r^2 / R) / (1 + sqrt(1 - (1 + K)(r/R)^2)) + A*r^2 + B*r^4 + C*r^6 + D*r^8 + Tctr
    """
    R, K, A, B, C, D, Tctr, Diam = lensparams
    rmax = Diam / 2.0  # Convert diameter to radius in mm
    r = rin

    if rin > rmax + 500:
        return 5_000_000  # Safety return for unphysically large inputs

    base = (r**2 / R) / (1 + np.sqrt(1 - (1 + K) * (r / R)**2))
    poly = A * r**2 + B * r**4 + C * r**6 + D * r**8
    z = base + poly + Tctr

    return float(z)


def dF(rin, lensparams):
    """
    Compute the numerical derivative of the lens sag function with respect to radial distance.

    This function approximates dF/dr using central differencing by evaluating the lens profile
    at `rin - h` and `rin + h`. If `rin` exceeds `rmax + 500`, the function returns 0.

    Parameters
    ----------
    rin : float
        Radial distance from the center of the lens, in millimeters.
    lensparams : array-like
        Lens parameters in the following order:
            [0] R     : Radius of curvature
            [1] K     : Conic constant
            [2] A     : 4th order aspheric coefficient
            [3] B     : 6th order aspheric coefficient
            [4] C     : 8th order aspheric coefficient
            [5] D     : 10th order aspheric coefficient
            [6] Tctr  : Center thickness (unused)
            [7] Diam  : Maximum diameter of the lens (used to compute rmax)

    Returns
    -------
    float
        Approximate derivative dF/dr at `rin`.

    Notes
    -----
    - A closed-form `dz` formula is defined but then immediately replaced by a numerical estimate.
    - The lens profile is evaluated using an external function `FlensNoball`, which must be defined elsewhere.
    - Assumes `rin` and all parameters are in millimeters.

    """
    R, K, A, B, C, D, Tctr, Diam = lensparams
    rmax = Diam / 2.0  # Diameter to radius

    r = float(rin)

    if rin > rmax + 500:
        return 0  # Safeguard against invalid large radius inputs

    # NOTE: Analytical expression is calculated but not used
    # dz = r / (np.sqrt(R**2 - (K + 1) * r**2)) + 4*A*r**3 + 6*B*r**5 + 8*C*r**7 + 8*D*r**7

    # Use central difference to compute derivative numerically
    h = 0.0001  # Small step size in mm
    Fl = FlensNoball(rin - h, lensparams)
    Fr = FlensNoball(rin + h, lensparams)
    dz = -(Fr - Fl) / (2.0 * h)

    return dz


###################################### Plane Helper Functions ##########################################
# TODO: Implement all use cases of stepheight/plane helper functions here for correct cut cam file generation
def PlaneNoball(p, x, y, lensparams, stepheight):
    """
    Compute the (x, y, z) position on a lens surface (without a ball term),
    applying two iterations of rotation and center offset correction.

    This function uses a geometric model of a lens surface defined by aspheric
    parameters and simulates the effect of decentering and tilting in both
    X and Y directions. The rotation is applied iteratively to improve accuracy.

    Parameters
    ----------
    p : list or array-like
        Parameter vector [x0, y0, z0, a, b], where:
            x0, y0 : center offsets in mm
            z0     : vertical shift (height offset)
            a      : tilt about the X-axis (in radians)
            b      : tilt about the Y-axis (in radians)
    x : float
        X-coordinate of the point (in mm) to evaluate.
    y : float
        Y-coordinate of the point (in mm) to evaluate.
    lensparams : list
        Lens surface parameters in the form:
            [R, K, A, B, C, D, Tctr, Diam]
            R     : Radius of curvature (1/c)
            K     : Conic constant
            A–D   : Aspheric coefficients
            Tctr  : Center thickness
            Diam  : Full lens diameter
    stepheight : float
        Step height value to subtract from the reference surface (in mm).

    Returns
    -------
    tuple of float
        The rotated and offset-corrected coordinates (x_out, y_out, z_out) in mm.

    Notes
    -----
    - The function evaluates the sag of the lens at its outermost radius (Diam/2),
      then subtracts the step height and adds 0.100 mm as an empirical adjustment.
    - Two full iterations of rotational correction are applied.
    - Your system uses a coordinate convention where x and y roles are reversed
      compared to standard mathematical notation; this is respected as-is.
    """
    zout = FlensNoball(lensparams[7] / 2.0, lensparams) - stepheight + 0.100

    x0, y0, z0, a, b = p
    xmod = x - x0
    ymod = y - y0

    r1 = np.sqrt(xmod**2 + ymod**2)
    z1 = zout + z0

    # Rotation matrices (as np.array)
    A = np.array([
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a),  np.cos(a)]
    ])
    B = np.array([
        [np.cos(b), 0, -np.sin(b)],
        [0,         1, 0],
        [np.sin(b), 0,  np.cos(b)]
    ])

    pt = np.array([[xmod], [ymod], [z1]])
    newpt = A @ B @ pt
    x2, y2, z2 = newpt.flatten()

    dx1 = x2 - xmod
    dy1 = y2 - ymod
    xn1 = xmod - dx1
    yn1 = ymod - dy1
    rn1 = np.sqrt(xn1**2 + yn1**2)
    zn1 = zout + z0

    ptn1 = np.array([[xn1], [yn1], [zn1]])
    newptn1 = A @ B @ ptn1
    xn2, yn2, zn2 = newptn1.flatten()

    dx = xn2 - xmod
    dy = yn2 - ymod
    xn3 = xn1 - dx
    yn3 = yn1 - dx  # Your coordinate convention: intentionally reusing dx here
    rn3 = np.sqrt(xn3**2 + yn3**2)
    zn3 = zout + z0

    ptn3 = np.array([[xn3], [yn3], [zn3]])
    newptn3 = A @ B @ ptn3
    xn4, yn4, zn4 = newptn3.flatten()

    return float(xn4), float(yn4), float(zn4)

def generate_lens_cut_files(
    p,
    pathname,
    spindle,
    calibrationfilepath,
    cutparamsfile,
    cutdiameter,
    bladeradius,
    cuttype,
    lensparams,
    stepheight,
    x_rot_shift=0.0,
    yres=0.500,
    dzdy_h=0.01,
):
    """
    Generate lens cut cam files (SawPy-faithful): surface-normal compensation + plane-guard using stepheight.

    This reproduces the SawPy camming logic:
      1) Compute surface-based youts,zs using grad-like normal factors
      2) Compute plane-only youtsplane,zsplane using PlaneNoball(stepheight)
      3) Replace zs with zsplane when zsplane > zs (less aggressive cut)

    Parameters
    ----------
    p : array-like
        Fit vector from lensfit: [x0, y0, z0, a, b]
    pathname : str
    spindle : str
    calibrationfilepath : str
    cutparamsfile : str
    cutdiameter : float
    bladeradius : float
    cuttype : str
        'Thick', 'Med', or 'Thin'
    lensparams : list
        [R, K, A, B, C, D, Tctr, Diam]
    stepheight : float
        Same step_height you used in SawPy (e.g., 7.046)
    x_rot_shift : float
        Same meaning as SawPy (shift for 180/270 cases; can be 0)
    yres : float
        Y resolution (mm), default 0.5 like SawPy
    dzdy_h : float
        Step size (mm) for numerical dz/dy used to build normal factors

    Returns
    -------
    str
        cutpath directory written
    """
    import os
    import numpy as np
    from pathlib import Path

    # ---- directories ----
    cutpath = os.path.join(pathname, spindle, f"CutCamming{cuttype}-Noshift")
    Path(cutpath).mkdir(parents=True, exist_ok=True)

    # ---- cut params ----
    thick_depth, med_depth, thin_depth, pitch = cu.get_cut_parameters(os.path.join(pathname, cutparamsfile))

    if cuttype == "Thick":
        depth = thick_depth
    elif cuttype == "Med":
        depth = thick_depth + med_depth
    elif cuttype == "Thin":
        depth = thick_depth + med_depth + thin_depth
    else:
        raise ValueError(f"cuttype must be 'Thick', 'Med', or 'Thin' (got {cuttype})")

    # ---- fitted center + cut geometry ----
    xcenter = float(p[0])
    ycenter = float(p[1])
    radius = float(cutdiameter) / 2.0

    # SawPy:
    # Xstart = Xcenter - Cutdiam/2 + x_rot_shift + 0.5
    # Xend   = Xcenter + Cutdiam/2
    xstart = xcenter - radius + x_rot_shift + 0.5
    xend = xcenter + radius

    # ---- spindle offsets ----
    Xoffset, Yoffset, Zoffset = cu.get_spindle_offsets(calibrationfilepath, spindle)

    # -------------------------------------------------------------------------
    # Local helpers: FlrtNoball + PlaneNoball that accept the 5-vector p
    # -------------------------------------------------------------------------
    def FlrtNoball_local(pfit, x, y, lensparams):
        x0, y0, z0, a, b = map(float, pfit)

        xmod = x - x0
        ymod = y - y0

        A = np.asmatrix([[1, 0, 0],
                    [0, np.cos(a), -np.sin(a)],
                    [0, np.sin(a),  np.cos(a)]])
        B = np.asmatrix([[ np.cos(b), 0, -np.sin(b)],
                    [0,          1, 0],
                    [ np.sin(b), 0,  np.cos(b)]])

        r1 = np.sqrt(xmod**2 + ymod**2)
        z1 = FlensNoball(r1, lensparams) + z0

        pt1 = np.asmatrix([[xmod], [ymod], [z1]])
        newpt1 = A @ B @ pt1
        x2, y2, z2 = newpt1

        dx1 = float(x2 - xmod)
        dy1 = float(y2 - ymod)

        xn1 = float(xmod - dx1)
        yn1 = float(ymod - dy1)
        rn1 = np.sqrt(xn1**2 + yn1**2)
        zn1 = FlensNoball(rn1, lensparams) + z0

        ptn1 = np.asmatrix([[xn1], [yn1], [zn1]])
        newptn1 = A @ B @ ptn1
        xn2, yn2, zn2 = newptn1

        dx2 = float(xn2 - xmod)
        dy2 = float(yn2 - ymod)

        xn3 = float(xn1 - dx2)
        yn3 = float(yn1 - dy2)   # IMPORTANT: dy2
        rn3 = np.sqrt(xn3**2 + yn3**2)
        zn3 = FlensNoball(rn3, lensparams) + z0

        ptn3 = np.asmatrix([[xn3], [yn3], [zn3]])
        newptn3 = A @ B @ ptn3
        xn4, yn4, zn4 = newptn3

        return float(xn4), float(yn4), float(zn4)

    def PlaneNoball_local(pfit, x, y, lensparams, stepheight):
        """
        Plane-only model used as a SawPy guard.
        Matches the spirit of SawPy PlaneNoball(): linear plane through the fit + a stepheight-based offset.
        """
        x0, y0, z0, a, b = map(float, pfit)
        R, K, A4, B6, C8, D10, Tctr, Diam = lensparams

        # SawPy PlaneNoball uses:
        # zout = FlensNoball(Diam/2) - stepheight + 0.100
        zedge = FlensNoball(float(Diam) / 2.0, lensparams)
        zout = zedge - float(stepheight) + 0.100

        # Plane with slopes a,b around center (x0,y0), anchored at (x0,y0)->(z0 + zout)
        # sign convention: consistent with using a,b as small rotations in Flrt
        z = (z0 + zout) - a * (x - x0) - b * (y - y0)
        return float(x), float(y), float(z)

    # Numerical dz/dy for normal factors (surface and plane)
    def dzdy_surface(pfit, x, y):
        h = float(dzdy_h)
        zp = FlrtNoball_local(pfit, x, y + h, lensparams)[2]
        zm = FlrtNoball_local(pfit, x, y - h, lensparams)[2]
        return (zp - zm) / (2.0 * h)

    def dzdy_plane(pfit, x, y):
        h = float(dzdy_h)
        zp = PlaneNoball_local(pfit, x, y + h, lensparams, stepheight)[2]
        zm = PlaneNoball_local(pfit, x, y - h, lensparams, stepheight)[2]
        return (zp - zm) / (2.0 * h)

    # y-bounds for a given x on the cut circle
    def y_bounds_for_x(xx):
        dx = xx - xcenter
        if abs(dx) > radius:
            return None
        dy = np.sqrt(radius**2 - dx**2)
        return (ycenter - dy, ycenter + dy + 0.0001)

    # -------------------------------------------------------------------------
    # Write cam set (SawPy-like)
    # -------------------------------------------------------------------------
    xs = np.arange(xstart, xend, pitch)
    master_lines = []

    for j, xx in enumerate(xs):
        bounds = y_bounds_for_x(xx)
        if bounds is None:
            continue
        ystart, yend = bounds
        ys = np.arange(ystart, yend, yres)
        if ys.size < 2:
            continue

        youts = np.zeros_like(ys, dtype=float)
        zs = np.zeros_like(ys, dtype=float)

        for i, yy in enumerate(ys):
            # ---------------- surface model ----------------
            zsurf = FlrtNoball_local(p, xx, yy, lensparams)[2]

            dzdy_s = dzdy_surface(p, xx, yy)
            Fy = -dzdy_s
            Fz = 1.0
            den = np.sqrt(Fy**2 + Fz**2)
            Fyy = Fy / den
            Fzz = Fz / den

            youts[i] = yy - bladeradius * Fyy
            z_cmd_surface = zsurf - Zoffset + bladeradius * Fzz - depth

            # ---------------- plane guard (SawPy stepheight) ----------------
            zpl = PlaneNoball_local(p, xx, youts[i], lensparams, stepheight)[2]

            dzdy_p = dzdy_plane(p, xx, youts[i])
            Fy2 = -dzdy_p
            Fz2 = 1.0
            den2 = np.sqrt(Fy2**2 + Fz2**2)
            Fzz2 = Fz2 / den2

            z_cmd_plane = zpl - Zoffset + bladeradius * Fzz2 - depth

            # SawPy behavior: take the higher Z (less aggressive cut)
            zs[i] = z_cmd_surface
            if z_cmd_plane > zs[i]:
                zs[i] = z_cmd_plane
                # SawPy sometimes kept youts[i] as-is (commented out swapping).
                # If you want exact SawPy variants, we can toggle this line:
                # youts[i] = youts[i]  # leave unchanged

        fname_prefix = os.path.join(cutpath, f"CutCam{cuttype}")
        cu.make_cam_file(fname_prefix, j, xx + Xoffset, youts + Yoffset, zs)

        master_lines.append(
            f"{j:04d} {xx+Xoffset} {youts[0]+Yoffset} {zs[0]} {youts[-1]+Yoffset}\n"
        )

    with open(os.path.join(cutpath, "Master.txt"), "w") as f:
        f.writelines(master_lines)

    return cutpath


def shiftZ_silicon(directory, spindle, ftype, zshift,
                   lastlinecut, firstline, numlines):
    """
    Applies Z shift corrections to CAM files by copying them from the 
    `CutCamming{ftype}-Final` folder into a fresh `CutCamming{ftype}` 
    folder, modifying only the specified line range. Generates a new 
    Master.txt and writes a unique log file under .../{spindle}/Logs/.

    Args:
        directory: str
            Base directory containing spindle subfolders.
        spindle: str
            Identifier for the spindle (e.g., "SpindleC").
        ftype: str
            CAM file type (e.g., "Thin", "Thick").
        zshift: float
            Amount to shift Z values (applied only to lines in the target range).
        lastlinecut: int
            Index of the last line cut in the previous run (used for backup naming).
        firstline: int
            First line number in the range of lines to shift.
        numlines: int
            Number of lines to process starting from `firstline`.

    Returns:
        None
            The function modifies files on disk and writes a log file. 
    """

    # ---- Paths (source vs destination) ----
    subfolder_src = f'CutCamming{ftype}-Final'   # source (originals live here)
    subfolder_out = f'CutCamming{ftype}'           # destination (new output)

    camname = f"{directory}{spindle}/{subfolder_out}/CutCam{ftype}"   # destination base
    cin     = f"{directory}{spindle}/{subfolder_src}/CutCam{ftype}"   # source base

    masterfilein  = f"{directory}{spindle}/{subfolder_src}/Master.txt"   # read from -Shifted
    masterfiledir = f"{directory}{spindle}/{subfolder_out}/"             # write to new folder
    masterfileout = f"{masterfiledir}/Master.txt"

    # ---- Logging setup (unique file per run) ----
    logs_dir = f"{directory}{spindle}/Logs"
    os.makedirs(logs_dir, exist_ok=True)
    tstamp = datetime.now().strftime("%Y%m%d_%H%M")
    line_start = int(firstline)
    line_end   = int(firstline + numlines - 1)
    logfile = (
        f"{logs_dir}/ShiftZ_{ftype}_{line_start:04d}to{line_end:04d}.log")

    lockfile = f"{directory}{spindle}/{subfolder_out}/lockfile.lock"
    print("Lockfile present" if os.path.isfile(lockfile) else "Lockfile not present")

    # ---- Load Master from source ----
    mfile = np.loadtxt(masterfilein)
    if mfile.ndim == 1:
        mfile = mfile.reshape(1, -1)
    nums = mfile[:, 0].astype(int)
    xs, ys, zs, ystops = mfile[:, 1], mfile[:, 2], mfile[:, 3], mfile[:, 4]

    # ---- Back up existing destination folder if present ----
    if os.path.isdir(masterfiledir):
        os.rename(masterfiledir, f"{masterfiledir.rstrip('/')}_UpToLine_{lastlinecut}")
    os.makedirs(masterfiledir, exist_ok=True)

    # ---- Decide which lines to modify ----
    mask = (nums >= line_start) & (nums <= line_end)

    # ---- Prepare Z output: apply zshift only within the window ----
    zsout = zs.copy()
    zsout[mask] = zsout[mask] + zshift

    # ---- Update CAM files (from -final) and write new Master.txt ----
    updated_lines = []  # for logging
    with open(masterfileout, 'w') as mfileout:
        for i, num in enumerate(nums):
            linenum = f"{int(num):04d}"
            src_cam = f"{cin}{linenum}.Cam"                    # read source CAM
            dst_base = f"{directory}{spindle}/{subfolder_out}/CutCam{ftype}"  # write dest CAM

            pts = np.loadtxt(src_cam, skiprows=4)
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            yscam = pts[:, 1]
            if mask[i]:
                zscam = pts[:, 2] + zshift                      # only apply zshift
                updated_lines.append((int(num), src_cam, f"{dst_base}{linenum}.Cam", zshift))
            else:
                zscam = pts[:, 2]                               # no shift for untouched lines
                updated_lines.append((int(num), src_cam, f"{dst_base}{linenum}.Cam", 0.0))

            cu.make_cam_file(dst_base, num, xs[i], yscam, zscam)

            # Always write Master; Z updated only where mask=True
            mfileout.write(f"{linenum} {xs[i]} {ys[i]} {zsout[i]} {ystops[i]}\n")

    # ---- Trim processed lines for downstream workflow ----
    cu.remove_lines(masterfiledir, firstline, firstline + numlines - 1, ftype)

    # ---- Write log (unique, non-overwriting) ----
    with open(logfile, "w") as lf:
        lf.write("=== shiftZ_silicon run log ===\n")
        lf.write(f"Timestamp (Date--HHMM)     : {tstamp}\n")
        lf.write(f"Spindle                    : {spindle}\n")
        lf.write(f"Type (ftype)               : {ftype}\n")
        lf.write(f"Source folder              : {directory}{spindle}/{subfolder_src}/\n")
        lf.write(f"Dest folder                : {masterfiledir}\n")
        lf.write(f"Master source              : {masterfilein}\n")
        lf.write(f"Master dest                : {masterfileout}\n")
        lf.write(f"Last line cut              : {lastlinecut}\n")
        lf.write(f"Applied zshift             : {zshift:+.6f}\n")

    print(f"Log written: {logfile}")
