import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import optimize as opt
from scipy import interpolate
import core_utils as cu
from core_utils import get_cut_parameters, get_spindle_offsets, make_cam_file
from pathlib import Path


def lensfit(
    pathname,
    spindle,
    calibrationfilepath,
    metrologyfilename,
    lensparams,
    afixed,
    bfixed,
    stepheight,
    cutdiameter,
    x_rot_shift,
    plot=True
):
    """
    Fit a lens surface using least squares and return the best-fit parameters.

    Parameters
    ----------
    pathname : str
        Path to the directory containing metrology data.
    spindle : str
        Spindle name used for path construction and calibration lookup.
    calibrationfilepath : str
        Path to the spindle calibration file.
    metrologyfilename : str
        Name of the metrology data file (CSV).
    lensparams : tuple
        Parameters defining the lens surface.
    afixed, bfixed : float
        Fixed tilt parameters for the initial guess.
    stepheight : float
        Step height parameter used in plane fitting.
    cutdiameter : float
        Diameter of the desired lens cut.
    x_rot_shift : float
        Optional shift in X to be applied.
    plot : bool, default=True
        Whether to generate plots for data, residuals, and model.

    Returns
    -------
    p2 : ndarray
        Best-fit parameters for the lens surface.
    cov2 : ndarray
        Covariance of the fit.
    infodict2 : dict
        Diagnostic information from least squares.
    mesg2 : str
        Optimization status message.
    ier2 : int
        Integer flag indicating the reason for termination.
    """
    pts = np.loadtxt(pathname + metrologyfilename, delimiter=',')
    xin, yin, zin, r = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    qin = zin + r

    p0 = [83.255, 365.741, -40, afixed, bfixed]

    p, *_ = opt.leastsq(F, p0, args=(xin, yin, qin, afixed, bfixed, lensparams), ftol=1e-14, xtol=1e-14, diag=(1, 1, 1, 10, 10))
    p2, cov2, infodict2, mesg2, ier2 = opt.leastsq(
        FuncNew, p0, args=(xin, yin, qin, afixed, bfixed, lensparams),
        full_output=1, ftol=1e-14, xtol=1e-14, diag=(1, 1, 1, 10, 10))

    xouts, youts, zmodels, modelresids = [], [], [], []
    for xval, yval, qval in zip(xin, yin, qin):
        xout, yout, zmodel = Flrt(p, xval, yval, afixed, bfixed, lensparams)
        xouts.append(xout - xval + p[0])
        youts.append(yout - yval + p[1])
        modelresids.append(zmodel - qval)
    xouts, youts, zmodels, modelresids = map(np.array, (xouts, youts, zmodels, modelresids))

    if plot:
        fig = plt.figure(figsize=(12, 4))
        xgrid = np.linspace(xin.min(), xin.max(), 300)
        ygrid = np.linspace(yin.min(), yin.max(), 300)
        xoutgrid, youtgrid = np.meshgrid(xgrid, ygrid)
        qingrid = interpolate.griddata((xin, yin), qin, (xoutgrid, youtgrid), method='linear')
        residsgrid = interpolate.griddata((xin, yin), modelresids, (xoutgrid, youtgrid), method='linear')
        zmodelsgrid = interpolate.griddata((xin, yin), zmodels, (xoutgrid, youtgrid), method='linear')

        ax = fig.add_subplot(1, 3, 1, aspect='equal')
        ax.set_title('Data')
        cax = ax.contourf(xgrid, ygrid, qingrid, 40, cmap=cm.jet)
        fig.colorbar(cax)

        ax = fig.add_subplot(1, 3, 2, aspect='equal')
        ax.set_title('Residuals')
        cax = ax.contourf(xgrid, ygrid, residsgrid, 40, cmap=cm.jet)
        fig.colorbar(cax)

        ax = fig.add_subplot(1, 3, 3, aspect='equal')
        ax.set_title('Model Values')
        cax = ax.contourf(xgrid, ygrid, zmodelsgrid, 40, cmap=cm.jet)
        fig.colorbar(cax)

        plt.tight_layout()
        plt.savefig(pathname + 'Residuals.png', bbox_inches='tight')
        plt.show()

    return p2, cov2, infodict2, mesg2, ier2


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

    A = np.mat([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    B = np.mat([[np.cos(b), 0, -np.sin(b)], [0, 1, 0], [np.sin(b), 0, np.cos(b)]])

    pt = np.mat([[xmod], [ymod], [z1]])
    newpt = A @ B @ pt
    x2, y2, z2 = newpt

    dx1 = float(x2 - xmod)
    dy1 = float(y2 - ymod)
    xn1 = float(xmod - dx1)
    yn1 = float(ymod - dy1)
    rn1 = np.sqrt(xn1**2 + yn1**2)
    zn1 = Flens(rn1, lensparams) + z0

    ptn1 = np.mat([[xn1], [yn1], [zn1]])
    newptn1 = A @ B @ ptn1
    xn2, yn2, zn2 = newptn1

    dx = float(xn2 - xmod)
    dy = float(yn2 - ymod)
    xn3 = float(xn1 - dx)
    yn3 = float(yn1 - dx)
    rn3 = np.sqrt(xn3**2 + yn3**2)
    zn3 = Flens(rn3, lensparams) + z0

    ptn3 = np.mat([[xn3], [yn3], [zn3]])
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

    A = np.mat([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    B = np.mat([[np.cos(b), 0, -np.sin(b)], [0, 1, 0], [np.sin(b), 0, np.cos(b)]])

    pt = np.mat([[x - x0], [y - y0], [q]])  # Rotate then translate
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

###################################### Plane Fit for Flange ##########################################
def fit_flange(path, flangemetfile):
    """
    Perform a 3D plane fit on flange metrology data, correcting for angular tilt 
    in two directions (around X and Y axes), and return the negative rotation angles.

    Parameters
    ----------
    path : str
        Path to the directory containing the flange metrology file.
    flangemetfile : str
        Filename of the flange metrology .csv/.dat/.txt file.

    Returns
    -------
    tuple of float
        Negative rotation angles (a, b) in radians about the X and Y axes respectively.

    Notes
    -----
    - The metrology file must be a file with four columns: X, Y, Z, and gauge offset (r).
    - The fit minimizes Z + r using a rotated coordinate system.
    - Assumes small angles for a and b (on the order of 10^-4).
    - The model previously used a function F(x, y) that returned a constant 0.5 
      as a flat surface offset. This has been inlined directly into the return statement.
    """
    flangemetpath = path + flangemetfile
    pts = np.loadtxt(flangemetpath, delimiter=',')

    xin, yin, zin, r = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    qin = zin + r

    # Initial guess for parameters: [a, b, x0, y0, z0]
    p0 = [0, 0, 0, 0, 0]

    def planefit(p, x, y, q):
        """Model function returning rotated Z values for least squares fitting."""
        a, b, x0, y0, z0 = p
        A = np.array([[1, 0, 0],
                      [0, np.cos(a), -np.sin(a)],
                      [0, np.sin(a),  np.cos(a)]])
        B = np.array([[ np.cos(b), 0, -np.sin(b)],
                      [0,          1, 0],
                      [np.sin(b),  0, np.cos(b)]])

        xm, ym, qm = x - x0, y - y0, q - z0
        pts = np.vstack((xm, ym, qm))  # shape (3, N)
        rotated_pts = A @ B @ pts
        zp = rotated_pts[2, :]

        # Originally: outs[i] = zp + F(xp, yp) with F(x, y) = 0.5
        # Inlined as a constant surface offset:
        return zp + 0.5

    # Fit the model
    p, cov, infodict, mesg, ier = opt.leastsq(
        planefit,
        p0,
        args=(xin, yin, qin),
        full_output=1,
        ftol=1e-14,
        xtol=1e-14,
        diag=(100, 100, 1, 1, 1)
    )

    resids = infodict['fvec']

    print('Fitting...')
    print(f'Angles: a = {-p[0]:.6e}, b = {-p[1]:.6e}')
    print('Check that the angle values are of order 10^-4')

    # Plotting
    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.plot_trisurf(xin, yin, qin, cmap=cm.jet, linewidth=0.2)
    ax.set_title('Data')

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.plot_trisurf(xin, yin, resids, cmap=cm.jet, linewidth=0.2)
    ax.set_title('Residual Errors')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z residual (mm)')

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.plot_trisurf(xin, yin, r, cmap=cm.jet, linewidth=0.2)
    ax.set_title('Gauge Readout')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')

    plt.tight_layout()
    plt.show()

    return -p[0], -p[1]


def generate_lens_cut_files(
    p,
    pathname,
    spindle,
    calibrationfilepath,
    cutparamsfile,
    cutdiameter,
    bladeradius,
    cuttype,
    x_rot_shift=0.5
):
    """
    Generate cut cam files for a lens surface using fitted parameters.

    Parameters
    ----------
    p : array-like
        Best-fit parameters from lensfit.
    pathname : str
        Directory to output cut files.
    spindle : str
        Spindle identifier.
    calibrationfilepath : str
        Path to spindle calibration data.
    cutparamsfile : str
        Filename of cut parameters.
    cutdiameter : float
        Diameter of the lens to be cut.
    bladeradius : float
        Radius of the cutting blade.
    cuttype : str
        'Thick', 'Med', or 'Thin'
    x_rot_shift : float, default=0.5
        X-shift adjustment to center the cut.

    Returns
    -------
    None
    """
    thick_depth, med_depth, thin_depth, pitch = cu.get_cut_parameters(os.path.join(pathname, cutparamsfile))
    cutpath = os.path.join(pathname, spindle, f'CutCamming{cuttype}-Noshift')
    Path(cutpath).mkdir(parents=True, exist_ok=True)

    xcenter = p[0]
    ycenter = p[1]
    radius = cutdiameter / 2.0
    xstart = xcenter - radius + x_rot_shift + 0.5
    xend = xcenter + radius

    measdiam = 300
    xstartmeas = xcenter - measdiam/2.0 + 0.5 # matches original xstartmeas in sawpy lensfit 

    depth_dict = {'Thick': thick_depth, 'Med': thick_depth + med_depth, 'Thin': thick_depth + med_depth + thin_depth}
    depth = depth_dict[cuttype]

    newxoffset, newyoffset, newzoffset = cu.get_spindle_offsets(calibrationfilepath, spindle)
    offsets = (newxoffset, newyoffset, newzoffset)
    measrad = 0.5
    yres = 0.5

    def get_ys(xx):
        dx = xx - xcenter
        if abs(dx) > radius:
            return np.array([])
        dy = np.sqrt(radius ** 2 - dx ** 2)
        return np.arange(ycenter - dy, ycenter + dy + 0.0001, yres)

    def plane_func(x, y, a, b, c):
        return -a * x - b * y - c

    cu._write_cam_set(
        cutpath=cutpath,
        thickness_label=cuttype,
        p=p[:3],
        plane_func=plane_func,
        Xstart=xstart,
        Xend=xend,
        get_ys=get_ys,
        pitch=pitch,
        Yres=yres,
        offsets=offsets,
        bladeradius=bladeradius,
        depth=depth,
        measrad=measrad
    )

