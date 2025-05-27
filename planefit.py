import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import optimize as opt
import core_utils as cu


### Helper Functions ###
def fourier_fit(pos, z, n_max):
    """
    Compute a 2D Fourier fit of the residuals across the measurement surface.

    Parameters
    ----------
    pos : ndarray of shape (N, 2)
        Positions of measurement points as [[x1, y1], [x2, y2], ..., [xN, yN]].
    z : ndarray of shape (N,)
        Residuals or values to be fit.
    n_max : int
        Maximum spatial frequency in each direction. The number of modes used is (2 * n_max + 1)^2.

    Returns
    -------
    A : ndarray of complex, shape ((2 * n_max + 1)^2,)
        Complex Fourier coefficients for each mode.
    """
    i_max = (2 * n_max + 1)**2
    k_max = len(pos)

    F = np.zeros((i_max, k_max), dtype=complex)

    for i in range(i_max):
        m = int(i % (2 * n_max + 1)) - n_max
        n = int(i / (2 * n_max + 1)) - n_max

        for k in range(k_max):
            x, y = pos[k]
            F[i, k] = np.exp(1j * (m * x / 300. * 2 * np.pi + n * y / 300. * 2 * np.pi))

    M = np.linalg.pinv(F)
    A = np.dot(z, M)

    return A


def fourier_eval(A_coef, x, y, n_max):
    """
    Evaluate a 2D complex Fourier series at a single point (x, y).

    Parameters
    ----------
    A_coef : ndarray of complex
        Fourier coefficients obtained from fourier_fit(), length must be (2 * n_max + 1)^2.
    x : float
        X-coordinate of the point to evaluate.
    y : float
        Y-coordinate of the point to evaluate.
    n_max : int
        Maximum spatial frequency used in the fit.

    Returns
    -------
    float
        Real part of the Fourier sum evaluated at (x, y).
    """
    i_max = (2 * n_max + 1)**2
    temp = 0 + 0j

    for i in range(i_max):
        m = int(i % (2 * n_max + 1)) - n_max
        n = int(i / (2 * n_max + 1)) - n_max

        temp += A_coef[i] * np.exp(1j * (m * x / 300. * 2 * np.pi + n * y / 300. * 2 * np.pi))

    return np.real(temp)


### Plane Fitting Functions Relevant to Either Test Wafer, Dressing Board, Lens, Alumina Filter ###
def planefit(filepath, do_plot=True):
    """
    Perform plane fitting and Fourier residual correction on lens or alumina metrology data.

    Parameters
    ----------
    filepath : str
        Path to the metrology .dat file with columns [x, y, z, r].
    do_plot : bool, optional
        Whether to generate 2D contour plots of the results.

    Returns
    -------
    p : array
        Plane fit parameters [a, b, c].
    corrections : array
        Fourier-based correction values at each data point.
    zmodel : array
        Final model surface: plane + correction.
    residuals : array
        Original plane residuals (qin - F).
    corrected_residuals : array
        Residuals after Fourier correction (residuals - corrections).
    """
    # Config
    fourier_max = 2
    correction_max = 0.070

    # Load data
    pts = np.loadtxt(filepath, delimiter=',')
    xin, yin, zin, r = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    qin = zin + r

    def F(x, y, a, b, c):
        return -a * x - b * y - c

    def residuals_func(params, x, y, q):
        a, b, c = params
        return q - F(x, y, a, b, c)

    # Fit plane
    p0 = [0, 0, 0]
    p, cov, infodict, mesg, ier = opt.leastsq(
        residuals_func, p0,
        args=(xin, yin, qin),
        full_output=1, ftol=1e-14, xtol=1e-14
    )

    print("Plane fit parameters (a, b, c):", p)
    print("Leastsq message:", mesg, "| ier =", ier)

    residuals = qin - F(xin, yin, *p)
    pos = np.column_stack((xin, yin))
    A_coef = fourier_fit(pos, residuals, fourier_max)

    corrections = np.array([
        np.clip(fourier_eval(A_coef, x, y, fourier_max), -correction_max, correction_max)
        for x, y in zip(xin, yin)
    ])
    corrected_residuals = residuals - corrections
    zmodel = F(xin, yin, *p) + corrections

    if do_plot:
        # Grid for smooth contour plotting
        xgrid = np.linspace(xin.min(), xin.max(), 300)
        ygrid = np.linspace(yin.min(), yin.max(), 300)
        xoutgrid, youtgrid = np.meshgrid(xgrid, ygrid)

        qingrid   = interpolate.griddata((xin, yin), qin,   (xoutgrid, youtgrid), method="linear")
        resgrid   = interpolate.griddata((xin, yin), residuals, (xoutgrid, youtgrid), method="linear")
        corrgrid  = interpolate.griddata((xin, yin), corrected_residuals, (xoutgrid, youtgrid), method="linear")
        zmodgrid  = interpolate.griddata((xin, yin), zmodel, (xoutgrid, youtgrid), method="linear")

        # Plot
        fig = plt.figure(figsize=(18, 4), dpi=150)
        fig.suptitle("Plane Fit + Fourier Correction", fontsize=16)

        def add_contour(ax, data, title):
            cax = ax.contourf(xgrid, ygrid, data, 40, cmap=cm.jet)
            cbar = fig.colorbar(cax, ax=ax)
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.4f}'))
            cbar.update_ticks()
            ax.set_aspect('equal')
            ax.set_title(title)
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")

        add_contour(fig.add_subplot(1, 4, 1), qingrid, "Data")
        add_contour(fig.add_subplot(1, 4, 2), resgrid, "Plane Residuals")
        add_contour(fig.add_subplot(1, 4, 3), corrgrid, "Correction Residuals")
        add_contour(fig.add_subplot(1, 4, 4), zmodgrid, "Model Surface")

        plt.tight_layout()
        plt.show()

    return p, corrections, zmodel, residuals, corrected_residuals


def quickfit_plane(filepath, do_plot=True):
    """
    Load metrology data from 'filepath', perform a simple least-squares plane fit,
    and generate a 4-panel 3D visualization of the data, residuals, gauge values,
    and model surface.

    This is intended for quick plane fitting  on dressing boards and test wafers.
    It is not as involved as plane fitting for lenses and alumina filters, which
    require a fourier fit and fourier eval function.

    Parameters
    ----------
    filepath : str
        Full path to the metrology .dat file containing columns [x, y, z, r].
    do_plot : bool, optional
        If True, generate 3D surface plots for visual inspection.

    Returns
    -------
    p : array
        Best-fit plane parameters [a, b, c] such that z = -a*x - b*y - c.
    cov : array
        Covariance matrix from the fit.
    infodict : dict
        Dictionary with information returned by scipy.optimize.leastsq.
    mesg : str
        Message describing the exit status of the fit.
    ier : int
        Integer flag indicating success (1–4) or failure (0 or 5).
    """
    # Load data
    pts = np.loadtxt(filepath, delimiter=",")
    xin = pts[:, 0]
    yin = pts[:, 1]
    zin = pts[:, 2]
    r   = pts[:, 3]

    # Combined surface height from gauge offset
    qin = zin + r

    def F(x, y, a, b, c):
        return -a * x - b * y - c

    def planefit_residuals(params, x, y, q):
        a, b, c = params
        return q - F(x, y, a, b, c)

    # Initial guess
    p0 = [0.0, 0.0, 0.0]

    # Least-squares fit
    p, cov, infodict, mesg, ier = opt.leastsq(
        planefit_residuals,
        p0,
        args=(xin, yin, qin),
        full_output=1,
        ftol=1e-14,
        xtol=1e-14
    )

    print("Plane fit parameters (a, b, c):", p)
    print("Leastsq message:", mesg, "| ier =", ier)

    if do_plot:
        residuals = qin - F(xin, yin, *p)
        zmodel = F(xin, yin, *p)

        fig = plt.figure(figsize=(16, 4), dpi=120)

        # Panel 1: Data
        ax = fig.add_subplot(1, 4, 1, projection="3d")
        ax.plot_trisurf(xin, yin, qin, cmap=cm.jet, linewidth=0.2)
        ax.set_title("Data")

        # Panel 2: Residuals
        ax = fig.add_subplot(1, 4, 2, projection="3d")
        ax.plot_trisurf(xin, yin, residuals, cmap=cm.jet, linewidth=0.2)
        ax.set_title("Residuals")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z residual (mm)")

        # Panel 3: Gauge Readout
        ax = fig.add_subplot(1, 4, 3, projection="3d")
        ax.plot_trisurf(xin, yin, r, cmap=cm.jet, linewidth=0.2)
        ax.set_title("Gauge (r)")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")

        # Panel 4: Model Plane
        ax = fig.add_subplot(1, 4, 4, projection="3d")
        ax.plot_trisurf(xin, yin, zmodel, cmap=cm.jet, linewidth=0.2)
        ax.set_title("Model")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")

        plt.tight_layout()
        plt.show()

    return p, cov, infodict, mesg, ier


def generate_planar_cut_files(
    p,
    pathname,
    spindle,
    calibrationfilepath,
    cutparamsfile,
    bladeradius,
    cuttype,
    xstart,
    xend,
    geometry,  # 'circular' or 'rectangular'
    ystart=None,
    yend=None,
    cutdiameter=None,
    ycenter=None,
    use_noshift_suffix=True
):
    """
    Generate cut-camming files for a planar surface, supporting both circular and rectangular geometries.

    Parameters
    ----------
    p : array-like of float
        Plane coefficients [a, b, c] from the plane fitting function.

    pathname : str
        Base directory where output folders and files are stored.

    spindle : str
        Spindle identifier (used in file path construction).

    calibrationfilepath : str
        Path to the spindle calibration file (used to retrieve X, Y, Z offsets).

    cutparamsfile : str
        File containing cut parameters such as thick/med/thin depth and pitch.

    bladeradius : float
        Radius of the cutting blade.

    cuttype : str
        Which cut type to generate: 'Thick', 'Thin', or 'Med'.

    xstart : float
        Starting X position for the cut region.

    xend : float
        Ending X position for the cut region.

    geometry : str
        Geometry of the region to be cut. Must be either 'circular' or 'rectangular'.

    ystart : float, optional
        Starting Y position for rectangular cuts (only used if geometry='rectangular'). Default is None.

    yend : float, optional
        Ending Y position for rectangular cuts (only used if geometry='rectangular'). Default is None.

    cutdiameter : float, optional
        Diameter of the circular region (only used if geometry='circular'). Default is None.

    ycenter : float, optional
        Y center of the circular region (only used if geometry='circular'). Default is None.

    use_noshift_suffix : bool, default=True
        If True, appends '-Noshift' to cut directory names (applies to all current use cases). Default is True.

    Notes
    -----
    - For **circular geometry**, provide: `cutdiameter`, `ycenter`
    - For **rectangular geometry**, provide: `ystart`, `yend`

    Returns
    -------
    None or str
        Returns 'Lockfile present' if a lock file exists and cut generation is skipped. Otherwise, writes cam files and returns None.
    """
    # Construct cut paths
    suffix = '-Noshift' if use_noshift_suffix else ''
    cutpaththick = os.path.join(pathname, spindle, f'CutCammingThick{suffix}/')
    cutpaththin  = os.path.join(pathname, spindle, f'CutCammingThin{suffix}/')
    cutpathmed   = os.path.join(pathname, spindle, f'CutCammingMed{suffix}/')

    # Lockfile check
    lockpath = {'Thick': cutpaththick, 'Thin': cutpaththin, 'Med': cutpathmed}.get(cuttype)
    if cu._check_lockfile(lockpath):
        return 'Lockfile present'

    # Ensure directories exist
    for path in (cutpaththick, cutpaththin, cutpathmed):
        os.makedirs(path, exist_ok=True)

    # Load parameters
    thick_depth, med_depth, thin_depth, cutpitch = cu.get_cut_parameters(os.path.join(pathname, cutparamsfile))
    newxoffset, newyoffset, newzoffset = cu.get_spindle_offsets(calibrationfilepath, spindle)

    # Depth values
    depths = {
        'Thick': thick_depth,
        'Med': thick_depth + med_depth,
        'Thin': thick_depth + med_depth + thin_depth
    }
    offsets = (newxoffset, newyoffset, newzoffset)
    pitch = cutpitch
    yres = 0.500
    measrad = 0.500

    print(f"Using plane parameters: {p}")
    print(f"X range: ({xstart}, {xend}), Blade radius: {bladeradius}")
    print(f"Offsets: {offsets}, Cut depths: {depths}")

    # Define Y-range generator
    if geometry == 'circular':
        if cutdiameter is None or ycenter is None:
            raise ValueError("cutdiameter and ycenter must be provided for circular geometry")
        radius = cutdiameter / 2.0
        xcenter = (xstart + xend) / 2.0

        def get_ys(xx):
            dx = xx - xcenter
            if abs(dx) > radius:
                return np.array([])
            dy = np.sqrt(radius**2 - dx**2)
            return np.arange(ycenter - dy, ycenter + dy + 0.0001, yres)

    elif geometry == 'rectangular':
        if ystart is None or yend is None:
            raise ValueError("ystart and yend must be provided for rectangular geometry")

        def get_ys(xx):
            return np.arange(ystart, yend, yres)

    else:
        raise ValueError("geometry must be 'circular' or 'rectangular'")

    # Plane function
    def F(x, y, a, b, c):
        return -a * x - b * y - c

    # Select path and values
    cam_args = {
        'cutpath':{'Thick': cutpaththick, 'Thin': cutpaththin, 'Med': cutpathmed}[cuttype],
        'thickness_label':cuttype,
        'p':p,
        'plane_func':F,
        'xstart':xstart,
        'xend':xend,
        'get_ys':get_ys,
        'pitch':pitch,
        'yres':yres,
        'offsets':offsets,
        'bladeradius':bladeradius,
        'depth':depths[cuttype],
        'measrad':measrad
    }

    cu._write_cam_set(**cam_args)

    return None
