import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import optimize as opt
import core_utils as cu


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


def generate_hkcut_files(
    p,
    pathname,
    spindle,
    calibrationfilepath,
    cutparamsfile,
    bladeradius,
    cuttype,
    xstart,
    xend,
    ystart,
    yend,
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

    ystart : float, optional
        Starting Y position for rectangular cuts.

    yend : float, optional
        Ending Y position for rectangular cuts.

    use_noshift_suffix : bool, default=True
        If True, appends '-Noshift' to cut directory names (applies to all current use cases). Default is True.


    Returns
    -------
    None or str
        Returns 'Lockfile present' if a lock file exists and cut generation is skipped. Otherwise, writes cam files.
    """
    # Construct cut paths
    suffix = '-Noshift' if use_noshift_suffix else ''
    cutpaththick = os.path.join(pathname, spindle, f'CutCammingThick{suffix}/')
    cutpaththin  = os.path.join(pathname, spindle, f'CutCammingThin{suffix}/')
    cutpathmed   = os.path.join(pathname, spindle, f'CutCammingMed{suffix}/')

    # Lockfile check
    lockpath = {'Thick': cutpaththick, 'Thin': cutpaththin, 'Med': cutpathmed}.get(cuttype)
    cu._check_lockfile(lockpath)

    # Make the relevant directories
    for path in (cutpaththick, cutpaththin, cutpathmed):
        os.makedirs(path, exist_ok=True)

    # Load blade cut depth parameters
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


    if ystart is None or yend is None:
        raise ValueError("ystart and yend must be provided for rectangular geometry")

    def get_ys(xx):
        return np.arange(ystart, yend, yres)

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

    cu._write_cam_set_planar(**cam_args)
