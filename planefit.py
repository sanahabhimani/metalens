## this is me making some changes to a few plane fit functions

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import optimize as opt
import core_utils as cu


def planefit(filepath, do_plot=True):
    """
    Load metrology data from 'filepath', perform a plane fit, optionally plot the results.

    Parameters
    ----------
    filepath : str
        Full path to the metrology .dat file containing columns [x, y, z, r].
    do_plot : bool, optional
        If True, generate a 4-panel 3D plot of the data, residuals, gauge, and fit.

    Returns
    -------
    p, cov, infodict, mesg, ier
        Fit parameters, covariance, info dict, message, and the integer status from leastsq.
    """
    # Load CSV data
    pts = np.loadtxt(filepath, delimiter=",")
    xin = pts[:, 0]
    yin = pts[:, 1]
    zin = pts[:, 2]
    r   = pts[:, 3]

    # Combine gauge reading
    qin = zin + r

    def F(x, y, a, b, c):
        return -a*x - b*y - c

    def planefit_residuals(params, x, y, q):
        a, b, c = params
        return q - F(x, y, a, b, c)

    # Initial guess
    p0 = [0.0, 0.0, 0.0]

    # Fit
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

    # Optional plotting
    if do_plot:
        residuals = qin - F(xin, yin, *p)

        fig = plt.figure()
        # Data
        ax = fig.add_subplot(1, 4, 1, projection="3d")
        ax.plot_trisurf(xin, yin, qin, cmap=cm.jet, linewidth=0.2)
        ax.set_title("Data")

        # Residuals
        ax = fig.add_subplot(1, 4, 2, projection="3d")
        ax.plot_trisurf(xin, yin, residuals, cmap=cm.jet, linewidth=0.2)
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")
        ax.set_zlabel("Z residual (mm)")
        plt.title("Residuals")

        # Gauge readout
        ax = fig.add_subplot(1, 4, 3, projection="3d")
        ax.plot_trisurf(xin, yin, r, cmap=cm.jet, linewidth=0.2)
        ax.set_title("Gauge (r)")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")

        # Model surface
        zmodel = F(xin, yin, *p)
        ax = fig.add_subplot(1, 4, 4, projection="3d")
        ax.plot_trisurf(xin, yin, zmodel, cmap=cm.jet, linewidth=0.2)
        ax.set_title("Model")
        plt.xlabel("X (mm)")
        plt.ylabel("Y (mm)")

        plt.show()

    return p, cov, infodict, mesg, ier


## How to call the functions
# p, cov, infodict, mesg, ier = planefit("/some/path/met_data.dat")


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
