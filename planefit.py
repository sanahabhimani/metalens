import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import optimize as opt
from scipy import interpolate
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


### Plane Fitting Functions Relevant to Planar Lens and Alumina Filter ###
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

    return p, corrections, zmodel, residuals, corrected_residuals, xin, yin


def generate_planar_files(
    xin,
    yin,
    p,
    corrections,
    pathname,
    spindle,
    calibrationfilepath,
    cutparamsfile,
    cutdiameter,
    xcenter,
    ycenter,
    bladeradius,
    flag,
    correction_max=0.070,
    yres=0.5,
    measrad=0.5,
    use_noshift_suffix=True
):
    """
    Generate cut camming files for a lens using Fourier-corrected plane fitting results.

    Parameters
    ----------
    xin, yin : ndarray
        X and Y positions from metrology data.
    p : array-like
        Plane fit coefficients [a, b, c].
    corrections : array-like
        Fourier correction values (pre-clipped).
    pathname : str
        Root path to write camming files.
    spindle : str
        Spindle identifier (used in folder names).
    calibrationfilepath : str
        Path to calibration file for spindle offsets.
    cutparamsfile : str
        File containing cut depths and pitch.
    cutdiameter : float
        Diameter of the circular region to be cut.
    xcenter, ycenter : float
        Center coordinates for the circular cut.
    bladeradius : float
        Radius of the blade used for cutting.
    flag : str
        Cut type: 'Thick', 'Med', or 'Thin'.
    correction_max : float, optional
        Maximum allowed value for Fourier correction.
    yres : float, optional
        Resolution in Y-direction.
    measrad : float, optional
        Measurement radius (default is 0.5 mm).
    use_noshift_suffix : bool, default=True
        If True, appends '-Noshift' to cut directory names. Default is True.

    Returns
    -------
    None
    """
    # Determine cut path
    suffix = '-Noshift' if use_noshift_suffix else ''
    cutpath = os.path.join(pathname, spindle, f'CutCamming{flag}{suffix}/')

    # Lockfile check
    lockfile = os.path.join(cutpath, 'lockfile.lock')
    if os.path.exists(lockfile):
        print(f"Lockfile {flag} present")
        return 'Lockfile present'

    os.makedirs(cutpath, exist_ok=True)

    # Load cut parameters and spindle offsets
    thick_depth, med_depth, thin_depth, cutpitch = cu.get_cut_parameters(os.path.join(pathname, cutparamsfile))
    newxoffset, newyoffset, newzoffset = cu.get_spindle_offsets(calibrationfilepath, spindle)

    # Set depth and offsets
    depth = {'Thick': thick_depth,
             'Med': thick_depth + med_depth,
             'Thin': thick_depth + med_depth + thin_depth}[flag]

    zoffset = newzoffset
    xoffset = newxoffset
    yoffset = newyoffset

    # Plane function
    def F(x, y, a, b, c):
        return -a * x - b * y - c

    # X range for cutting
    xstart = xcenter - cutdiameter / 2.0 + yres
    xend = xcenter + cutdiameter / 2.0
    xs = np.arange(xstart, xend, cutpitch)

    # Write master file
    cutmasterfile = open(os.path.join(cutpath, 'Master.txt'), 'w')

    for j, xx in enumerate(xs):
        dx = xx - xcenter
        if abs(dx) > cutdiameter / 2.0:
            continue

        dy = np.sqrt((cutdiameter / 2.0)**2 - dx**2)
        ystart = ycenter - dy
        yend = ycenter + dy + 0.0001
        ys = np.arange(ystart, yend, yres)

        zs = np.zeros_like(ys)
        for i, yy in enumerate(ys):
            correction_idx = np.argmin(np.hypot(xin - xx, yin - yy))
            correction = corrections[correction_idx]
            if abs(correction) > correction_max:
                correction = np.sign(correction) * correction_max

            zs[i] = F(xx, yy, *p) + correction - zoffset + bladeradius - depth - measrad

        fname = os.path.join(cutpath, f'CutCam{flag}')
        make_cam_file(fname, j, xx + xoffset, ys + yoffset, zs)

        xvar = xx + xoffset
        xstr = '%.3f' % xvar
        cutmasterfile.write(f"{j:04d} {xstr} {ys[0] + yoffset:.3f} {zs[0]:.3f} {ys[-1] + yoffset:.3f}\n")

    cutmasterfile.close()
