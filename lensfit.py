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
    xstart = xcenter - radius + x_rot_shift
    xend = xcenter + radius

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

