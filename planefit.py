## this is me making some changes to a few plane fit functions

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import optimize as opt


def make_cam_file(filename, filenum, xval,ys,zs):
    #code to automate the making of camfiles
    numstr = "%04g" % (filenum)
    numpts = "%04g" %(len(ys))
    fname = filename + numstr + '.Cam'
    print(fname)
    f = open(fname,'w')
    f.write(';Filename: '+fname + '\n')
    f.write('Number of points ' + numpts + '\n')
    f.write('Master Units (PRIMARY)'+ '\n')
    f.write('Slave Units (PRIMARY)'+ '\n')
    for i,y in enumerate(ys):
        num = "%04g" % (i+1)
        f.write(num + ' ' + str(ys[i]) + ' ' + str(zs[i]) + '\n')

    return 1


def get_cut_parameters(cutparamsfile):
    file = np.loadtxt(cutparamsfile,dtype=str,skiprows=1)
    thick_depth, med_depth, thin_depth, cutpitch = float(file[0]),float(file[1]),float(file[2]),float(file[3])
    return thick_depth, med_depth, thin_depth, cutpitch


def get_spindle_offsets(spindlecalfile,spindle):
    file = np.loadtxt(spindlecalfile,dtype=str,skiprows=1)
    calfileindex = np.where(file[:,0]==spindle)[0][0]
    xoffset, yoffset, zoffset = file[calfileindex,1], file[calfileindex,2], file[calfileindex,3]

    return float(xoffset), float(yoffset), float(zoffset)


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
    xstart,
    xend,
    ystart,
    yend,
    bladeradius,
    flag
):
    """
    Generate cut-camming files (Thick, Thin, or Med) for a flat/planar surface,
    using pre-computed plane-fit parameters p = [a, b, c].

    Unlike the old plane_fit_* functions, this does NOT do a plane fit itself;
    it just writes out the cam files. The assumption is that you've already
    run planefit(...) or any other plane-fitting method to get p.

    Parameters
    ----------
    p : array-like of float
        Plane coefficients [a, b, c] for z = -a*x - b*y - c, as returned by planefit(...).
    pathname : str
        Base path for file I/O.
    spindle : str
        Spindle identifier or name.
    calibrationfilepath : str
        Path to calibration data (used for offset retrieval).
    cutparamsfile : str
        File containing thick_depth, med_depth, thin_depth, and pitch.
    xstart, xend, ystart, yend : float
        Boundaries for the cutting area.
    bladeradius : float
        Radius of the blade.
    flag : str
        "Thick", "Thin", or "Med" indicating which cut-camming set to generate.

    Returns
    -------
    str or None
        Returns "Lockfile present" if a lock file is found, otherwise None.
        The function also writes files to disk as a side effect.
    """

    # We assume you want to always use -Noshift/ in the directory paths,
    # as if it were 'dressing' or 'test wafer' with -Noshift.
    cutpaththick = os.path.join(pathname, spindle, 'CutCammingThick-Noshift/')
    cutpaththin  = os.path.join(pathname, spindle, 'CutCammingThin-Noshift/')
    cutpathmed   = os.path.join(pathname, spindle, 'CutCammingMed-Noshift/')

    # 1) Lockfile checks
    if flag == 'Thick':
        if _check_lockfile(cutpaththick):
            return 'Lockfile present'
    elif flag == 'Thin':
        if _check_lockfile(cutpaththin):
            return 'Lockfile present'
    elif flag == 'Med':
        if _check_lockfile(cutpathmed):
            return 'Lockfile present'

    # 2) Ensure directories exist
    for path in (cutpaththick, cutpaththin, cutpathmed):
        if not os.path.isdir(path):
            os.makedirs(path)

    # 3) Load cutting parameters + spindle offsets
    thick_depth, med_depth, thin_depth, cutpitch = get_cut_parameters(
        os.path.join(pathname, cutparamsfile)
    )
    newxoffset, newyoffset, newzoffset = get_spindle_offsets(
        calibrationfilepath,
        spindle
    )

    # Depth offsets
    Thickdepth = thick_depth
    Meddepth   = Thickdepth + med_depth
    Thindepth  = Meddepth   + thin_depth

    # Some constants
    Xstart = xstart
    Xend   = xend
    Ystart = ystart
    Yend   = yend
    pitch  = cutpitch
    Yres   = 0.500
    measrad = 0.500

    # Print a bit of info
    print("Using plane parameters:", p)
    print("X range:", (Xstart, Xend), "Y range:", (Ystart, Yend))
    print("Blade radius:", bladeradius)
    print("Cut depths [Thick, Med, Thin]:", (Thickdepth, Meddepth, Thindepth))
    print("Offsets [X, Y, Z]:", (newxoffset, newyoffset, newzoffset))

    # 4) Write out the cam files based on flag
    # We'll define a local helper for computing the plane
    def F(x, y, a, b, c):
        return -a*x - b*y - c

    if flag == 'Thick':
        _write_cam_set(
            cutpaththick, 'Thick', p, F,
            Xstart, Xend, Ystart, Yend,
            pitch, Yres,
            (newxoffset, newyoffset, newzoffset),
            bladeradius, Thickdepth, measrad
        )
    elif flag == 'Thin':
        _write_cam_set(
            cutpaththin, 'Thin', p, F,
            Xstart, Xend, Ystart, Yend,
            pitch, Yres,
            (newxoffset, newyoffset, newzoffset),
            bladeradius, Thindepth, measrad
        )
    elif flag == 'Med':
        _write_cam_set(
            cutpathmed, 'Med', p, F,
            Xstart, Xend, Ystart, Yend,
            pitch, Yres,
            (newxoffset, newyoffset, newzoffset),
            bladeradius, Meddepth, measrad
        )

    return None  # Indicate success if no lockfile prevented it


def _check_lockfile(path):
    """
    Helper to check if a lockfile.lock is present in the given path.
    Returns True if it's present, False otherwise.
    """
    lockfile = os.path.join(path, 'lockfile.lock')
    try:
        with open(lockfile) as _f:
            print(f"Lockfile present in {path}")
            return True
    except IOError:
        print(f"No lockfile in {path}")
        return False


def _write_cam_set(
    cutpath,
    thickness_label,
    p, plane_func,
    Xstart, Xend, Ystart, Yend,
    pitch, Yres,
    offsets,
    bladeradius, depth, measrad
):
    """
    Write out the 'Master.txt' and all the sub-cam files for a given thickness type.
    """
    (Xoffset, Yoffset, Zoffset) = offsets
    xs = np.arange(Xstart, Xend, pitch)

    master_file_path = os.path.join(cutpath, 'Master.txt')
    with open(master_file_path, 'w') as masterfile:
        for j, xx in enumerate(xs):
            ys = np.arange(Ystart, Yend, Yres)
            zs = np.zeros(len(ys))
            for i, yy in enumerate(ys):
                zcalc = plane_func(xx, yy, *p)
                # Apply offsets, blade radius, measured radius, depth, etc.
                zs[i] = zcalc - Zoffset + bladeradius - depth - measrad

            # E.g. "CutCamThick", "CutCamThin", etc.
            fname_prefix = os.path.join(cutpath, f'CutCam{thickness_label}')
            make_cam_file(fname_prefix, j, xx + Xoffset, ys + Yoffset, zs)

            linenum = f"{j:04d}"
            masterfile.write(
                f"{linenum} {xx + Xoffset} {ys[0] + Yoffset} {zs[0]} {ys[-1] + Yoffset}\n"
            )

'''
### How to call functions ###

from planefit import planefit
# Suppose we have a full path to our .dat or .csv file
p, cov, infodict, mesg, ier = planefit("/path/to/data/metfile.csv", do_plot=True)
# p is [a, b, c]

result = generate_planar_cut_files(
    p,
    pathname="/some/base/path",
    spindle="Spindle01",
    calibrationfilepath="/path/to/calibrationfile",
    cutparamsfile="cutparams.txt",
    xstart=0, xend=100,
    ystart=0, yend=75,
    bladeradius=2.5,
    flag="Thick"
)

if result == "Lockfile present":
    print("Aborting since a lockfile was found.")
else:
    print("Cut files generated successfully.")
'''
