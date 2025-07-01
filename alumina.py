import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from scipy import optimize as opt
import core_utils as cu
import planefit as pf

def shiftXZ_alumina_filter(directory, spindle, ftype, Xshift, zshift_fixed, correction_zshift,
                           wear_coeff, exposureval, lastlinecut, firstline, numlines):
    """
    Applies X and Z shift corrections to CAM files and updates Master.txt files
    for alumina filter processing, accounting for cumulative blade wear and exposure.

    Parameters
    ----------
    directory : str
        Root directory containing spindle folders.
    spindle : str
        Spindle identifier (subfolder within the root directory).
    ftype : str
        Type of cut ('Thick', 'Thin', or 'Med').
    Xshift : float
        Fixed shift to apply to x-coordinates.
    zshift_fixed : float
        Fixed base shift to apply to z-coordinates.
    correction_zshift : float
        Additional correction shift to apply to z-coordinates.
    wear_coeff : float
        Blade wear coefficient used to calculate wear-based z-shifts.
    exposureval : float
        Maximum allowable cumulative exposure before stopping blade use.
    lastlinecut : int
        Line number of the last cut in the previous session.
    firstline : int
        First line number to process in this run.
    numlines : int
        Number of lines to cut/process in this run.
    """
    # Define folders and file paths
    subfolder = f'CutCamming{ftype}'
    camname = f"{directory}{spindle}/{subfolder}/CutCam{ftype}"
    cin = f"{directory}{spindle}/{subfolder}-Noshift/CutCam{ftype}"
    masterfilein = f"{directory}{spindle}/{subfolder}-Noshift/Master.txt"
    masterfilein_actual = f"{directory}{spindle}/{subfolder}-ActualDiameter/Master.txt"
    lockfile = f"{directory}{spindle}/{subfolder}/lockfile.lock"

    if os.path.isfile(lockfile):
        print("Lockfile present")
    else:
        print("Lockfile not present")

    # Load Master and Actual Master data
    mfile = np.loadtxt(masterfilein)
    nums = mfile[:, 0]
    xs, ys, zs, ystops = mfile[:, 1], mfile[:, 2], mfile[:, 3], mfile[:, 4]
    mfile_actual = np.loadtxt(masterfilein_actual)
    xs_actual, ys_actual, ystops_actual = mfile_actual[:, 1], mfile_actual[:, 2], mfile_actual[:, 4]

    # Compute deltays in actual data and insert into correct region
    deltays = ystops_actual - ys_actual
    first_index = np.where(xs == xs_actual[0])[0][0]
    last_index = np.where(xs == xs_actual[-1])[0][0]
    deltays_mod = np.zeros_like(zs)
    deltays_mod[first_index:last_index + 1] = deltays

    # Compute cumulative wear shift
    wearshiftsarray = deltays_mod * wear_coeff
    cum_wearshift = np.zeros_like(wearshiftsarray)

    # Resume from previous exposure
    if lastlinecut == 0:
        init_exp = 0
    else:
        oldshifts_path = f"{directory}{spindle}/WearshiftValues_{ftype}.txt"
        oldshifts = np.loadtxt(oldshifts_path)
        init_exp = oldshifts[lastlinecut, 1]

        os.rename(oldshifts_path, f"{directory}{spindle}/WearshiftValues_{ftype}_upto{lastlinecut}.txt")
        current_dir = f"{directory}{spindle}/{subfolder}"
        if os.path.isdir(current_dir):
            os.rename(current_dir, f"{current_dir}_upto{lastlinecut}")

    # Determine exposure-based stopping point
    s = init_exp
    lines2cut = np.arange(firstline, firstline + numlines)
    for val in lines2cut:
        if abs(s) >= exposureval:
            print("Blade Limit Line", val)
            lastline = val
            break
        cum_wearshift[val] = s
        s += wearshiftsarray[val]

    # Plot wear shift profile
    plt.figure()
    plt.plot(cum_wearshift)
    plt.xlabel('Line Number')
    plt.ylabel('Z Shift (mm)')
    plt.title('Cumulative Wear Shift')
    plt.show()

    # Apply shifts
    zsout = zs + zshift_fixed + cum_wearshift + correction_zshift
    xsout = xs + Xshift

    # Save wear shift values
    wearfile_path = f"{directory}{spindle}/WearshiftValues_{ftype}.txt"
    wearfileout = open(wearfile_path,'w')
    for i in range(len(cum_wearshift)):
        linenum = f"{i:04d}"
        wearfileout.write(f"{linenum} {cum_wearshift[i] + correction_zshift}\n")
    wearfileout.close()

    # Update CAM files and Master.txt
    masterfiledir = f"{directory}{spindle}/{subfolder}/"
    masterfileout = f"{masterfiledir}/Master.txt"
    if not os.path.isdir(masterfiledir):
        os.makedirs(masterfiledir)
    else:
        os.rename(masterfiledir, f"{masterfiledir.rstrip('/')}_UpToLine_{lastlinecut}")
        os.makedirs(masterfiledir)

    with open(masterfileout, 'w') as mfileout:
        for i, num in enumerate(nums):#mfile[firstline:firstline+numlines, 0]):
            linenum = f"{int(num):04d}"
            fname = f"{cin}{linenum}.Cam"
            cname = f"{directory}{spindle}/{subfolder}/CutCam{ftype}"
            pts = np.loadtxt(fname, skiprows=4)
            yscam = pts[:, 1]
            zscam = pts[:, 2] + zshift_fixed + cum_wearshift[i] + correction_zshift

            cu.make_cam_file(cname, num, xs[i], yscam, zscam)
            mfileout.write(f"{linenum} {xsout[i]} {ys[i]} {zsout[i]} {ystops[i]}\n")

    cu.remove_lines(masterfiledir, firstline, firstline + numlines - 1, ftype)


def wear_coefficient_update(path, spindle, cuttype, files):
    """
    Updates the wear coefficient by analyzing cumulative wear shift data across multiple cut files.

    Parameters
    ----------
    path : str
        Base path where the spindle and cut folders are located.
    spindle : str
        Spindle name (subfolder inside path).
    cuttype : str
        Cut type ('Thick', 'Thin', 'Med') used to find CutCamming folders.
    files : list of str
        List of filenames (relative to spindle folder) containing wear shift values.

    Notes
    -----
    - Analyzes wear shift data and computes the updated wear coefficient (linear fit slope).
    - Prints the updated wear coefficient.
    - No plots are generated.
    """
    # Load initial wear shift data
    # TODO: change os.path.join to using Pathlib
    testfile = np.loadtxt(os.path.join(path, spindle, files[0]))
    num_lines = len(testfile[:, 0])
    wearshifts = np.zeros(num_lines)
    lines = np.zeros(num_lines)
    filtered_wearshifts = []
    filtered_lines = []

    for filename in files:
        tempdata = np.loadtxt(os.path.join(path, spindle, filename))
        # Select indices where the wear shift changes
        change_indices = np.where(tempdata[:, 1] != tempdata[0, 1])[0]

        if len(change_indices) == 0:
            continue  # No changes detected, skip this file

        tempind = change_indices[0]
        filtered_lines.append(tempind)
        filtered_wearshifts.append(tempdata[tempind, 1])

        for j in change_indices:
            wearshifts[int(j)] = tempdata[j, 1]
            lines[int(j)] = tempdata[j, 0]

    new_wearshifts = np.copy(wearshifts)
    new_lines = np.copy(lines)

    # Mask lines where wearshift is still the initial value
    new_wearshifts[new_wearshifts == wearshifts[0]] = np.nan
    new_lines[wearshifts == wearshifts[0]] = np.nan

    # Load master files
    master_actual = np.loadtxt(os.path.join(path, spindle, f'CutCamming{cuttype}-ActualDiameter/Master.txt'))
    master_noshift = np.loadtxt(os.path.join(path, spindle, f'CutCamming{cuttype}-Noshift/Master.txt'))

    first_index = np.where(master_noshift[:, 1] == master_actual[0, 1])[0][0]
    last_index = np.where(master_noshift[:, 1] == master_actual[-1, 1])[0][0]

    deltays = master_actual[:, 4] - master_actual[:, 2]
    smartdeltays = np.zeros_like(master_noshift[:, 0])
    valid_indices = ~np.isnan(new_wearshifts)

    smartdeltays[valid_indices] = deltays[valid_indices]
    smartcumdist = np.cumsum(smartdeltays)

    filtered_cumsum = [smartcumdist[int(i)] for i in filtered_lines]

    # Linear regression
    fitresults = spstats.linregress(filtered_cumsum, filtered_wearshifts)
    print('Updated Wear Coefficient:', fitresults.slope)


def generate_alumina_cutfiles(
    pathname,
    spindle,
    calibrationfilepath,
    cutparamsfile,
    flag,               # 'Thick', 'Med', or 'Thin'
    cutdiameterflag,    # 'ActualDiameter' or 'Noshift'
    cutdiameter_actual,
    cutdiameter_noshift,
    xcenter,
    ycenter,
    bladeradius,
    p,                  # plane‐fit parameters from planefit.planefit()
    corrections,
    A_coef,
    xin,
    yin,
):
    """
    Given the outputs of planefit.planefit (p, corrections, zmodel, residuals, etc.)
    plus all the original cut‐parameter inputs, create the same “CutCamming” files
    that plane_fit_alumina_filter used to produce.

    Parameters
    ----------
    pathname : str
        Base directory for all metrology and camming file subfolders.
        Example: '/some/base/path/'
    spindle : str
        Name of the spindle subfolder under pathname (e.g. 'SpindleC').
    calibrationfilepath : str
        Full path to the calibration file, used to compute spindle offsets.
    cutparamsfile : str
        Relative path (under pathname) to the file that defines
        thick/med/thin depths and cut pitch. Passed to `get_cut_parameters`.
    flag : str
        Which blade set to generate: 'Thick', 'Med', or 'Thin'.
    cutdiameterflag : str
        Either 'ActualDiameter' or 'Noshift'. Chooses which subfolder
        naming convention to use for cutcamming.
    cutdiameter_actual : float
        The “ActualDiameter” value in mm.
    cutdiameter_noshift : float
        The “Noshift” diameter value (Actual + shift) in mm.
    xcenter, ycenter : float
        Center (X, Y) of the circular cut region.
    bladeradius : float
        Radius of this blade (in mm).
    p : array_like, shape (3,)
        Plane‐fit parameters [a, b, c] from planefit.planefit().
    corrections : array_like, shape (N,)
        Fourier‐correction values (clipped) at each data point (xin, yin).
    A_coef: array_like, shape (N, )
        The Fourier coefficients for reconstructing the surface at arbitrary points
        accurately.
    xin, yin : array_like, shape (N,)
        The X and Y coordinates of each metrology measurement, as returned
        from planefit.planefit().

    Returns
    -------
    None
        Writes out one “Master.txt” and all “CutCam” files for the specified flag.
    """
    fourier_max = 2
    correction_max = 0.070

    # 1) Determine cut‐camming subfolder names
    metpath     = os.path.join(pathname, 'MetrologyCamming')
    if cutdiameterflag == 'Noshift':
        cutpaththick = os.path.join(pathname, spindle, 'CutCammingThick-Noshift')
        cutpaththin  = os.path.join(pathname, spindle, 'CutCammingThin-Noshift')
        cutpathmed   = os.path.join(pathname, spindle, 'CutCammingMed-Noshift')
    elif cutdiameterflag == 'ActualDiameter':
        cutpaththick = os.path.join(pathname, spindle, 'CutCammingThick-ActualDiameter')
        cutpaththin  = os.path.join(pathname, spindle, 'CutCammingThin-ActualDiameter')
        cutpathmed   = os.path.join(pathname, spindle, 'CutCammingMed-ActualDiameter')
    else:
        raise ValueError("cutdiameterflag must be 'Noshift' or 'ActualDiameter'")

    # 2) Before proceeding, check for lockfile in the relevant folder
    if flag == 'Thick':
        lock_folder = cutpaththick
    elif flag == 'Thin':
        lock_folder = cutpaththin
    elif flag == 'Med':
        lock_folder = cutpathmed
    else:
        raise ValueError("flag must be one of 'Thick', 'Med', or 'Thin'")

    lockfile_path = os.path.join(lock_folder, 'lockfile.lock')
    try:
        with open(lockfile_path) as _:
            print(f"Lockfile present in {lock_folder}; skipping camming‐file generation.")
            return
    except IOError:
        # lockfile not present → continue
        pass

    # 3) Ensure each cut folder exists
    os.makedirs(cutpaththick, exist_ok=True)
    os.makedirs(cutpaththin,  exist_ok=True)
    os.makedirs(cutpathmed,   exist_ok=True)

    # 4) Load depths and pitch from cutparamsfile
    #    get_cut_parameters should return (thick_depth, med_depth, thin_depth, cutpitch)
    thick_depth, med_depth, thin_depth, cutpitch = cu.get_cut_parameters(os.path.join(pathname, cutparamsfile))

    # 5) Decide which diameter to use
    if cutdiameterflag == 'Noshift':
        Cutdiam = cutdiameter_noshift
    else:  # 'ActualDiameter'
        Cutdiam = cutdiameter_actual

    # 6) Compute Xstart, Xend, pitch, Y resolution, measurement radius
    Xstart = xcenter - Cutdiam/2.0 + 0.500
    Xend   = xcenter + Cutdiam/2.0
    pitch  = cutpitch
    Yres   = 0.500
    measrad = 0.500

    # 7) Get spindle offsets (Xoffset, Yoffset, Zoffset) from calibration file
    newxoffset, newyoffset, newzoffset = cu.get_spindle_offsets(calibrationfilepath, spindle)
    #    (Assumes get_spindle_offsets returns a 3‐tuple of floats.)

    # 8) Assign blade‐specific offsets/depths
    #    - “Thick” blade is hubbed
    #    - “Med” blade is hubless (so initial depth + med_depth)
    #    - “Thin” blade is hubbed (so initial depth + med + thin)
    if flag == 'Thick':
        Xoffset = newxoffset
        Yoffset = newyoffset
        Zoffset = newzoffset
        Radius  = bladeradius
        Depth   = thick_depth
    elif flag == 'Med':
        Xoffset = newxoffset
        Yoffset = newyoffset
        Zoffset = newzoffset
        Radius  = bladeradius
        Depth   = thick_depth + med_depth
    else:  # 'Thin'
        Xoffset = newxoffset
        Yoffset = newyoffset
        Zoffset = newzoffset
        Radius  = bladeradius
        Depth   = thick_depth + med_depth + thin_depth

    # 9) Begin writing camming files
    #    We'll generate:
    #      Master.txt  (one line per xs index: filenum, xstart, ystart, zstart, ystop)
    #      CutCam<Flag><index>  (one g‐code file per X‐slice)

    if flag == 'Thick':
        out_folder = cutpaththick
        master_filename = os.path.join(out_folder, 'Master.txt')
    elif flag == 'Med':
        out_folder = cutpathmed
        master_filename = os.path.join(out_folder, 'Master.txt')
    else:  # 'Thin'
        out_folder = cutpaththin
        master_filename = os.path.join(out_folder, 'Master.txt')

    xs = np.arange(Xstart, Xend, pitch)
    # Preallocate arrays if you need to record ystart/zstart; but since the original
    # didn’t return them, we’ll just write them on the fly.
    cutmasterfile = open(master_filename, 'w')

    for j, xx in enumerate(xs):
        # 10) For each X‐slice, find Ystart and Yend around the circle:
        half_span = np.sqrt(max((Cutdiam/2.0)**2 - (xx - xcenter)**2, 0))
        Ystart = ycenter - half_span
        Yend   = ycenter + half_span + 0.0001  # tiny epsilon like original

        ys = np.arange(Ystart, Yend, Yres)
        zs = np.zeros_like(ys)

        # 11) For each (xx, yy), compute the fourier correction + plane + blade offset
        for i, yy in enumerate(ys):
            raw_plane = -p[0]*xx - p[1]*yy - p[2]
            corr_val = pf.fourier_eval(A_coef, xx, yy, fourier_max)

            # TODO: FIX: check to see if abs(correction) argument from original code should apply
            corr_val = np.sign(corr_val) * min(abs(corr_val), correction_max)

            # Now compute Z: plane + corr − Zoffset + Radius − Depth − measrad
            zs[i] = raw_plane + corr_val - Zoffset + Radius - Depth - measrad

        # 12) Write the g‐code file for this X index
        fname = Path(out_folder) / f'CutCam{flag}'
        cu.make_cam_file(fname, j, xx + Xoffset, ys + Yoffset, zs)

        # 13) Record one line in Master.txt: index, xstart, ystart, zstart, ystop
        filenum = f"{j:04d}"
        x_str = f"{xx + Xoffset:.3f}"
        y_start_str = f"{ys[0] + Yoffset:.7f}"
        z_start_str = f"{zs[0]:.7f}"
        y_end_str = f"{ys[-1] + Yoffset:.7f}"
        cutmasterfile.write(f"{filenum} {x_str} {y_start_str} {z_start_str} {y_end_str}\n")

    cutmasterfile.close()
    print(f"Finished generating '{flag}' camming files in '{out_folder}'")

