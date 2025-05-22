import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import optimize as opt
import core_utils as cu

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
        return
    else:
        print("Lockfile not present")

    # Load Master and Actual Master data
    mfile = np.loadtxt(masterfilein)
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
    with open(wearfile_path, 'w') as wearfile:
        for i, shift in enumerate(cum_wearshift):
            linenum = f"{i:04d}"
            wearfile.write(f"{linenum} {shift + correction_zshift}\n")

    # Update CAM files and Master.txt
    masterfiledir = f"{directory}{spindle}/{subfolder}/"
    if os.path.isdir(masterfiledir):
        os.rename(masterfiledir, f"{masterfiledir.rstrip('/')}_UpToLine_{lastlinecut}")
    os.makedirs(masterfiledir, exist_ok=True)

    masterfileout = f"{masterfiledir}/Master.txt"
    with open(masterfileout, 'w') as mfileout:
        for i, num in enumerate(mfile[:, 0]):
            linenum = f"{int(num):04d}"
            fname = f"{cin}{linenum}.Cam"
            cname = f"{directory}{spindle}/{subfolder}/CutCam{ftype}"
            pts = np.loadtxt(fname, skiprows=4)
            yscam = pts[:, 1]
            zscam = pts[:, 2] + zshift_fixed + cum_wearshift[i] + correction_zshift

            make_cam_file(Path(cname), num, xs[i], yscam, zscam)
            mfileout.write(f"{linenum} {xsout[i]} {ys[i]} {zsout[i]} {ystops[i]}\n")

    remove_lines(masterfiledir, firstline, firstline + numlines - 1, ftype)


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

