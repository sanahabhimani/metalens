import numpy as np
import serial
import time
import os
import scipy.interpolate as interpolate
import scipy.optimize as opt
import scipy.stats as spstats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pathlib import Path


def check_metrology_probe(comport):
    """
    Communicates with a metrology probe over a serial port and returns its response.

    Parameters
    ----------
    comport : str
        The serial port to connect to (e.g., 'COM3' or '/dev/ttyUSB0').

    Returns
    -------
    str
        The decoded response from the probe after issuing the reset command.
    """
    reset_command = 'RMD0\r\n'
    reset_bytes = reset_command.encode()

    ser = serial.Serial(comport, 9600, timeout=1)
    time.sleep(0.02)

    ser.write(reset_bytes)
    time.sleep(0.02)

    response = ser.read(2048).decode("utf-8")
    print(response)

    ser.close()
    return response


def get_cut_parameters(cutparamsfile):
    """
    Loads cut parameters from a file.

    Parameters
    ----------
    cutparamsfile : str or Path
        Path to the cut parameters file. The file should contain values in the
        order: thick_depth, med_depth, thin_depth, cutpitch (one per line or column).

    Returns
    -------
    tuple of float
        A tuple containing:
        - thick_depth
        - med_depth
        - thin_depth
        - cutpitch
    """
    values = np.loadtxt(cutparamsfile, dtype=str, skiprows=1)
    thick_depth, med_depth, thin_depth, cutpitch = map(float, values[:4])
    return thick_depth, med_depth, thin_depth, cutpitch

def make_cam_file(filename, filenum, xval, ys, zs):
    """
    Creates an Aerotech-compatible .Cam file with x, y, and z position data for a given file number.

    The output filename is constructed by appending the 4-digit zero-padded `filenum`
    to the base `filename`, followed by the `.Cam` extension.

    Parameters
    ----------
    filename : str
        The base location where cutcamming files are to be housed. 
        Ex: '/ASO_LAT_Filters/UHF/Surface1/0deg/'
    filenum : int
        The file number, used to generate the specific .Cam filename.
    xval : float
        The x-coordinate value (same for all rows).
    ys : array-like
        Sequence of y-coordinate values.
    zs : array-like
        Sequence of z-coordinate values (must match ys in length).

    Notes
    -----
    The header is formatted exactly for Aerotech reading compatibility
    - Filename: /full/path/to/file.Cam
    - Number of points ####
    - Master Units (PRIMARY)
    - Slave Units (PRIMARY)
    """
    filename = Path(filename)
    numstr = f"{int(filenum):04d}"
    numpts = f"{len(ys):04d}"

    fname = filename.parent / f"{filename.name}{numstr}.Cam"
    print(fname)

    with open(fname, 'w') as f:
        # Correct Aerotech header
        f.write(f";Filename: {fname}\n")
        f.write(f"Number of points {numpts}\n")
        f.write("Master Units (PRIMARY)\n")
        f.write("Slave Units (PRIMARY)\n")

        for idx, (y, z) in enumerate(zip(ys, zs), start=1):
            f.write(f"{idx:04d} {y:.6f} {z:.6f}\n")


def _check_lockfile(path):
    """
    Raises an IOError if 'lockfile.lock' is present in the given directory.
    """
    lockfile = Path(path) / 'lockfile.lock'
    if lockfile.exists():
        # stop immediately if we see a lockfile
        raise IOError(f"Lockfile present in {lockfile.parent}")
    # otherwise, just return silently
    else:
        print("Lockfile not present, moving forward.")


def _write_cam_set_planar(
    cutpath,
    thickness_label,
    p, plane_func,
    xstart, xend,
    get_ys,
    pitch, yres,
    offsets,
    bladeradius, depth, measrad
):
    """
    Writes out Master.txt and CAM files for a given thickness type based on a planar fit.
    Supports dynamic Y-ranges for each scanline to accommodate both rectangular and circular geometries.

    Parameters
    ----------
    cutpath : str or Path
        Directory where files will be written.
    thickness_label : str
        Label for the CAM file prefix ('Thick', 'Thin', etc.).
    p : list or tuple
        Coefficients of the plane function.
    plane_func : callable
        Function representing the planar fit (e.g., a lambda or poly2d).
    xstart, xend : float
        Ranges for x positions.
    get_ys : callable
        Function that takes a single x value and returns a 1D array of corresponding y values
        for that scanline (used for circular or rectangular geometry).
    pitch : float
        Step size in x-direction.
    yres : float
        Step size in y-direction (kept for reference, not actively used here).
    offsets : tuple of float
        (Xoffset, Yoffset, Zoffset) to apply to the generated points.
    bladeradius : float
        Radius of the blade.
    depth : float
        Desired cut depth.
    measrad : float
        Radius of probe tip or other measurement correction.
    """
    from pathlib import Path
    import numpy as np

    cutpath = Path(cutpath)
    cutpath.mkdir(parents=True, exist_ok=True)

    Xoffset, Yoffset, Zoffset = offsets
    xs = np.arange(xstart, xend, pitch)

    master_file_path = cutpath / 'Master.txt'
    with open(master_file_path, 'w') as masterfile:
        for j, xx in enumerate(xs):
            ys = get_ys(xx)
            if len(ys) == 0:
                continue

            zs = np.zeros(len(ys))
            for i, yy in enumerate(ys):
                zcalc = plane_func(xx, yy, *p)
                zs[i] = zcalc - Zoffset + bladeradius - depth - measrad

            fname_prefix = cutpath / f"CutCam{thickness_label}"
            make_cam_file(fname_prefix, j, xx + Xoffset, ys + Yoffset, zs)

            linenum = f"{j:04d}"
            masterfile.write(
                f"{linenum} {xx + Xoffset} {ys[0] + Yoffset} {zs[0]} {ys[-1] + Yoffset}\n"
            )


def get_spindle_offsets(spindlecalfile, spindle):
    """
    Retrieves the x, y, and z offsets for a given spindle from a calibration file.

    Parameters
    ----------
    spindlecalfile : str or Path
        Path to the calibration file (text format, with headers).
    spindle : str
        The spindle ID to look up in the file.

    Returns
    -------
    tuple of float
        The (xoffset, yoffset, zoffset) values for the given spindle.

    Raises
    ------
    IndexError
        If the spindle ID is not found in the file.
    """
    data = np.loadtxt(spindlecalfile, dtype=str, skiprows=1)
    match_idx = np.where(data[:, 0] == spindle)[0][0]

    xoffset = float(data[match_idx, 1])
    yoffset = float(data[match_idx, 2])
    zoffset = float(data[match_idx, 3])

    return xoffset, yoffset, zoffset


def YZ_Calibration_Fit(filepath, bladeradius, minind, maxind, y_guess, z_guess):
    """
    Fits a circle to Y-Z metrology data using least-squares minimization,
    assuming a known blade radius. The fit estimates the center (y, z)
    of the circle traced by the probe.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file containing the metrology probe data.
    bladeradius : float
        Radius of the blade (used as the fixed circle radius).
    minind : int
        Start index for the region of interest in the data.
    maxind : int
        End index for the region of interest in the data.
    y_guess : float
        Initial guess for the y-coordinate of the circle center.
    z_guess : float
        Initial guess for the z-coordinate of the circle center.

    Returns
    -------
    list
        Best-fit [y_center, z_center] for the probe circle.
    """
    # Load and slice the data
    data = np.loadtxt(filepath, delimiter=',')
    y = data[minind:maxind, 1]
    z = data[minind:maxind, 2] + data[minind:maxind, 3]  # z + g compensation

    # Initial guess and fit target
    p0 = [y_guess, z_guess]
    fitfunc = lambda p, y, z: np.sqrt((y - p[0])**2 + (z - p[1])**2)
    errfunc = lambda p, y, z: fitfunc(p, y, z) - bladeradius

    # Least squares fitting
    result = opt.leastsq(
        errfunc, p0, args=(y, z), full_output=True, ftol=1e-14, xtol=1e-14
    )
    p2, cov, infodict, mesg, ier = result
    resids = infodict['fvec']

    print(f'The center in probe coordinates is [y, z] = {p2}')

    # Plot results
    fig, axs = plt.subplots(3, 1, figsize=(8, 15))

    # Circle fit
    axs[0].scatter(y, z, c='r', label='Data Points')
    circ = plt.Circle((p2[0], p2[1]), radius=bladeradius, fill=False, color='blue', linestyle='--')
    axs[0].add_patch(circ)
    axs[0].set_title('Data points and best-fit circle')
    axs[0].set_xlim(min(y) - 0.1, max(y) + 0.1)
    axs[0].set_ylim(min(z) - 0.01, max(z) + 0.01)
    axs[0].set_aspect('equal')
    axs[0].legend()

    # Residuals plot
    axs[1].plot(resids, label='Residuals')
    axs[1].set_title('Residuals')
    axs[1].set_xlabel('Point Index')
    axs[1].set_ylabel('Error (mm)')
    axs[1].legend()

    # Residuals histogram
    axs[2].hist(resids, bins=20)
    axs[2].set_title('Residuals Histogram')
    axs[2].set_xlabel('Error (mm)')
    axs[2].set_ylabel('Count')

    plt.tight_layout()
    plt.show()

    return p2


def camming_files_combine(cut_type, path, spindle, files):
    """
    Combines CAM files and Master.txt files from multiple directories into a single concatenated folder.

    Parameters
    ----------
    cut_type : str
        String indicating the cut type ('Thick', 'Thin', 'Med').
    path : str
        Root directory where the spindle and camming folders are located.
    spindle : str
        Name or identifier for the spindle (subfolder inside path).
    files : list of str
        List of file paths containing CAM files and Master.txt to combine.

    Notes
    -----
    - CAM files are renumbered continuously across multiple directories.
    - Master.txt entries are updated accordingly.
    - Headers in CAM files match Aerotech expected format.
    - Works efficiently without recursion.
    """
    name_base = f'CutCam{cut_type}'
    name_suffix = '.Cam'

    while len(files) > 1:
        new_folder = os.path.join(path, spindle, f'CutCamming{cut_type}-concat')

        if not os.path.isdir(new_folder):
            os.makedirs(new_folder)

        # Load master files
        master1 = np.loadtxt(os.path.join(files[0], 'Master.txt'))
        master2 = np.loadtxt(os.path.join(files[1], 'Master.txt'))

        master_all = list(master1)

        start = len(master1)
        add_on = len(master2)

        # Copy CAM files from the first directory (no renumbering needed)
        for entry in master1:
            num = int(entry[0])
            cam_old = os.path.join(files[0], f"{name_base}{num:04d}{name_suffix}")
            cam_new = os.path.join(new_folder, f"{name_base}{num:04d}{name_suffix}")

            temp = np.loadtxt(cam_old, skiprows=4)
            numpts = len(temp)

            with open(cam_new, 'w') as f:
                f.write(f";Filename: {cam_old}\n")
                f.write(f"Number of points {numpts:04d}\n")
                f.write("Master Units (PRIMARY)\n")
                f.write("Slave Units (PRIMARY)")
                for idx, row in enumerate(temp, start=1):
                    f.write(f"\n{idx:04d} {row[1]} {row[2]}")

        # Copy and renumber CAM files from the second directory
        last_line_num = int(master1[-1, 0])
        for idx, entry in enumerate(master2):
            old_num = int(entry[0])
            new_num = last_line_num + 1 + idx

            cam_old = os.path.join(files[1], f"{name_base}{old_num:04d}{name_suffix}")
            cam_new = os.path.join(new_folder, f"{name_base}{new_num:04d}{name_suffix}")

            temp = np.loadtxt(cam_old, skiprows=4)
            numpts = len(temp)

            with open(cam_new, 'w') as f:
                f.write(f";Filename: {cam_old}\n")
                f.write(f"Number of points {numpts:04d}\n")
                f.write("Master Units (PRIMARY)\n")
                f.write("Slave Units (PRIMARY)")
                for idx, row in enumerate(temp, start=1):
                    f.write(f"\n{idx:04d} {row[1]} {row[2]}")

            # Update master file entry
            master_temp = [new_num, entry[1], entry[2], entry[3], entry[4]]
            master_all.append(master_temp)

        # Save the combined Master.txt
        master_all = np.array(master_all)
        np.savetxt(os.path.join(new_folder, 'Master.txt'), master_all, fmt='%04g')

        # Move forward: new folder becomes the first input, drop files[0] and files[1]
        files = [new_folder] + files[2:]


def remove_lines(path, first_line, last_line, cuttype):
    """
    Removes specified CAM files and updates the Master.txt file for a given range of cut lines.

    Parameters
    ----------
    path : str
        Directory path where Master.txt and CAM files are stored.
    first_line : int
        The first line number to keep (inclusive).
    last_line : int
        The last line number to keep (inclusive).
    cuttype : str
        String indicating the cut type ('Thick', 'Thin', 'Med') to build CAM filenames.

    Notes
    -----
    - If a 'lockfile.lock' is present in the directory, the function exits without making changes.
    - Only CAM files corresponding to line numbers within [first_line, last_line] are retained.
    - All other CAM files outside the specified range are deleted.
    - Master.txt is rewritten to only include lines within [first_line, last_line].
    """
    if os.path.isfile(os.path.join(path, 'lockfile.lock')):
        print("Lockfile present")
        return
    else:
        print("Lockfile not present")

    masterpath = os.path.join(path, 'Master.txt')
    masterarray = np.loadtxt(masterpath)
    orig_num_lines = masterarray.shape[0]

    # Rewrite Master.txt with selected lines
    nums = masterarray[first_line:last_line + 1, 0]
    xs = masterarray[first_line:last_line + 1, 1]
    ystarts = masterarray[first_line:last_line + 1, 2]
    zs = masterarray[first_line:last_line + 1, 3]
    yends = masterarray[first_line:last_line + 1, 4]

    with open(masterpath, 'w') as mfileout:
        for num, x, ystart, z, yend in zip(nums, xs, ystarts, zs, yends):
            linenum = f"{int(num):04d}"
            mfileout.write(f"{linenum} {x} {ystart} {z} {yend}\n")

    # Remove old .Cam files outside the [first_line, last_line] range
    lines_to_remove = list(range(0, first_line)) + list(range(last_line + 1, orig_num_lines))

    for i in lines_to_remove:
        camfile = os.path.join(path, f"CutCam{cuttype}{i:04d}.Cam")
        if os.path.exists(camfile):
            os.remove(camfile)


def shiftXZ_nocomp(directory, ftype, xshift, zshift):
    """
    Applies an x and z shift to all .Cam files and the corresponding Master.txt
    in a specified cut camming directory.

    Parameters
    -----------
    directory : str or Path
        The base directory containing the CutCamming folders.
    ftype : str
        One of 'Thick', 'Thin', or 'Med' indicating which folder to process.
    xshift : float
        The amount to shift x positions by.
    zshift : float
        The amount to shift z positions by.

    Raises
    -------
    ValueError
        If ftype is not one of the allowed values.
    RuntimeError
        If a lockfile is present and operation should not proceed.
    """
    directory = Path(directory)

    folder_suffix = {'Thick': 'Thick', 'Thin': 'Thin', 'Med': 'Med'}
    if ftype not in folder_suffix:
        raise ValueError(f"Unsupported ftype: {ftype}. Choose from 'Thick', 'Thin', or 'Med'.")

    suffix = folder_suffix[ftype]
    subfolder = directory / f'CutCamming{suffix}'
    shift_folder = directory / f'CutCamming{suffix}-Noshift'

    camname_prefix = subfolder / f'CutCam{suffix}'
    cin_prefix = shift_folder / f'CutCam{suffix}'

    masterfile_in = shift_folder / 'Master.txt'
    masterfile_out = subfolder / 'Master.txt'
    lockfile = subfolder / 'lockfile.lock'

    # Create output directory if needed
    subfolder.mkdir(parents=True, exist_ok=True)

    if lockfile.exists():
        raise RuntimeError(f"Lockfile present at {lockfile}. Operation aborted.")

    # Load and shift master file
    mfile = np.loadtxt(masterfile_in)
    nums, xs, ys, zs, ystops = mfile.T
    xsout = xs + xshift
    zsout = zs + zshift

    with open(masterfile_out, 'w') as mfile_out:
        for i, num in enumerate(nums):
            linenum = f"{int(num):04d}"
            camfile_in = cin_prefix.with_name(f"{cin_prefix.name}{linenum}.Cam")
            pts = np.loadtxt(camfile_in, skiprows=4)
            yscam = pts[:, 1]
            zscam = pts[:, 2] + zshift

            # This should already be defined in your project
            make_cam_file(camname_prefix, num, xs[i], yscam, zscam)

            mfile_out.write(f"{linenum} {xsout[i]} {ys[i]} {zsout[i]} {ystops[i]}\n")
