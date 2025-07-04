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

import core_utils as cu


def find_touch_depth(bladeradius, touchlength):
    """
    Calculates the touch depth for a blade based on the touch length.

    Parameters
    ----------
    bladeradius : float
        Radius of the blade (in mm or same units as touchlength).
    touchlength : float
        Length of the contact (chord) between the blade and the surface.

    Returns
    -------
    float
        The depth of the touch (vertical distance from the blade's edge to the chord line).
    """
    return bladeradius * (1 - np.cos(np.arcsin(touchlength / (2 * bladeradius))))



def get_cam_test_touch(path, spindle, cuttype, noshiftflag, cutdepth, touchdepth, linenumber, wearshift):
    """
    Generates a temporary CAM test touch file and appends to a log using CAM and Master.txt data.

    Parameters
    ----------
    path : str
        Base directory containing spindle and CutCamming folders.
    spindle : str
        Spindle identifier.
    cuttype : str
        Type of cut ('Thick', 'Thin', or 'Med').
    noshiftflag : str
        Either 'Noshift' or '' indicating which CutCamming folder to access.
    cutdepth : float
        Cut depth value to apply to the CAM z-value.
    touchdepth : float
        Additional depth adjustment for touch evaluation.
    linenumber : int
        Line number in Master.txt to evaluate.
    wearshift : float
        Wear compensation shift to apply to the final touch value.
    """
    suffix = f"-{noshiftflag}" if noshiftflag == 'Noshift' else ''
    camnum = f"{int(linenumber):04d}"
    cutfolder = f"CutCamming{cuttype}{suffix}"

    subfolder = os.path.join(path, spindle, cutfolder)
    masterpath = os.path.join(subfolder, 'Master.txt')
    campath = os.path.join(subfolder, f"CutCam{cuttype}{camnum}.Cam")

    # Load master and cam files
    masterfile = np.loadtxt(masterpath)
    xval = masterfile[linenumber, 1]

    camfile = np.loadtxt(campath, skiprows=4)
    midpoint = len(camfile) // 2
    yval = camfile[midpoint, 1]
    zval = camfile[midpoint, 2]

    touchval = zval + cutdepth + touchdepth + wearshift

    # Prepare output directory and files
    testtouchfolder = os.path.join(path, 'TestTouches')
    os.makedirs(testtouchfolder, exist_ok=True)

    testtouchlog = os.path.join(testtouchfolder, 'TestTouchLog.txt')
    testtouchfile = os.path.join(testtouchfolder, 'TestTouch.txt')

    # Append to log
    timestamp = time.ctime()
    log_entry = f'"{spindle}" {xval} {yval} {touchval} {timestamp}\n'
    with open(testtouchlog, 'a') as logfile:
        logfile.write(log_entry)

    print(f"Appending Test Touch Log File: {log_entry.strip()}")

    # Write temporary test file
    print('Making Temp Test Touch File')
    with open(testtouchfile, 'w') as tempfile:
        tempfile.write(f'"{spindle}" {xval} {yval} {touchval}')
    print('Done')


def get_backup_test_touch(path, spindle, filename, pointnumber, wearshift):
    """
    Generates a temporary test touch file from a backup test file and logs the event.

    Parameters
    ----------
    path : str
        Base directory containing the spindle and backup test files.
    spindle : str
        Identifier for the spindle (subfolder inside path).
    filename : str
        Name of the backup test file to read (e.g., 'backup_test_data.txt').
    pointnumber : int
        Index of the touch point to extract from the file.
    wearshift : float
        Wear correction shift to apply to the z-coordinate.

    Notes
    -----
    Writes a log entry and a single-line temporary test file into the `TestTouches/` directory.
    """
    filepath = os.path.join(path, spindle, filename)
    testtouches = np.loadtxt(filepath)

    xval = testtouches[pointnumber, 1]
    yval = testtouches[pointnumber, 2]
    zval = testtouches[pointnumber, 3]
    touchval = zval + wearshift

    testtouchfolder = os.path.join(path, 'TestTouches')
    os.makedirs(testtouchfolder, exist_ok=True)

    testtouchlog = os.path.join(testtouchfolder, 'TestTouchLog.txt')
    testtouchfile = os.path.join(testtouchfolder, 'TestTouch.txt')

    timestamp = time.ctime()
    log_entry = f'"{spindle}" {xval} {yval} {touchval} {timestamp}\n'

    # Append to log file
    with open(testtouchlog, 'a') as logfile:
        logfile.write(log_entry)
    print(f"Appending Test Touch Log File: {log_entry.strip()}")

    # Write temporary touch file
    print('Making Temp Test Touch File')
    with open(testtouchfile, 'w') as tempfile:
        tempfile.write(f'"{spindle}" {xval} {yval} {touchval}')


def gen_test_touch_points(path, calfilepath, testtouchmetfile, spindle, touchdepth, bladeradius):
    """
    Generates test touch points by combining metrology data and spindle calibration offsets.

    Parameters
    ----------
    path : str
        Base directory containing spindle folders and test files.
    calfilepath : str
        Path to the spindle calibration file used for offset corrections.
    testtouchmetfile : str
        Filename containing test touch metrology data (CSV format with 4 columns).
    spindle : str
        Spindle identifier (used as a folder name and filename prefix).
    touchdepth : float
        Additional depth correction to apply to Z.
    bladeradius : float
        Radius of the blade used to calculate actual touch position.

    Notes
    -----
    Writes a file named `<spindle>_Test_Touches.txt` into the spindle's folder,
    with lines in the format:
    line_number x y z
    """
    # Load metrology data
    metdata = np.loadtxt(os.path.join(path, spindle, testtouchmetfile), delimiter=',')
    xdata, ydata, zdata, probedata = metdata[:, 0], metdata[:, 1], metdata[:, 2], metdata[:, 3]

    # Get spindle offsets
    xoffset, yoffset, zoffset = cu.get_spindle_offsets(calfilepath, spindle)

    # Apply offsets and compute touch positions
    xtouch = xdata + xoffset
    ytouch = ydata + yoffset
    ztouch = (zdata + probedata) - zoffset + bladeradius - 0.5 + touchdepth

    # Write to output file
    output_path = os.path.join(path, spindle, f"{spindle}_Test_Touches.txt")
    with open(output_path, 'w') as f:
        for i, (x, y, z) in enumerate(zip(xtouch, ytouch, ztouch)):
            f.write(f"{i:04d} {x} {y} {z}\n")


def rename_tt_lines(input_file, output_file, shift_amount):
    """
    Shift the first column (line number) in a 4-column test touch file by a specified amount,
    formatting the result as a 4-digit zero-padded string (e.g., 0001, 0002, ...).

    Raises an error if any line does not have exactly 4 columns.

    Parameters
    ----------
    input_file : str
        Path to the original test touch file.
    output_file : str
        Path to save the updated file.
    shift_amount : int
        Amount to shift the first column (line number) by.
    """
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for lineno, line in enumerate(f_in, 1):
            parts = line.strip().split()

            if len(parts) != 4:
                raise ValueError(f"Line {lineno} does not have exactly 4 columns: {line.strip()}")

            # Shift and pad the first column
            shifted_number = int(parts[0]) + shift_amount
            parts[0] = f"{shifted_number:04d}"

            f_out.write(' '.join(parts) + '\n')


def get_cam_test_touch_pick_ypoint(path, spindle, cuttype, noshiftflag,
                                    cutdepth, touchdepth, linenumber, wearshift, ypoint):
    """
    Generates a CAM test touch file using a specified y-point index in the CAM file.

    Parameters
    ----------
    path : str
        Base directory containing spindle folders.
    spindle : str
        Spindle identifier.
    cuttype : str
        Type of cut ('Thick', 'Thin', or 'Med').
    noshiftflag : str
        'Noshift' or '' indicating the subfolder path variation.
    cutdepth : float
        Cut depth to apply to the z-coordinate.
    touchdepth : float
        Additional depth for simulating probe touch.
    linenumber : int
        Line number to extract from Master.txt.
    wearshift : float
        Wear compensation to add to final touch value.
    ypoint : int
        Index into CAM file to select a specific y-point (not the midpoint).
    """
    suffix = f"-{noshiftflag}" if noshiftflag == 'Noshift' else ''
    camnum = f"{int(linenumber):04d}"
    subfolder = f"CutCamming{cuttype}{suffix}"

    masterpath = os.path.join(path, spindle, subfolder, 'Master.txt')
    campath = os.path.join(path, spindle, subfolder, f"CutCam{cuttype}{camnum}.Cam")

    # Load data
    masterfile = np.loadtxt(masterpath)
    xval = masterfile[linenumber, 1]

    camfile = np.loadtxt(campath, skiprows=4)
    yval = camfile[ypoint, 1]
    zval = camfile[ypoint, 2]

    touchval = zval + cutdepth + touchdepth + wearshift

    # Prepare output paths
    testtouchfolder = os.path.join(path, 'TestTouches')
    os.makedirs(testtouchfolder, exist_ok=True)

    testtouchlog = os.path.join(testtouchfolder, 'TestTouchLog.txt')
    testtouchfile = os.path.join(testtouchfolder, 'TestTouch.txt')

    timestamp = time.ctime()
    log_entry = f'"{spindle}" {xval} {yval} {touchval} {timestamp}\n'

    # Append to log
    with open(testtouchlog, 'a') as logfile:
        logfile.write(log_entry)

    print(f"Appending Test Touch Log File: {log_entry.strip()}")

    # Write temporary test file
    print('Making Temp Test Touch File')
    with open(testtouchfile, 'w') as tempfile:
        tempfile.write(f'"{spindle}" {xval} {yval} {touchval}')

