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
    xoffset, yoffset, zoffset = get_spindle_offsets(calfilepath, spindle)

    # Apply offsets and compute touch positions
    xtouch = xdata + xoffset
    ytouch = ydata + yoffset
    ztouch = (zdata + probedata) - zoffset + bladeradius - 0.5 + touchdepth

    # Write to output file
    output_path = os.path.join(path, spindle, f"{spindle}_Test_Touches.txt")
    with open(output_path, 'w') as f:
        for i, (x, y, z) in enumerate(zip(xtouch, ytouch, ztouch)):
            f.write(f"{i:04d} {x} {y} {z}\n")


def get_cam_test_touch_pick_ypoint(path,spindle,cuttype,noshiftflag,cutdepth,touchdepth,linenumber,wearshift,ypoint):
    if noshiftflag == 'Noshift':
        pathadd = '-Noshift'
    else:
        pathadd = ''
    
    if cuttype == 'Thick':
        camnum = "%04g"%linenumber
        subfolder = 'CutCammingThick'+pathadd+'/'
        masterpath = path+spindle+'/'+ subfolder + 'Master.txt'
        campath = path+spindle+'/'+ subfolder + 'CutCamThick'+camnum+'.Cam'
    elif cuttype == 'Thin':
        camnum = "%04g"%linenumber
        subfolder = 'CutCammingThin'+pathadd+'/'
        masterpath = path+spindle+'/'+ subfolder + 'Master.txt'
        campath = path+spindle+'/'+ subfolder + 'CutCamThin'+camnum+'.Cam'
    elif cuttype == 'Med':
        camnum = "%04g"%linenumber
        subfolder = 'CutCammingMed'+pathadd+'/'
        masterpath = path+spindle+'/'+ subfolder + 'Master.txt'
        campath = path+spindle+'/'+ subfolder + 'CutCamMed'+camnum+'.Cam'
    masterfile = np.loadtxt(masterpath)
    xval = masterfile[linenumber,1]
    camfile = np.loadtxt(campath,skiprows=4)
    yval = camfile[ypoint,1]
    zval = camfile[ypoint,2]
    touchval = zval+cutdepth+touchdepth+wearshift

    testtouchfolder = path+'TestTouches/'
    testtouchlog = testtouchfolder+'TestTouchLog.txt'
    testtouchfile = testtouchfolder+'TestTouch.txt'
    if os.path.isdir(testtouchfolder)==False:
        os.mkdir(testtouchfolder)
    testtouchlogfile = open(testtouchlog,'a')
    testtouchlogfile.write('"'+spindle +'" '+str(xval)+' '+str(yval)+' '+str(touchval)+' ' +str(time.ctime())+'\n')
    testtouchlogfile.close()
    print('Appending Test Touch Log File:'+' '+spindle +' '+str(xval)+' '+str(yval)+' '+str(touchval)+' ' +str(time.ctime()))

    print('Making Temp Test Touch File')
    print('Done')

    testtouchtempfile = open(testtouchfile,'w')
    testtouchtempfile.write('"'+spindle +'" '+str(xval)+' '+str(yval)+' '+str(touchval))
    testtouchtempfile.close()


    return 1


