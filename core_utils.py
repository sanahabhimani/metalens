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


def make_cam_file(filename, filenum, xval, ys, zs):
    """
    Creates an Aerotech-compatible .Cam file with x, y, and z position data for a given file number.

    The output filename is constructed by appending the 4-digit zero-padded `filenum`
    to the base `filename`, followed by the `.Cam` extension.

    Parameters
    ----------
    filename : Path
        Base path (excluding file number and extension) where the .Cam file will be written.
        For example: Path('.../CutCamThin') will become CutCamThin0023.Cam, etc.
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
    The header is formatted exactly for Aerotech reading compatibility:
    - ;Filename: /full/path/to/file.Cam
    - Number of points ####
    - Master Units (PRIMARY)
    - Slave Units (PRIMARY)
    """
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


def YZ_Calibration_Fit(filepath,bladeradius,minind,maxind,y_guess,z_guess):
    minindex = minind
    maxindex = maxind
    pts =np.loadtxt(filepath, delimiter=',')
    x = pts[minindex:maxindex,0]
    y = pts[minindex:maxindex,1]
    zin = pts[minindex:maxindex,2]
    g = pts[minindex:maxindex,3]
    z = zin + g
    p0 = [y_guess, z_guess]#, bladeradius] #initial guess for y,z,r
    rad = bladeradius
    fitfunc = lambda p, y, z:np.sqrt((y-p[0])**2 + (z-p[1])**2)
    errfunc = lambda p, y, z: fitfunc(p, y, z) - rad
    pout,cov,infodict,mesg,ier = opt.leastsq(errfunc,p0, args=(y,z),full_output=1, ftol=1e-14,xtol=1e-14)
    p2 = pout
    resids = infodict['fvec']
    print('The center in probe coordinates is [y,z] = ',p2)
    fig = plt.figure(figsize=(8,15))
    ax = fig.add_subplot(3,1,1)
    ax.scatter(y,z, c='r')
    circ = plt.Circle((p2[0],p2[1]), radius=rad, fill=True)
    #ax.set_aspect('equal')
    ax.add_patch(circ)
    ax.set_title('Data points and best fit circle')
    ax.set_xlim(min(y)-0.1,max(y)+0.1)
    ax.set_ylim(min(z)-0.01,max(z)+0.01)
    ax = fig.add_subplot(3,1,2)
    ax.plot(resids)
    ax.set_title('Residuals')
    ax.set_xlabel('Point')
    ax.set_ylabel('mm')
    ax = fig.add_subplot(3,1,3)
    ax.hist(resids)
    ax.set_title('Residuals')
    ax.set_xlabel('mm')
    ax.set_ylabel('count')
    plt.show()

    return p2


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


def shiftXZ_alumina_filter(directory,spindle, ftype, Xshift, zshift_fixed, correction_zshift,wear_coeff,exposureval,lastlinecut,firstline,numlines):
    if ftype == 'Thick':
        subfolder = 'CutCammingThick'
        camname = directory+spindle+'/' + subfolder + '/CutCamThick'
        cin = directory+'/'+spindle+'/'  + subfolder + '-Noshift/CutCamThick'
    elif ftype == 'Thin':
        subfolder = 'CutCammingThin'
        camname = directory+spindle+'/'  + subfolder + '/CutCamThin'
        cin = directory+'/'+spindle+'/'  + subfolder + '-Noshift/CutCamThin'
    elif ftype == 'Med':
        subfolder = 'CutCammingMed'
        camname = directory+spindle+'/'  + subfolder + '/CutCamMed'
        cin = directory+spindle+'/'  + subfolder + '-Noshift/CutCamMed'

    masterfilein = directory+spindle+'/'  + subfolder + '-Noshift/Master.txt'
    masterfileout = directory+spindle+'/'  + subfolder + '/Master.txt'

    print(masterfilein)
    print('prinintg')

    masterfilein_actual = directory+spindle+'/'  + subfolder + '-ActualDiameter/Master.txt'

    if os.path.isfile(directory+spindle+'/'  + subfolder+'/lockfile.lock')==True:
        print("lockfile present")
        #return 0
    else:
        print('Lockfile not present')

    #load the master file
    mfile = np.loadtxt(masterfilein)
    nums = mfile[:,0]
    xs = mfile[:,1]
    ys = mfile[:,2]
    zs = mfile[:,3]
    ystops = mfile[:,4]

    mfile_actual = np.loadtxt(masterfilein_actual)
    nums_actual = mfile_actual[:,0]
    xs_actual = mfile_actual[:,1]
    ys_actual = mfile_actual[:,2]
    zs_actual = mfile_actual[:,3]
    ystops_actual = mfile_actual[:,4]

    #calculate the distance cut for each line
    deltays = ystops_actual - ys_actual

    first_index = np.where(xs==xs_actual[0])[0]
    last_index = np.where(xs==xs_actual[-1])[0]

    deltays_mod = np.zeros(len(zs))
    print(len(zs))

    deltays_mod[first_index[0]:last_index[0]+1] = deltays

    if ftype =='Thick':
        bladewearfactor = wear_coeff
        exposure = exposureval
    elif ftype == 'Thin':
        bladewearfactor = wear_coeff
        exposure = exposureval

    wearshiftsarray = deltays_mod*bladewearfactor
    cum_wearshift = np.zeros(len(wearshiftsarray))

    if lastlinecut==0:
        init_exp=0
    else:
        oldshiftsstring = directory+spindle+'/' +'WearshiftValues_'+ftype+'.txt'
        oldshifts = np.loadtxt(oldshiftsstring)
        init_exp = oldshifts[lastlinecut,1]
        renameshiftsstring = 'WearshiftValues_'+ftype+'_upto'+str(lastlinecut)+'.txt'
        os.rename(oldshiftsstring,directory+spindle+'/' +renameshiftsstring)
        if os.path.isdir(directory+spindle+'/' +subfolder)==True:
            os.rename(directory+spindle+'/' +subfolder,directory+'/'+spindle+'/' +subfolder+'_upto'+str(lastlinecut))

    s = init_exp

    lines2cut = np.arange(firstline,firstline+numlines)

    for val in lines2cut:
        #exposure check

        if abs(s)>=exposure:
            print('Blade Limit Line',val)
            lastline = val
            break

        cum_wearshift[val] = s
        s += wearshiftsarray[val]

    plt.figure()
    plt.plot(cum_wearshift)
    plt.xlabel('Line Number')
    plt.ylabel('Z Shift (mm)')
    plt.show()
    #zsout_actual = zs_actual + zshift + cum_wearshift
    #xsout_actual = xs_actual + Xshift

    print("Shifting Lines Now")

    zsout = zs + zshift_fixed +cum_wearshift+correction_zshift
    xsout = xs + Xshift

    wearfilestr = directory+spindle+'/' +'WearshiftValues_'+ftype+'.txt'

    wearfileout = open(wearfilestr,'w')

    for i in range(len(cum_wearshift)):
        linenumber = "%04g"%i
        wearfileout.write(str(linenumber) + ' ' + str(cum_wearshift[i]+correction_zshift) + ' ' +'\n')
    wearfileout.close()

    #open the new masterfile
    #mfileout = open(masterfileout,'w')

    #iterate over each of the filenumbers

    masterfileout = directory+spindle+'/'  + subfolder  + '/Master.txt'
    masterfiledir = directory+spindle+'/'  + subfolder  + '/'
    if os.path.isdir(masterfiledir)==False:
        os.makedirs(masterfiledir)
    else:
        os.rename(masterfiledir,directory+spindle+'/'  + subfolder+'_UpToLine_'+str(lastlinecut))
        os.makedirs(masterfiledir)
    mfileout = open(masterfileout,'w')

    for i,num in enumerate(nums):
        linenum = "%04g"%num
        fname = cin + linenum + '.Cam'#generate the input camfile name
        #generate output camfile name
        cname = directory+spindle+'/'  + subfolder+'/CutCam'+ftype
        pts = np.loadtxt(fname,skiprows=4)
        yscam = pts[:,1]
        zscam = pts[:,2] + zshift_fixed + cum_wearshift[i]+correction_zshift
        make_cam_file(cname,num,xs[i],yscam,zscam) #make the new camfile with
                                                        #the shifted z values

        mfileout.write(linenum + ' ' + str(xsout[i]) + ' ' + str(ys[i]) + ' '+ str(zsout[i]) + ' ' + str(ystops[i]) + '\n')

    mfileout.close()
    remove_lines(directory+spindle+'/' + subfolder+'/',firstline,firstline+numlines-1,ftype)
    return 1


def shiftXZ_nocomp(directory, ftype, xshift, zshift):
    """
    Applies an x and z shift to all .Cam files and the corresponding Master.txt 
    in a specified cut camming directory.

    Parameters:
    -----------
    directory : str or Path
        The base directory containing the CutCamming folders.
    ftype : str
        One of 'Thick', 'Thin', or 'Med' indicating which folder to process.
    xshift : float
        The amount to shift x positions by.
    zshift : float
        The amount to shift z positions by.

    Raises:
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
