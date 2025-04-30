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


def get_cam_test_touch(path,spindle,cuttype,noshiftflag,cutdepth,touchdepth,linenumber, wearshift):
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
    midpoint = int(len(camfile[:,0])/2.)
    yval = camfile[midpoint,1]
    zval = camfile[midpoint,2]
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

def get_backup_test_touch(path,spindle,filename,pointnumber,wearshift):
    filepath = path+spindle+'/'+filename
    testtouchesfile = np.loadtxt(filepath)
    xval = testtouchesfile[pointnumber,1]
    yval = testtouchesfile[pointnumber,2]
    zval = testtouchesfile[pointnumber,3]
    touchval = zval+wearshift

    testtouchfolder = path+'TestTouches/'
    testtouchlog = testtouchfolder+'TestTouchLog.txt'
    testtouchfile = testtouchfolder+'TestTouch.txt'

    if os.path.isdir(testtouchfolder)==False:
        os.mkdir(testtouchfolder)
    testtouchlogfile = open(testtouchlog,'a')
    testtouchlogfile.write('"'+spindle +'" '+str(xval)+' '+str(yval)+' '+str(touchval)+' ' +str(time.ctime())+'\n')
    testtouchlogfile.close()
    print('Appending Test Touch Log File:'+' "'+spindle +'" '+str(xval)+' '+str(yval)+' '+str(touchval)+' ' +str(time.ctime()))

    print('Making Temp Test Touch File')
    print('Done')

    testtouchtempfile = open(testtouchfile,'w')
    testtouchtempfile.write('"'+spindle +'" '+str(xval)+' '+str(yval)+' '+str(touchval))
    testtouchtempfile.close()
    return 1

def gen_test_touch_points(path,calfilepath,testtouchmetfile,spindle,touchdepth, bladeradius):
    metdata = np.loadtxt(path+spindle+'/'+testtouchmetfile,delimiter=',')
    xdata = metdata[:,0]
    ydata = metdata[:,1]
    zdata = metdata[:,2]
    probedata = metdata[:,3]
    xoffset,yoffset,zoffset = get_spindle_offsets(calfilepath,spindle)
    xtouch = xdata+xoffset
    ytouch = ydata+yoffset
    ztouch = (zdata+probedata)-zoffset+bladeradius-0.5+touchdepth
    f = open(path+'/'+spindle+'/'+spindle+'_Test_Touches.txt','w')
    for i in range(len(xtouch)):
        numstr = "%04g" %i
        f.write(numstr + ' ' + str(xtouch[i]) + ' ' + str(ytouch[i]) + ' ' + str(ztouch[i])+'\n')
    f.close()
    return 1


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


