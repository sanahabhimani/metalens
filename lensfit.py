import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import optimize as opt
import core_utils as cu



def lensfit(pathname, metrologyfilename, afixed, bfixed, lensparams):
    """
    Fit a lens surface to metrology data using two optimization techniques.

    Parameters
    ----------
    pathname : str
        Directory containing the metrology file.
    metrologyfilename : str
        CSV file with columns: x, y, z, radius.
    afixed, bfixed : float
        Fixed parameters for optimization.
    lensparams : any
        Additional lens parameters required for modeling.

    Returns
    -------
    p : ndarray
        Fitted parameters from method 1.
    p2 : ndarray
        Fitted parameters from method 2.
    cov : ndarray
        Covariance matrix for p.
    infodict : dict
        Optimization metadata for p.
    mesg : str
        Message from optimizer for p.
    ier : int
        Optimizer exit code for p.
    resids : ndarray
        Residuals from method 1.
    qin : ndarray
        Adjusted z-values (zin + r).
    xin, yin : ndarray
        Input x and y positions from metrology file.
    """
    pts = np.loadtxt(pathname + metrologyfilename, delimiter=',')
    xin, yin, zin, r = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    qin = zin + r

    # Starting parameter vector
    p0 = [83.255, 365.741, -40, afixed, bfixed]

    print('Fitting with technique 1')
    p, cov, infodict, mesg, ier = opt.leastsq(
        F, p0, args=(xin, yin, qin, afixed, bfixed, lensparams),
        full_output=1, ftol=1e-14, xtol=1e-14, diag=(1, 1, 1, 10, 10))
    resids = infodict['fvec']

    print('Fitting with technique 2')
    p2, cov2, infodict2, mesg2, ier2 = opt.leastsq(
        FuncNew, p0, args=(xin, yin, qin, afixed, bfixed, lensparams),
        full_output=1, ftol=1e-14, xtol=1e-14, diag=(1, 1, 1, 10, 10))

    print('P1: ', p)
    print('P2: ', p2)
    print('Diff: ', p2 - p)

    return p, p2, cov, infodict, mesg, ier, resids, qin, xin, yin






def lensfit(pathname,spindle,calibrationfilepath,metrologyfilename,flag,lensparams,afixed,bfixed,stepheight,bladeradius,cutdiameter,cutparamsfile,x_rot_shift):
    
    spindledir = spindle+'/'
    
    cutpaththick = pathname+spindledir+'CutCammingThick-Noshift/'
    cutpaththin = pathname+spindledir+'CutCammingThin-Noshift/'
    cutpathmed = pathname+spindledir+'CutCammingMed-Noshift/'

    if flag=='Thick':
        try:
           with open(cutpaththick + 'lockfile.lock') as f:
                print('Lockfile thick present')
                return 'Lockfile present'
        except IOError as e:
           print( 'Lockfile thick not present')
    
    if flag=='Thin':
        try:
           with open(cutpaththin + 'lockfile.lock') as f:
               print('Lockfile thin present')
               return 'Lockfile present'
        except IOError as e:
           print('Lockfile thin not present')
    
    if flag=='Med':
        try:
           with open(cutpathmed + 'lockfile.lock') as f:
               print('Lockfile thin present')
               return 'Lockfile present'
        except IOError as e:
           print('Lockfile thin not present')
        
    if not os.path.isdir(cutpaththick):
       os.makedirs(cutpaththick)
    if not os.path.isdir(cutpaththin):
       os.makedirs(cutpaththin)
    if not os.path.isdir(cutpathmed):
       os.makedirs(cutpathmed)
       
    pts = np.loadtxt(pathname + metrologyfilename, delimiter=',')
    xin = pts[:,0]
    yin = pts[:,1]
    zin = pts[:,2]
    r = pts[:,3]

    qin = zin  + r

    #starting parameter vector
    p0 = [83.255,365.741,-40,afixed,bfixed]

    #fit with the F(x,y) approach---------------------------------------------
    print('Fitting with technique 1')
    p,cov,infodict,mesg,ier = opt.leastsq(F,p0,args=(xin,yin,qin, afixed, bfixed, lensparams),full_output=1, ftol=1e-14, xtol=1e-14,diag=(1,1,1,10,10))
    resids = infodict['fvec']


    #Fit with the "rotate data" approach--------------------------------------
    print('Fitting with technique 2')
    p2,cov2,infodict2,mesg2,ier2 = opt.leastsq(FuncNew,p0,args=(xin,yin,qin, afixed, bfixed, lensparams),full_output=1, ftol=1e-14, xtol=1e-14,diag=(1,1,1,10,10))
    print('P1: ', p)
    print('P2: ', p2)
    print('Diff: ', p2-p)
    #resids2 = infodict2['fvec']


    print('Done fitting')
    #generate model values --------------------------------------------------
    zmodels = np.zeros(len(xin))
    xouts = np.zeros(len(xin))
    youts = np.zeros(len(xin))
    modelresids = np.zeros(len(xin))
    for i,xval in enumerate(xin):
        yval = yin[i]
        xouts[i],youts[i],zmodels[i] = Flrt(p,xval,yval, afixed, bfixed, lensparams)
        xouts[i] = xouts[i] - xin[i]+p[0]
        youts[i] = youts[i]-yin[i]+p[1]
        modelresids[i] = zmodels[i] - qin[i]

    print ('Max deviations: ', max(xouts), max(youts))

    print("\n")

    #Plotting everything (2D)---------------------------------------------
    fig = plt.figure()
    xgrid = np.linspace(xin.min(),xin.max(),300)
    ygrid = np.linspace(yin.min(),yin.max(),300)

    xoutgrid, youtgrid = np.meshgrid(xgrid,ygrid)
    qingrid = interpolate.griddata((xin, yin),qin,(xoutgrid, youtgrid),method='linear')
    residsgrid = interpolate.griddata((xin, yin),resids,(xoutgrid, youtgrid),method='linear')
    zmodelsgrid = interpolate.griddata((xin, yin),zmodels,(xoutgrid, youtgrid),method='linear')
    modelresidsgrid = interpolate.griddata((xin,yin),modelresids,(xoutgrid,youtgrid),method='linear')


    #Plot the data
    ax = fig.add_subplot(1, 3, 1, aspect='equal')
    cax = ax.contourf(xgrid, ygrid,qingrid,40, cmap=cm.jet)
    fig.colorbar(cax)
    ax.scatter(xin,yin,marker='o',c='b',s=1)
    ax.set_title('Data')

    #Plot the F(x,y) residuals
    ax = fig.add_subplot(1, 3, 2, aspect='equal')
    cax = ax.contourf(xgrid, ygrid,residsgrid,40, cmap=cm.jet)
    fig.colorbar(cax)
    ax.set_title('Data')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title('Residual Errors')

    #plot the model surface
    ax = fig.add_subplot(1, 3, 3, aspect='equal')

    cax = ax.contourf(xgrid, ygrid,zmodelsgrid,40, cmap=cm.jet)
    fig.colorbar(cax)
    ax.set_title('Model Values')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    
    plt.savefig(pathname+'Residuals.png',bbox_inches='tight')

    plt.show()#----------------------------------(HERE)


    #Now generate the paths for cutting-------------------------------------------

 
    #Input offset parameters and radii
    thick_depth, med_depth, thin_depth, cutpitch = cu.get_cut_parameters(pathname+cutparamsfile)

    Xcenter = p[0]
    Ycenter = p[1]
    Cutdiam = cutdiameter #DIAMETER OF THE CUT------------------
    Measdiam = 300.00

    
    Xstart = Xcenter - Cutdiam/2.00 + x_rot_shift+0.5
    Xend = Xcenter  + Cutdiam/2.00 

    print("Start and End: ",Xstart, Xend)Xstartmeas = Xcenter - Measdiam/2.00 + 0.500

    pitch = cutpitch #Pitch of Cuts
    Yres = 0.500

    Measpitch = 10.0

    newxoffset, newyoffset, newzoffset = cu.get_spindle_offsets(calibrationfilepath,spindle)

    Xoffset = newxoffset
    Yoffset = newyoffset
    Zoffset = newzoffset
    Radius = bladeradius

    print("Offsets:",Xoffset,Yoffset,Zoffset,"Blade Radius:",Radius)

    print("Cut Parameters:", "Thick Depth=",thick_depth, "Medium Depth=",med_depth,"Thin Depth=", thin_depth, "Pitch=",cutpitch,"Cut Diameter=",cutdiameter)
        
    measrad = 0.500

    MetShift = 50.0

    #Input values for generating the camfiles

    #"Plot" blade parameters---------------HUBBED
    XoffsetPlot = 0.000
    YoffsetPlot = 0.000
    ZoffsetPlot = 0.000
    RadiusPlot = bladeradius
    Plotdepth = 0.000
    
    #Thick blade parameters---------------HUBLESS
    XoffsetThick = newxoffset
    YoffsetThick = newyoffset
    ZoffsetThick = newzoffset
    RadiusThick = bladeradius
    Thickdepth = thick_depth

    #Med blade parameters----------------HUBLESS
    XoffsetMed = XoffsetThick 
    YoffsetMed = YoffsetThick
    ZoffsetMed = ZoffsetThick
    RadiusMed = bladeradius
    Meddepth = thick_depth + med_depth

    #Thin blade parameters---------------HUBLESS
    XoffsetThin = XoffsetThick 
    YoffsetThin = YoffsetThick
    ZoffsetThin = ZoffsetThick
    RadiusThin = bladeradius
    Thindepth = thick_depth + med_depth + thin_depth

    #Generate commands for test touches for all 3 blades, save in different files
    #as complete move commands

    Touchpitch = 25.0
    touchx = np.floor(Touchpitch/pitch) #scaling numbers for test touches, take every nth point
    touchy = np.floor(Touchpitch/Yres)

    #paste in the 3layer generation on a sparse grid
    #Have a second set of test touch files for the rim, dense sampling (every 5 lines) 
    #at the -X edge of the lens
    
       
    #Generate metrology confirmation path (just a vertical shift of 40mm)
    #write out metrology camfile set
    xs = np.arange(Xstartmeas,Xend,Measpitch/2.0)
    z0 = np.zeros(len(xs))

    y0 = np.zeros(len(xs))


    #Make Plots of blade outputs (PlotBlade) [NOT DONE!!!]
    if flag=='plot':
        xs = np.arange(Xstart,Xend,pitch*30)
        z0 = np.zeros(len(xs))
        y0 = np.zeros(len(xs))

        cutmasterfile = open(cutpaththick+'Master.txt','w')

        for j,xx in enumerate(xs):
            Ystart = Ycenter - np.sqrt((Cutdiam/2.0)**2 - (xx - Xcenter)**2)
            Yend = Ycenter + np.sqrt((Cutdiam/2.0)**2 - (xx - Xcenter)**2) + 0.0001

            ys = np.arange (Ystart,Yend,Yres)
            zs = np.zeros(len(ys))
            zsfc = np.zeros(len(ys))
            zball = np.zeros(len(ys))
            youts = np.zeros(len(ys))
            y0[j] = ys[0]
            z0[j] = zs[0]
            ax = plt.gca()
            ax.cla()

            yshires = np.arange(Ystart,Yend,Yres/25.0) #for looking at goodness of linear approx
            youtshires = np.zeros(len(yshires))
            zshires = np.zeros(len(yshires)) #zs for looking
            zsfchires = np.zeros(len(yshires))
            zshilinear = np.zeros(len(yshires)) #for the linear interpolation version

            youtsplane = np.zeros(len(ys))
            zsplane = np.zeros(len(ys))

            for ii,yyy in enumerate(yshires): #evaluate the func at all the hires points
                Fx1,Fy1,Fz1 = gradF(p,xx,yyy, lensparams)
                #print Fx, Fy, Fz
                Fyy1 = Fy1/np.sqrt(Fy1**2 + Fz1**2)
                Fzz1 = Fz1/np.sqrt(Fy1**2 + Fz1**2)
                youtshires[ii] = yyy - RadiusPlot*Fyy1
                zshires[ii] = FlrtNoball(p,xx,yyy,lensparams)[2] - ZoffsetPlot + RadiusPlot*Fzz1 - Plotdepth
                zsfchires[ii] = FlrtNoball(p,xx,yyy,lensparams)[2]
                #zball[i] = Flrt(p,xx,yyy)[2]

            for i,yy in enumerate(ys):
                Fx,Fy,Fz = gradF(p,xx,yy,lensparams)
                Fx,Fy,Fz = gradF(p,xx,yy,lensparams)
                #print Fx, Fy, Fz
                Fyy = Fy/np.sqrt(Fy**2 + Fz**2)
                Fzz = Fz/np.sqrt(Fy**2 + Fz**2)
                youts[i] = yy - RadiusPlot*Fyy
                zs[i] = FlrtNoball(p,xx,yy, lensparams)[2] - ZoffsetPlot + RadiusPlot*Fzz - Plotdepth
                zsfc[i] = FlrtNoball(p,xx,yy, lensparams)[2]
                zball[i] = Flrt(p,xx,yy, afixed, bfixed, lensparams)[2]


                Fx,Fy,Fz = gradFplane(p,xx,youts[i], lensparams,stepheight) #repeat the exercise with the plane only model
                #print Fx, Fy, Fz
                Fyy = Fy/np.sqrt(Fy**2 + Fz**2)
                Fzz = Fz/np.sqrt(Fy**2 + Fz**2)
                youtsplane[i] = youts[i] - RadiusPlot*Fyy
                zsplane[i] = PlaneNoball(p,xx,youts[i],lensparams,stepheight)[2] - ZoffsetPlot + RadiusPlot*Fzz - Plotdepth

                #do the comparison between plane model and surface model and take the higher of the two
                if zsplane[i] > zs[i]:
                    zs[i] = zsplane[i]

                if (i%30 == 0):
                    print('adding circle')
                    c = plt.Circle((youts[i],zs[i]),radius=RadiusPlot, fill=False)
                    ax.add_artist(c)

            ax.plot(youts,zs,'r')
            ax.plot(ys,zsfc,'b')
            ax.plot(ys,zball,'g')
            ax.plot(youtsplane,zsplane,'c')

            print('ys from ',min(ys),max(ys))
            print('youts from ',min(youts),max(youts))
            print('Max y change of an eval point ', max(youts-ys))

            #finding the max difference due to linear interpolation
            zsinterp = np.zeros(len(yshires))
            interpfun = interpolate.interp1d(youts,zs,kind='linear',bounds_error=False)
            for ival, yval in enumerate(youtshires):
                zsinterp[ival] = interpfun(yval)
                if np.isnan(zsinterp[ival]):
                    zsinterp[ival] = zshires[ival]

            zsinterp[0] = zshires[0] #shouldn't need this, but smthg is going wrong...
            zsinterp[-1] = zshires[-1]

            zdiffs = zsinterp-zshires

            print('Max deviation from linear interpolation ', max(abs(zdiffs[25:-25])))

        cutmasterfile.close()
        plt.show()
         #Write out cutting camfile set (Thick) [NOT DONE!!!]
    if flag=='Thick':
        xs = np.arange(Xstart,Xend,pitch)
        z0 = np.zeros(len(xs))
        y0 = np.zeros(len(xs))

        cutmasterfile = open(cutpaththick+'Master.txt','w')

        for j,xx in enumerate(xs):
            Ystart = Ycenter - np.sqrt((Cutdiam/2.0)**2 - (xx - Xcenter)**2)
            Yend = Ycenter + np.sqrt((Cutdiam/2.0)**2 - (xx - Xcenter)**2) + 0.0001

            ys = np.arange (Ystart,Yend,Yres)
            zs = np.zeros(len(ys))
            y0[j] = ys[0]
            z0[j] = zs[0]

            zs = np.zeros(len(ys))
            zsfc = np.zeros(len(ys))
            zball = np.zeros(len(ys))
            youts = np.zeros(len(ys))

            youtsplane = np.zeros(len(ys))
            zsplane = np.zeros(len(ys))

            for i,yy in enumerate(ys):
                Fx,Fy,Fz = gradF(p,xx,yy, lensparams)
                #print Fx, Fy, Fz
                Fyy = Fy/np.sqrt(Fy**2 + Fz**2)
                Fzz = Fz/np.sqrt(Fy**2 + Fz**2)
                youts[i] = yy - RadiusThick*Fyy
                zs[i] = FlrtNoball(p,xx,yy,lensparams)[2] - ZoffsetThick + RadiusThick*Fzz - Thickdepth
                zsfc[i] = FlrtNoball(p,xx,yy,lensparams)[2]
                zball[i] = Flrt(p,xx,yy,afixed,bfixed,lensparams)[2]

                Fx,Fy,Fz = gradFplane(p,xx,youts[i],lensparams,stepheight) #repeat the exercise with the plane only model
                #print Fx, Fy, Fz
                Fyy = Fy/np.sqrt(Fy**2 + Fz**2)
                Fzz = Fz/np.sqrt(Fy**2 + Fz**2)
                youtsplane[i] = youts[i] - RadiusThick*Fyy
                zsplane[i] = PlaneNoball(p,xx,youts[i],lensparams,stepheight)[2] - ZoffsetThick + RadiusThick*Fzz - Thickdepth

                #do the comparison between plane model and surface model and take the higher of the two
                if zsplane[i] > zs[i]:
                    zs[i] = zsplane[i]
                    #youts[i] = youtsplane[i]

            fname = cutpaththick + 'CutCamThick'
            make_cam_file(fname,j,xx+XoffsetThick,youts+YoffsetThick,zs)

            linenum = "%04g" % (j)
            cutmasterfile.write(linenum + ' ' + str(xx+XoffsetThick) + ' ' + str(youts[0]+YoffsetThick)+ ' ' + str(zs[0]) + ' ' + str(youts[-1]+YoffsetThick) + '\n')

        cutmasterfile.close()

        plt.plot(youts,zs+ZoffsetThick,'r') #plot things for last x

        yplotcents = []
         zplotcents = []
        for ii in range(len(zs)):
            if (ii%30 == 0):
                print('adding circle')
                c = plt.Circle((youts[ii],zs[ii]+ZoffsetThick),radius=RadiusThick, fill=False)
                plt.gca().add_artist(c)
                yplotcents.append(youts[ii])
                zplotcents.append(zs[ii]+ZoffsetThick)

        plt.scatter(yplotcents,zplotcents)
        plt.plot(youtsplane,zsplane+ZoffsetThick,'g')
        plt.plot(ys,zsfc,'b')
        plt.show()


    #Write out cutting camfile set (Thin) [NOT DONE!!!]
    if flag=='Thin':

        xs = np.arange(Xstart,Xend,pitch)
        z0 = np.zeros(len(xs))
        y0 = np.zeros(len(xs))

        cutmasterfile = open(cutpaththin+'Master.txt','w')

        for j,xx in enumerate(xs):

            Ystart = Ycenter - np.sqrt((Cutdiam/2.0)**2 - (xx - Xcenter)**2)
            Yend = Ycenter + np.sqrt((Cutdiam/2.0)**2 - (xx - Xcenter)**2) + 0.0001

            ys = np.arange (Ystart,Yend,Yres)
            zs = np.zeros(len(ys))
            y0[j] = ys[0]
            z0[j] = zs[0]

            zs = np.zeros(len(ys))
            zsfc = np.zeros(len(ys))
            zball = np.zeros(len(ys))
            youts = np.zeros(len(ys))

            youtsplane = np.zeros(len(ys))
            zsplane = np.zeros(len(ys))


            for i,yy in enumerate(ys):
                Fx,Fy,Fz = gradF(p,xx,yy,lensparams)
                #print Fx, Fy, Fz
                Fyy = Fy/np.sqrt(Fy**2 + Fz**2)
                Fzz = Fz/np.sqrt(Fy**2 + Fz**2)
                youts[i] = yy - RadiusThin*Fyy
                zs[i] = FlrtNoball(p,xx,yy,lensparams)[2] - ZoffsetThin + RadiusThin*Fzz - Thindepth
                zsfc[i] = FlrtNoball(p,xx,yy,lensparams)[2]
                zball[i] = Flrt(p,xx,yy,afixed,bfixed,lensparams)[2]

                Fx,Fy,Fz = gradFplane(p,xx,youts[i],lensparams,stepheight) #repeat the exercise with the plane only model
                #print Fx, Fy, Fz
                Fyy = Fy/np.sqrt(Fy**2 + Fz**2)
                Fzz = Fz/np.sqrt(Fy**2 + Fz**2)
                youtsplane[i] = youts[i] - RadiusThin*Fyy
                zsplane[i] = PlaneNoball(p,xx,youts[i],lensparams,stepheight)[2] - ZoffsetThin + RadiusThin*Fzz - Thindepth

                #do the comparison between plane model and surface model and take the higher of the two
                if zsplane[i] > zs[i]:
                    zs[i] = zsplane[i]
                    #youts[i] = youtsplane[i]

            fname = cutpaththin + 'CutCamThin'
            make_cam_file(fname,j,xx+XoffsetThin,youts+YoffsetThin,zs)

            linenum = "%04g" % (j)
            cutmasterfile.write(linenum + ' ' + str(xx+XoffsetThin) + ' ' + str(youts[0]+YoffsetThin)+ ' ' + str(zs[0]) + ' ' + str(youts[-1]+YoffsetThin) + '\n')

        cutmasterfile.close()

        plt.plot(youts,zs+ZoffsetThin,'r') #plot things for last x
        plt.plot(youtsplane,zsplane+ZoffsetThin,'g')
        plt.plot(ys,zsfc,'b')

        yplotcents = []
        zplotcents = []
        for ii in range(len(zs)):
            if (ii%30 == 0):
                print('adding circle')
                c = plt.Circle((youts[ii],zs[ii]+ZoffsetThin),radius=RadiusThin, fill=False)
                plt.gca().add_artist(c)
                yplotcents.append(youts[ii])
                zplotcents.append(zs[ii]+ZoffsetThin)

        plt.scatter(yplotcents,zplotcents)

        plt.show()


    #Write out cutting camfile set (Med) [NOT DONE!!!]
    if flag=='Med':

        xs = np.arange(Xstart,Xend,pitch)
        z0 = np.zeros(len(xs))
        y0 = np.zeros(len(xs))

        cutmasterfile = open(cutpathmed+'Master.txt','w')

        for j,xx in enumerate(xs):

            Ystart = Ycenter - np.sqrt((Cutdiam/2.0)**2 - (xx - Xcenter)**2)
            Yend = Ycenter + np.sqrt((Cutdiam/2.0)**2 - (xx - Xcenter)**2) + 0.0001

            ys = np.arange (Ystart,Yend,Yres)
            zs = np.zeros(len(ys))
             y0[j] = ys[0]
            z0[j] = zs[0]

            zs = np.zeros(len(ys))
            zsfc = np.zeros(len(ys))
            zball = np.zeros(len(ys))
            youts = np.zeros(len(ys))

            youtsplane = np.zeros(len(ys))
            zsplane = np.zeros(len(ys))

            for i,yy in enumerate(ys):
                Fx,Fy,Fz = gradF(p,xx,yy,lensparams)
                #print Fx, Fy, Fz
                Fyy = Fy/np.sqrt(Fy**2 + Fz**2)
                Fzz = Fz/np.sqrt(Fy**2 + Fz**2)
                youts[i] = yy - RadiusMed*Fyy
                zs[i] = FlrtNoball(p,xx,yy,lensparams)[2] - ZoffsetMed + RadiusMed*Fzz - Meddepth
                zsfc[i] = FlrtNoball(p,xx,yy,lensparams)[2]
                zball[i] = Flrt(p,xx,yy, afixed, bfixed, lensparams)[2]


                Fx,Fy,Fz = gradFplane(p,xx,youts[i], lensparams,stepheight) #repeat the exercise with the plane only model
                #print Fx, Fy, Fz
                Fyy = Fy/np.sqrt(Fy**2 + Fz**2)
                Fzz = Fz/np.sqrt(Fy**2 + Fz**2)
                youtsplane[i] = youts[i] - RadiusMed*Fyy
                zsplane[i] = PlaneNoball(p,xx,youts[i], lensparams, stepheight)[2] - ZoffsetMed + RadiusMed*Fzz - Meddepth

                #do the comparison between plane model and surface model and take the higher of the two
                if zsplane[i] > zs[i]:
                    zs[i] = zsplane[i]
                    #youts[i] = youtsplane[i]

            fname = cutpathmed + 'CutCamMed'
            make_cam_file(fname,j,xx+XoffsetMed,youts+YoffsetMed,zs)

            linenum = "%04g" % (j)
            cutmasterfile.write(linenum + ' ' + str(xx+XoffsetMed) + ' ' + str(youts[0]+YoffsetMed)+ ' ' + str(zs[0]) + ' ' + str(youts[-1]+YoffsetMed) + '\n')

        cutmasterfile.close()
          plt.plot(youts,zs,'r') #plot things for last x
        plt.plot(youtsplane,zsplane,'g')
        plt.plot(ys,zsfc,'b')
        plt.show()

    return p,cov,infodict,mesg,ier
