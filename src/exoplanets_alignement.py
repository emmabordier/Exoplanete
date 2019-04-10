# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 09:55:58 2018

@author: pierre
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib.dates as mdates
import time
from itertools import izip
#import matplotlib.rcParams

plt.ion()

#matplotlib.rcParams.update({'font.size': 20, 'font.weight':'bold', 'axes.labelsize':22, 'axes.labelweight':'bold', 'axes.titlesize':20, 'axes.titleweight':'bold'})

############################# FUNCTIONS #######################################

def roll_param(array, roll_x, roll_y):
    array = np.roll(array, roll_x, axis = 0)
    array = np.roll(array, roll_y, axis = 1)
    return array

def roll(array):
    roll_x, roll_y = (dim//2 for dim in array.shape)
    array = roll_param(array, roll_x, roll_y)
    return array
    
def round_mask(shape, radius, x_0=None, y_0=None):
    """ Generate a a round mask centered in x_0, y_0.

    Parameters
    ==========
    shape : tuple
        A 2-elements tuple containing the shape of the mask. 
    radius : int
        The radius of the mask

    Returns
    =======
    A np.ndarray of `shape`, where all pixels that are closer than `radius` to
    the center are set at 1, and all other pixels to 0. 

    Example
    =======
    >>> round_mask((4,4), 2)
    array([[ 0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  1.,  1.,  1.],
           [ 0.,  0.,  1.,  0.]])
    """
    if not x_0:
        x_0 = shape[0]//2
    if not y_0:
        y_0 = shape[1]//2 

    mask = np.ndarray(shape)
    i_0 = x_0
    j_0 = y_0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.sqrt((i-i_0)**2 + (j-j_0)**2) <= radius:
                mask[i,j] = 1
            else:
                mask[i,j] = 0
    return mask

def centre(cube):
    """Aligns all the images at the same position
    - cube : 3D array containing the series of fits images. First dimension must be time,
    second and third dimensions must be images x and y.
    """
    s0 = cube[0]
    conj_fft_s0 = np.conj(np.fft.fft2(s0))
    for n in range(1, cube.shape[0]):
        s = cube[n]
        convolution = np.fft.ifft2(
            conj_fft_s0 *
            np.fft.fft2(s))
        convolution = (np.real(convolution))
        max_x, max_y = np.where(convolution == convolution.max()) #cherche les positions ou la valeur est maximale dans convolution
        if len(max_x) != 1 or len(max_y) != 1:
            m = 'Attention {} maxima trouvés pour l’image {}'
            m = m.format(len(max_x), n)
            print(m)
        #print("> Aligning images: {:3.2%}".format((n+1)/cube.shape[0]), end='\r')
        s = roll_param(s, -max_x[0], -max_y[0])
        cube[n,:,:] = s
    return cube
    
def aperture_photom(data, peaks, fwhm):
    """ Compute aperture photometry on data 

    Parameters
    ==========
    data : 
        a 2D np.ndarray containing the *original* image
    peaks : 
        a 2D np.ndarray of booleans marking the position of sources, same dimensions as 'data'
    fwhm : 
        the fwhm (in px) of the PSF

    Return a tuple containing:
        - source (x, y) (in px)
        - intensity (in DN)
        - noise 
        - SN
    """

    frame_size = 8*int(fwhm)

    # Radius of the disk
    r1 = 1.5 * fwhm
    disk = round_mask((frame_size, frame_size), r1)

    # Inner and outer radii of the annulus
    r2 = 2.5 * fwhm
    r3 = 3.5 * fwhm
    corona = round_mask((frame_size, frame_size), r3) \
        - round_mask((frame_size, frame_size), r2)

    surface_ratio = disk.sum() / corona.sum()
    sources_catalog = []

    for i, j in np.array(np.where(peaks)).T:
        tranche = data[i-frame_size//2:i+frame_size//2, j-frame_size//2:j+frame_size//2]
        try:
            flux = np.sum(disk * tranche)
            background = surface_ratio * np.sum(corona * tranche)
            noise = surface_ratio * np.std(corona * tranche)
            intensity = (flux - background)
            SN = flux / noise
            sources_catalog.append(((i, j), intensity, noise, SN))
        except ValueError:
            # This happens when either the disk or annulus contains at least a
            # single nan
            try:
                mask_c = np.isnan(corona * tranche)
                mask_d = np.isnan(disk * tranche)
                if (mask_c.sum() <= 10 and mask_d.sum() <= 2):
                    corona = np.ma.array(corona, mask=mask_c).filled(fill_value=0)
                    disk = np.ma.array(disk, mask=mask_d).filled(fill_value=0)
                    surface_ratio = disk.sum() / corona.sum()
                    flux = np.sum(disk * tranche)
                    background = surface_ratio * np.sum(corona * tranche)
                    noise = surface_ratio * np.std(corona * tranche)
                    intensity = (flux - background)
                    SN = flux / noise
                    sources_catalog.append(((i, j), intensity, noise, SN))
                else:
                    raise ValueError
            except ValueError:
                sources_catalog.append(((i, j), np.nan, np.nan, np.nan))

    return sources_catalog
    
    
############################### MAIN ##########################################
    
if __name__ == '__main__':


  
    dark_m=np.zeros((1351,1689))
    for i in range(1,11):
        dark=fits.open('dark_00'+str(i)+'.fit')
        dark_data=dark[0].data
        dark_m=dark_m+dark_data
    dark_m=dark_m/10

   
    flat_m=np.zeros((1351,1689))
    for i in range(11,21):
        flat=fits.open('flat_0'+str(i)+'.fit')
        flat_data=flat[0].data
        flat_m=flat_m+flat_data
    flat_m=flat_m/10


    fd_m=np.zeros((1351,1689))
    for i in range(21,31):
        fd=fits.open('flat_dark_0'+str(i)+'.fit')
        fd_data=fd[0].data
        fd_m=fd_m+fd_data
    fd_m=fd_m/10


    '''

    
    dark = fits.open('filepath.fit')
    dark_data = dark[0].data
    dark_header = dark[0].header
    dark_exptime = dark_header['EXPTIME']
    
    plt.figure(0)
    plt.clf()
    plt.imshow(dark_data/dark_exptime, interpolation='none', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title('Example of dark image normalized to its integration time')
  
    # Here you need data reduction, science cube creation and peaks and fwhm retrieval #
    


    reduced_cube=np.zeros((59,1351,1689))
    

    for i in range(2,10):
        f=fits.open('capture_00'+str(i)+'.fit')
        hdu=f[0]
        data=hdu.data
        reduced_cube[i-2,:,:]=data

    for i in range(10, reduced_cube.shape[0]):
        f=fits.open('capture_0'+str(i)+'.fit')
        hdu=f[0]
        data=hdu.data
        reduced_cube[i-2,:,:]=data
    
    reduced_cube_m=np.zeros((59,1351,1689))
    k=(flat_m-fd_m)/0.05
    h=k/np.average(k)
    for i in range(reduced_cube.shape[0]):
        reduced_cube_m[i,:,:]=((reduced_cube[i,:,:]/60-dark_m/60) / np.mean(reduced_cube[i,:,:]/60 - dark_m/60)) / h

 
    exoplanet_cube = centre(reduced_cube_m)
    np.save('exoplanet_cube.npy',exoplanet_cube)
    
    '''
    
    
    exoplanet_cube=np.load('exoplanet_cube.npy')
    




    peaks = np.zeros((1351,1689), dtype=bool)
    peaks[973,935]=True
    peaks[157,1301]=True
    peaks[271,297]=True

    
    plt.figure(0)
    plt.clf()
    plt.imshow(exoplanet_cube[48,:,:], interpolation='none', origin='lower', vmin=0.9, vmax=1.1)
    plt.show()

    fwhm = 15
    

    
    star0=[]
    star1=[]
    star2=[]    

    for i in range(exoplanet_cube.shape[0]):
        catalog = aperture_photom(exoplanet_cube[i,:,:], peaks, fwhm)
        star0.append(catalog[0][1])
        star1.append(catalog[1][1])
        star2.append(catalog[2][1])

    '''
    for i in range(len(star0)):
        if star0[i]<0.8:
            star0[i]=1.01
        if star1[i]<0.8:
            star1[i]=1.01
        if star2[i]<0.8:
            star2[i]=1.01
'''


    transit1=(star2/np.mean(star2))/(star0/np.mean(star0))
    transit2=(star2/np.mean(star2))/(star1/np.mean(star1))
       
    for i in range(len(transit2)-2):
        if transit2[i+1]<0.8:
            transit2[i+1]=(transit2[i]+transit2[i+2])/2
        if transit1[i+1]<0.8:
            transit1[i+1]=(transit1[i]+transit1[i+2])/2

 
    transit2_m1=np.mean(transit2[0:30])
    transit1_m1=np.mean(transit1[0:30])  
    moyenne1=(transit2_m1+transit1_m1)/2

    transit2_m2=np.mean(transit2[42:55])
    transit1_m2=np.mean(transit1[42:55])  
    moyenne2=(transit2_m2+transit1_m2)/2

    errorbar1=(np.std(transit2[0:30])/np.sqrt(31)+np.std(transit1[0:30])/np.sqrt(31))/2
    errorbar2=(np.std(transit2[42:55])/np.sqrt(14)+np.std(transit1[42:55])/np.sqrt(14))/2


    #with open("date.txt","w") as file:
    
    dates=[]    
    for i in range(2,10):
        f=fits.open('capture_00'+str(i)+'.fit')
        f_header = f[0].header
        f_date = f_header['DATE-OBS']
        dates.append(f_date)

    for i in range(10, 61):
        f=fits.open('capture_0'+str(i)+'.fit')
        f_header = f[0].header
        f_date = f_header['DATE-OBS']
        dates.append(f_date)

    abs=[]
    abs.append(dates[0])
    abs.append(dates[9])
    abs.append(dates[19])
    abs.append(dates[29])
    abs.append(dates[39])
    abs.append(dates[49])

    plt.figure(1)
    plt.clf() 
    plt.plot(star0/np.mean(star0), 'b-', lw=2)
    plt.plot(star1/np.mean(star1), 'g-', lw=2)
    plt.plot(star2/np.mean(star2), 'r-', lw=2)
    plt.xlabel('Time')
    plt.ylabel('Star light curve in arbitrary units')    
    plt.title('Planetary transit around Qatar-1B')    
   
    plt.figure(2)
    plt.clf()
    plt.xticks([0,10,20,30,40,50],abs, rotation=30)
    plt.plot(transit1, 'k-', lw=2, label='exoplanete/ref1')
    plt.plot(transit2, 'm-', lw=2, label='exoplanete/ref2')
    plt.plot((star1/np.mean(star1))/(star0/np.mean(star0)), 'c-', lw=2, label='ref1/ref2')
    plt.hlines(moyenne1,0,30,'b')
    plt.hlines(moyenne2,42,58,'g')
    plt.errorbar(15,moyenne1,3*errorbar1,ecolor='r')
    plt.errorbar(50,moyenne2,3*errorbar2,ecolor='r')
    plt.xlabel('Time')
    plt.ylabel('Star light curve in arbitrary units')    
    plt.title('Planetary transit around Qatar-1B 12/02/2018') 
    plt.legend()   

    plt.figure(3)
    plt.clf()
    plt.imshow(dark_m/60, interpolation='none', origin='lower', cmap='viridis')
    plt.colorbar()
    
    plt.figure(4)
    plt.clf()
    plt.imshow(flat_m/0.05, interpolation='none', origin='lower', cmap='viridis', vmin=260000, vmax=320000)
    plt.colorbar()

    plt.figure(5)
    plt.clf()
    plt.imshow(fd_m/0.05, interpolation='none', origin='lower', cmap='viridis')
    plt.colorbar()
    
    print('RUN COMPLETED')    

