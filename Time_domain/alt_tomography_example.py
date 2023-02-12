from __future__ import division
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifft2, fft2
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import rotate
from warnings import warn
import matplotlib.pylab as plt
from matplotlib import cm
from numpy import inf
from skimage.transform import radon, iradon, rescale

def ft(f, axis = 0):
    F = fftshift(fft(fftshift(f), axis = axis));
    return(F)

def ift(F, axis = 0):
    f = fftshift(fft(fftshift(F), axis = axis));
    return(f)
    
def ift2(F):
    f = fftshift(ifft2(fftshift(F)));
    return(f)
    
def ft2(f):
    F = fftshift(fft2(fftshift(f)));
    return(F)
    
def my_imshow(img, dB_scale = [0,0], extent = None):
###############################################################################
#                                                                             #
#  This program displays the processed data in dB.  The brightest point in    #
#  the image is used as the reference and the user can define the scale for   #
#  the intensity range.                                                       #
#                                                                             #
###############################################################################

    #Convert to dB
    img = 10*np.log10(np.abs(img)/np.abs(img).max())
    img[img == -inf] = dB_scale[0]

    #Determine if the image is RGB
    if len(img.shape) != 3:
    
        #Display the image
        if dB_scale == [0,0]:
            plt.imshow(img, cmap=cm.Greys_r, extent = extent)
        else:
            plt.imshow(img, cmap=cm.Greys_r,
                       vmin = dB_scale[0], vmax = dB_scale[-1], extent = extent)
    
    #If the image is RGB                 
    else:
        #Display the image
        if dB_scale == [0,0]:
            img_RGB = (img-img.min())/(img.max()-img.min())
            plt.imshow(img_RGB, extent = extent)
        else:
            img[img<=dB_scale[0]] = dB_scale[0]
            img[img>=dB_scale[-1]] = dB_scale[-1]
            img_RGB = (img-img.min())/(img.max()-img.min())
            plt.imshow(img_RGB, extent = extent)

img = np.load('./phantom.npy')
plt.title("Original")
plt.imshow(img, cmap=plt.cm.Greys_r)
plt.axis('off')

# Radon tranform
theta = np.linspace(0., 180., max(img.shape), endpoint=False)
sinogram = radon(img, theta=theta, circle=True)
m,n = sinogram.shape
x = np.arange(-n/2, n/2)
plt.figure()
plt.title("Radon Transform\n(Sinogram)")
plt.xlabel("Projection angle (deg)")
plt.ylabel("Projection position (pixels)")
plt.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(0, 180, x.min(), x.max()), aspect=180/x.size)
plt.show()

# Projection slice
fx = x/x.size
kx = 2*np.pi*fx
[X,Y] = np.meshgrid(x,x)
F_slices = ft(sinogram, axis = 0)
plt.figure()
plt.title("Fourier Slices")
plt.xlabel("Projection angle (deg)")
plt.ylabel("Spatial Frequency (pixels)$^{-1}$")
plt.imshow(abs(F_slices), cmap=plt.cm.Greys_r,
           extent=(0, 180, fx.min(), fx.max()), aspect=180)
plt.show()

# Filter
[Kx, Ky] = np.meshgrid(kx, kx)
th = theta*np.pi/180
u = 2*abs(Ky)
plt.figure()
plt.title("Rho Filter")
plt.xlabel("Projection angle (deg)")
plt.ylabel("Spatial Frequency (pixels)$^{-1}$")
plt.imshow(abs(u), cmap=plt.cm.Greys_r,
           extent=(0, 180, fx.min(), fx.max()), aspect=180)
plt.show()

# Filtered Slices
F_filt = u*F_slices
plt.figure()
plt.title("Filtered Fourier Slices")
plt.xlabel("Projection angle (deg)")
plt.ylabel("Spatial Frequency (pixels)$^{-1}$")
plt.imshow(abs(F_filt), cmap=plt.cm.Greys_r,
           extent=(0, 180, fx.min(), fx.max()), aspect=180)
plt.show()

# Filtered Radon
sinogram_filt = ift(F_filt).real
plt.figure()
plt.title("Filtered Radon Transform\n(Sinogram)")
plt.xlabel("Projection angle (deg)")
plt.ylabel("Projection position (pixels)")
plt.imshow(sinogram_filt, cmap=plt.cm.Greys_r,
           extent=(x.min(), x.max(), x.min(), x.max()), aspect=1)
plt.show()

#Reconstruction
reconstructed = 0
for i in range(len(theta)):
    t = X * np.cos(th[i]) + Y * np.sin(th[i])
    backprojected = np.interp(t, x, sinogram_filt[:, i],
                                  left=0, right=0)
    reconstructed += backprojected
reconstructed = reconstructed[:,::-1]
plt.figure()
plt.title("Reconstructed Image")
plt.xlabel("x position (pixels)")
plt.ylabel("y position (pixels)")
plt.imshow(reconstructed, cmap=plt.cm.Greys_r,
           extent=(x.min(), x.max(), x.min(), x.max()), aspect=1)
plt.show()
'''
# Alt Reconstruction
alt_reconstruction = iradon(sinogram, output_size = img.shape[0])
plt.figure()
plt.title("Reconstructed Image")
plt.xlabel("x position (pixels)")
plt.ylabel("y position (pixels)")
plt.imshow(alt_reconstruction, cmap=plt.cm.Greys_r,
           extent=(x.min(), x.max(), x.min(), x.max()), aspect=1)
plt.show()
'''