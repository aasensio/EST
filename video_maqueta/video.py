import numpy as np
import h5py
import sys
import scipy.io
import pyfftw
from astropy import units as u
import matplotlib.pyplot as pl
from ipdb import set_trace as stop
from soapy import confParse, SCI, atmosphere
import matplotlib.animation as manimation
import scipy.ndimage.filters
from tqdm import tqdm

def progressbar(current, total, text=None, width=30, end=False):
    """Progress bar
    
    Args:
        current (float): current value of the bar
        total (float): total of the bar
        text (string): additional text to show
        width (int, optional): number of spaces of the bar
        end (bool, optional): end character
    
    Returns:
        None: None
    """
    bar_width = width
    block = int(round(bar_width * current/total))
    text = "\rProgress {3} : [{0}] {1} of {2}".\
        format("#"*block + "-"*(bar_width-block), current, total, text)
    if end:
        text = text +'\n'
    sys.stdout.write(text)
    sys.stdout.flush()

def generate_video(input_seeing, input_corrected, output, n_anisoplanatic=2):

    pixel_size_km = 48.0

# load a sim config that defines lots of science cameras across the field
    config = confParse.loadSoapyConfig(input_seeing)

    config_corr = confParse.loadSoapyConfig(input_corrected)

    weights = np.zeros((n_anisoplanatic**2,256,256))
    delta = int(256 / n_anisoplanatic)
    loop = 0
    for i in range(n_anisoplanatic):
        for j in range(n_anisoplanatic):
            weights[loop,i*delta:(i+1)*delta,j*delta:(j+1)*delta] = 1.0

    for j in range(4):
        weights[j,:,:] = scipy.ndimage.filters.gaussian_filter(weights[j,:,:], 3)

    weights = weights / np.sum(weights,axis=0)[None,:,:]

    f = scipy.io.readsav('/scratch1/int_48h1_1783.save')
    im = f['int'][0,0:256,0:256]
    im_fft = pyfftw.interfaces.numpy_fft.fft2(im)

# Init a science camera
    sci_cameras = []
    for s in range(config.sim.nSci):
        sci_cameras.append(SCI.PSF(config, s, mask=np.ones((154,154))))

# Init a science camera
    sci_cameras_corr = []
    for s in range(config_corr.sim.nSci):
        sci_cameras_corr.append(SCI.PSF(config_corr, s, mask=np.ones((154,154))))

# init some atmosphere
    atmos = atmosphere.atmos(config)
    atmos_corr = atmosphere.atmos(config_corr)

    n_frames = 200

    f, ax = pl.subplots(ncols=2, nrows=1, figsize=(6,4), dpi=200)

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
    writer = FFMpegWriter(codec='libx264', fps=10, bitrate=15000, metadata=metadata)
    with writer.saving(f, '{0}.mp4'.format(output), n_frames):

# Get phase for this time step
        for j in tqdm(range(n_frames)):
            phase_scrns = atmos.moveScrns()
            phase_scrns_corr = atmos_corr.moveScrns()

    # Calculate all the PSF for this turbulence
            psf = [None] * 4
            images = np.zeros((4,256,256))
            for k in range(4):
                psf[k] = sci_cameras[k].frame(phase_scrns)
            
                nx_psf, ny_psf = psf[k].shape
                psf_roll = np.roll(psf[k].data, int(nx_psf/2), axis=0)
                psf_roll = np.roll(psf_roll, int(ny_psf/2), axis=1)
                
                psf_fft = pyfftw.interfaces.numpy_fft.fft2(psf_roll)

                images[k,:,:] = np.real(pyfftw.interfaces.numpy_fft.ifft2(psf_fft * im_fft))

            images_corr = np.zeros((4,256,256))
            for k in range(4):
                psf[k] = sci_cameras_corr[k].frame(phase_scrns_corr)
            
                nx_psf, ny_psf = psf[k].shape
                psf_roll = np.roll(psf[k].data, int(nx_psf/2), axis=0)
                psf_roll = np.roll(psf_roll, int(ny_psf/2), axis=1)
                
                psf_fft = pyfftw.interfaces.numpy_fft.fft2(psf_roll)

                images_corr[k,:,:] = np.real(pyfftw.interfaces.numpy_fft.ifft2(psf_fft * im_fft))

            image_final = np.sum(images * weights, axis=0)
            image_corr_final = np.sum(images_corr * weights, axis=0)
                        
            ax[0].imshow(image_final, extent=[0,256*48.0/1e3,0,256*48.0/1e3])
            ax[1].imshow(image_corr_final, extent=[0,256*48.0/1e3,0,256*48.0/1e3])

            ax[0].get_xaxis().set_visible(False)
            ax[0].get_yaxis().set_visible(False)
            ax[1].get_xaxis().set_visible(False)
            ax[1].get_yaxis().set_visible(False)
            # ax[0].set_xlabel('x [Mm]', fontsize=10)
            # ax[0].set_ylabel('y [Mm]', fontsize=10)
            # ax[1].set_xlabel('x [Mm]', fontsize=10)
            # ax[1].set_ylabel('y [Mm]', fontsize=10)
            # ax[0].set_title('Original', fontsize=10)
            # ax[1].set_title('MCAO corrected', fontsize=10)
            # ax[0].tick_params(labelsize=10)
            # ax[1].tick_params(labelsize=10)

            writer.grab_frame()

            f.savefig('frames/{0}_{1:04d}.tif'.format(output,j), format='tiff', dpi=600)

            for i in range(2):
                ax[i].cla()
    

#     f_images.close()
#     f_images_validation.close()

if (__name__ == '__main__'):
    generate_video('sh_8x8.yaml', 'sh_8x8_corrected.yaml', 'ao_est')
    # generate_video('sh_8x8.yaml', 'sh_8x8_nosocorrected.yaml', 'ao_est_worse')