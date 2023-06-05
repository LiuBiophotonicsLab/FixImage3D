import numpy as np
import h5py as h5
import numpy as np
from tqdm import tqdm
from skimage import exposure
import os
#from scipy import signal as sp

class FixImage3d(object):
    def __init__(self,
                 h5path,
                 res,
                 zxy,
                 savehome
                ):

        self.h5path = h5path
        self.res = res
        self.zxy = zxy
        self.savehome = savehome

        self.sample_name = os.path.basename(h5path).split(".h5")[0]

    
    def readH5Data(self, chan, res):
        """
        just to read .h5 file, with specific resolution and axis index
        """

        res = str(res)
        with h5.File(self.h5path, 'r') as f:
            img = f['t00000'][chan][res]['cells'][:,:,:].astype(np.uint16)
            if self.zxy == False:
                img = np.moveaxis(img, 0, 1)

        f.close()

        # in case the z levels are not cropped well
        zstart, zend = int(len(img)*0.04), int(len(img)*0.98)

        return img[zstart:zend]


    def saveFileName(self, chan):
        tiffname = \
                    (self.savehome + os.sep + self.sample_name + "_" + 
                    chan + ".tif")
        tiffname_corrected = \
                    (self.savehome + os.sep + self.sample_name + "_" + 
                    chan + "_corrected" + ".tif")

        h5name_corrected = \
                    (self.savehome + os.sep + self.sample_name + "_" +
                    "corrected" + ".h5")

        return tiffname, tiffname_corrected, h5name_corrected


    def savetif(self, img_3d, fname):
        import tifffile as tf
        img_3d = np.moveaxis(img_3d, 0, 2)

        tf.imwrite(fname, img_3d, photometric='rgb')


    def savehdf5(self, img_3d, fname, chan):
        """
        Save corrected img as .h5 file accroding to this format f['t00000/s01/1/cells'].

        Args: 
        - img_3d : stripe and depth corrected 3D volume 
        - fname : name of the .h5 file
        - chan : channel 

        """
        from skimage import transform

        hf = h5.File(fname, 'w')
        tgroup = hf.create_group('/t00000')
        sgroup = hf.create_group('/' + str(chan))
        

        res_list = [1, 2, 4, 8, 16]

        for z in range(len(res_list)):
            res = res_list[z]

            if res > 1: 
                img_3d = transform.downscale_local_mean(img_3d, 
                                                    (2, 2, 2)
                                                    ).astype("uint16")

            # resgroup = hf.create_group('/t00000/' + str(chan) + '/' + str(z))
            # resgroup.create_dataset(res, data=img_3d)

            keys = 't00000/' + str(chan) + '/' + str(z) + '/cells'
            hf.create_dataset(keys, data = img_3d)
            print(chan)




    def getBackgroundLevels(self, image, threshold=50):
        """
        Calculate foreground and background values based on image
        statistics, background is currently set to be 20% of foreground.

        Params:
        - image : 2D numpy array
        - threshold : int, threshold above which is counted as foreground

        Returns:
        - hi_val : int, Foreground values
        - background : int, Background value
        """

        image_DS = np.sort(image, axis=None)
        foreground_vals = image_DS[np.where(image_DS > threshold)]
        hi_val = foreground_vals[int(np.round(len(foreground_vals)*0.95))]
        background = hi_val/5

        return hi_val, background


    def stripe_fix(self, img):
        """
        Fix the vertical striping effect;
        dividing each row in the image by the sum of binning all columns.

        Params: 
        - img : 2D numpy array
        
        Returns: 
        - img_nobg : stripe-fixed image
        """
        
        # Get background
        img_background = self.getBackgroundLevels(img)[1]
        img_nobg = np.clip(img - 0.5*img_background, 0, 2**16)

        #### Calculate profiles with background removed ####
        line_prof_n_nobg = np.zeros((img_nobg.shape[1]),dtype = np.float64) 
        
        ## Grab horizontal line profile
        for col in range(img_nobg.shape[1]): ## Iterate through every column in image
            line_prof_n_nobg[col] = np.sum(img_nobg[:, col].astype(np.float64)) 

        ## Normalize line profile
        line_prof_n_nobg = line_prof_n_nobg/np.max(line_prof_n_nobg)

        for row in range(img_nobg.shape[0]):
            img_nobg[row, :] = img_nobg[row, :]/line_prof_n_nobg
        
        img_nobg[img_nobg<0] = 0
        return img_nobg.astype(np.uint16)


    def calculate_rescale_lim(self, img_8x, img_length):
        """
        calculate the p2 and p98, min and mean for 8x downsampled 3D image, 
        the p2 and p98 for highest resolution image.

        Params: 
        - img8x: 3D numpy array, 8x downsampled volume
        - img_length: int, length of z axis for highest resolution data
        
        Return: 
        - p2: 1D array, 2% min for highest res volume interpolated from 8x 
            downsampled volume
        - p98: 1D array, 98% max for highest res volume interpolated from 8x 
            downsampled volume
        - min: 1D array, min for highest res volume interpolated from 8x 
            downsampled volume 
        """

        # fix striping for 8x downsample data first
        for i in range(len(img_8x)):
            img_8x[i] = self.stripe_fix(img_8x[i]) 
            img_8x[i] = self.gamma_correction(img_8x[i])

        # find the metricsfor the stipe fix 8x downsample data
        p2, p98 = np.percentile(img_8x,
                                (2, 98), 
                                axis = (1,2)
                                # ,method='linear'
                                )

        global_max = p98.max()
        mean = img_8x.mean(axis = (1,2))
        mean_prof = mean/mean.max()
        min = img_8x.min(axis = (1,2))

        # interpolate to the shape of specified res data
        n = img_8x.shape[0]
        x = np.linspace(1, n, n)
        xvals = np.linspace(1, n , img_length)
        p2 = np.interp(xvals, x, p2)
        p98 = np.interp(xvals, x, p98)
        min = np.interp(xvals, x, min)
        mean_prof = np.interp(xvals, x, mean_prof)

        return p2, p98, min, mean_prof, global_max


    def mean_correction(self, img, mean_prof):
        """
        img: 2D image for the layer of interest
        mean_prof: float of the layer of interest of 1D array
        """

        img = img/mean_prof
        img[img>2**16-1] = 2**16-1

        return img


    def gamma_correction(self, img, gamma = 0.7):
        """
        img: 2D image
        gamma: around 0.5 to 0.8
        """

        img = exposure.adjust_gamma(img, gamma)

        return img


    def contrast_fix(self, 
                     img, 
                     p2, 
                     p98, 
                     min, 
                     global_max):
        """
        Rescale the p2 and p98 in the 2D image to out_range

        Params: 
        - img: 2D image for layer of interest
        - p2: 2% min for that layer
        - p98: 98% max for that layer
        - min: min for that layer
        - global_max: the max intensity for the 3D volume

        Return:
        - img_rescale: 2D image for that layer, contrast rescaled
        """
        # img = img - min
        img = self.gamma_correction(img, 0.75)
        img_rescale = exposure.rescale_intensity(img, 
                                                in_range=(p2*0.95, p98*1.1), 
                                                out_range = (0, global_max) # maybe just 0 - 15000? p98 at the brightest layer
                                                )

        

        return img_rescale.astype(np.uint16)



