import numpy as np
import h5py as h5
from skimage import exposure
from skimage import transform
import os


class FixImage3d(object):
    """
    A class for fixing and processing 3D image data.

    Attributes:
        h5path (str): Path to the HDF5 data file.
        res (int): The resolution of the data, i.e. "0", "1", "2", or "3"
        orient (int) : The orientation of the data, 0 for ZXY, 1 for XZY .
        chan (str): The channel identifier for the data, i.e. "s00", "s01"
        savehome (str): Directory path to save the processed data.
        sample_name (str): The sample name extracted from the HDF5 data file.
        p2 (float): The 2% minimum value for rescaling the image data.
        p98 (float): The 98% maximum value for rescaling the image data.
        global_max (float): The maximum intensity value for rescaling the image data.
        tiffname (str): The name of the TIFF file to save the original image.
        tiffname_corrected (str): The name of the TIFF file to save the corrected image.
        h5name_corrected (str): The name of the HDF5 file to save the corrected image.

    """

    def __init__(self,
                 h5path,
                 res,
                 orient,
                 chan,
                 savehome
                ):

        self.h5path = h5path
        self.res = res
        self.orient = orient
        self.chan = chan
        self.savehome = savehome
        self.sample_name = os.path.basename(h5path).split(".h5")[0]

        #################### Initialization #######################
        """
        Read 8x downsampled volume to calculate for the p2, p98 and global max for contrast fixing.
        Create file names to be saved for .tiff and .h5.
        """

        self.tiffname, self.tiffname_corrected, self.h5name_corrected = self.saveFileName()

    
    def readH5Data(self, res):
        """
        Read .h5 file with specific resolution and axis index.

        Args:
        - res (int): The resolution to read.

        Returns:
        - img (np.ndarray): The 3D numpy array with the specified resolution.
        """

        print("Reading data...")
        res = str(res)

        with h5.File(self.h5path, 'r') as f:
            img = f['t00000'][self.chan][res]['cells'][:,:,:].astype(np.uint16)
            if self.orient == 1:
                img = np.moveaxis(img, 0, 1)
        f.close()

        return img


    def saveFileName(self):
        """
        Create save file names and save them among the class.

        Returns:
        - tiffname (str): The name of the TIFF file.
        - tiffname_corrected (str): The name of the corrected TIFF file.
        - h5name_corrected (str): The name of the corrected .h5 file.
        """

        if self.chan == "s00":
            chan = "nuc"
        elif self.chan == "s01":
            chan = "cyto"

        tiffname = \
                    (self.savehome + os.sep + self.sample_name + "_" + 
                    chan + ".tif")
        tiffname_corrected = \
                    (self.savehome + os.sep + self.sample_name + "_" + 
                    chan + "_corrected1" + ".tif")

        h5name_corrected = \
                    (self.savehome + os.sep + 
                    self.sample_name + 
                    "_" +
                    "corrected1" + ".h5")

        return tiffname, tiffname_corrected, h5name_corrected


    def savetif(self, img3d, img3d_corrected):
        """
        Save 3D image data as TIFF files.

        Args:
        - img3d (np.ndarray): The original 3D image data.
        - img3d_corrected (np.ndarray): The corrected 3D image data.
        """

        import tifffile as tf
        print("Saving TIFF...")
        with tf.TiffWriter(self.tiffname) as tif:
            for i in range(len(img3d)):
                tif.write(img3d[i], contiguous=True)

        with tf.TiffWriter(self.tiffname_corrected) as tif:
            for i in range(len(img3d)):
                tif.write(img3d_corrected[i], contiguous=True)


    def savehdf5(self, img_3d, ind1 = 0):
        """
        Save corrected image as .h5 file according to the specified format.

        Args:
        - img_3d (np.ndarray): The stripe and depth-corrected 3D volume.
        - ind1 (int): Index for saving.

        Returns:
        - None
        """

        print("Saving HDF5...")

        if self.chan == "s00":
            self.shape = img_3d.shape
            self.h5init(self.h5name_corrected)

        with h5.File(self.h5name_corrected, 'a') as f:

            res_list = [1, 2, 4, 8]

            for z in range(len(res_list)):
                res = res_list[z]

                if res > 1: 
                    img_3d = transform.downscale_local_mean(img_3d, 
                                                            (2, 2, 2)
                                                            ).astype("uint16")

                if ind1 == 0:
                    ind1_r = ind1
                else:
                    ind1_r = np.ceil((ind1 + 1)/res - 1)

                data = f['/t00000/' + str(self.chan) + '/' + str(z) + '/cells']
                data[int(ind1_r):int(ind1_r+img_3d.shape[0])] = img_3d.astype('int16')

        f.close()


    def h5init(self, dest):
        """
        Initialize and create HDF5 dataset.

        Args:
        - dest (str): The destination path for the HDF5 dataset.
        """

        with h5.File(dest, 'a') as f:

            res_list = [1, 2, 4, 8]

            res_np = np.zeros((len(res_list), 3), dtype='float64')
            res_np[:, 0] = res_list
            res_np[:, 1] = res_list
            res_np[:, 2] = res_list

            tgroup = f.create_group('/t00000')

            for ch in range(2):

                idx = ch

                sgroup = f.create_group('/s' + str(idx).zfill(2))
                resolutions = f.require_dataset('/s' + str(idx).zfill(2) + '/resolutions',
                                                dtype='float64',
                                                shape=(res_np.shape),
                                                data=res_np)

                for z in range(len(res_list)-1, -1, -1):

                    res = res_list[z]

                    resgroup = f.create_group('/t00000/s' + str(idx).zfill(2) + '/' + str(z))

                    data = f.require_dataset('/t00000/s' + str(idx).zfill(2) + '/' + str(z) + '/cells',
                                             dtype='int16',
                                             shape=np.ceil(np.divide([self.shape[0], self.shape[1], self.shape[2]],
                                                                     res))
                                             )
        f.close()


    def getBackgroundLevels(self, image, threshold=50):
        """
        Calculate foreground and background values based on image statistics.

        Args:
        - image (np.ndarray): The 2D numpy array of the image.
        - threshold (int, optional): Threshold above which is counted as foreground.

        Returns:
        - hi_val (int): Foreground values.
        - background (int): Background value.
        """

        image_DS = np.sort(image, axis=None)
        foreground_vals = image_DS[np.where(image_DS > threshold)]
        hi_val = foreground_vals[int(np.round(len(foreground_vals)*0.95))]
        background = hi_val/5

        return hi_val, background


    def stripe_fix(self, img):
        """
        Fix the vertical striping effect in the image.

        Args:
        - img (np.ndarray): The 2D numpy array of the image.

        Returns:
        - img_nobg (np.ndarray): The stripe-fixed image.
        """

        # Calculate profiles with background removed 
        try:
            img_background = self.getBackgroundLevels(img)[1]
        except IndexError:
            img_background = 0
        img_nobg = np.clip(img - 0.5*img_background, 0, 2**16-1)
        line_prof_n_nobg = img_nobg.sum(axis=0)
        line_prof_n_nobg = line_prof_n_nobg/np.max(line_prof_n_nobg)
        line_prof_n_nobg[line_prof_n_nobg == 0] = 1

        # Divide the 2D image with the horizontal line profile
        img_nobg /= line_prof_n_nobg[np.newaxis, :]
        img_nobg = np.clip(img_nobg, 0, 2**16-1)

        return img_nobg.astype(np.uint16)


    def mean_correction(self, img, mean_prof):
        """
        Apply mean correction to the image.

        Args:
        - img (np.ndarray): The 2D image for the layer of interest.
        - mean_prof (float): The mean value for the layer of interest or 1D array.

        Returns:
        - img (np.ndarray): The corrected 2D image.
        """

        img = img/mean_prof
        img[img>2**16-1] = 2**16-1

        return img


    def gamma_correction(self, img, gamma = 0.7):
        """
        Apply gamma correction to the image.

        Args:
        - img (np.ndarray): The 2D image.
        - gamma (float, optional): Gamma value around 0.5 to 0.8.

        Returns:
        - img (np.ndarray): The gamma-corrected 2D image.
        """

        img = exposure.adjust_gamma(img, gamma)

        return img


    def contrast_fix(self, img, p98, p2, p98_max, p2_min, p2_max):
        """
        Rescale the p2 and p98 in the 2D image to the out_range.

        Args:
        - img (np.ndarray): The 2D image for the layer of interest.
        - i (int): Index for current layer.

        Returns:
        - img_rescale (np.ndarray): The rescaled 2D image for that layer.
        """  
        
        img_rescale = np.clip((img-p2+p2_max)*(p98_max-p2_min)/(p98-p2),
                              0, 
                              2**16-1)
        
        return img_rescale.astype(np.uint16)

    def rescale(self, img):
        img_rescale = np.zeros_like(img)
        img_min = img.min()
        img_max = img.max()
        for i in range(len(img)):
            img_rescale[i] = (img[i] - img_min)/(img_max - img_min)*10000
        return img_rescale.astype(np.uint16)



