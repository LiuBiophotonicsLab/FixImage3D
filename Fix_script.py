import os
import numpy as np
from tqdm import tqdm
import argparse
from FixImage3D import FixImage3d
import plot as pm
import time
from skimage import exposure


"""
====================================================================================================================
This script works for fused HDF5 data (.h5) that contain two channels. 

General Run format: 

    python Fix_script_copy.py --h5path [--res] [--orient] [--save_home] [--saveftype]


Arguments: 
    - h5path (Required) : 
        File directory for the HDF5 file to be corrected
    - res (optional) : 
        Default to be "0" for highest resolution. Other options: "1", "2", "3" for 2x, 4x, 8x downsampled data
    - orient (optional) : 
        Default to be 0 for volume with orientation ZXY. Other option: 1 for vol with orientation XZY.
    - save_home (optional) : 
        Directory for saving corrected files. Default to be the home directory for HDF5 file if not specified.
    - saveftype (optional) : 
        Default to be TIFF and HDF5. Other options: "tiff" or "h5" for saving tiff files only or hdf5 file only.


Run example: With 8x downsampled data, save corrected volume as TIFF files (separate channels) and HDF5 files.

    python Fix_script.py Example\\SampleData_AFM010_4xds.h5

====================================================================================================================
Sarah Chow, 06/2023

"""

def main():

    parser = argparse.ArgumentParser()

    # directory containing hdf5 data
    parser.add_argument("h5path", help='directory with hdf5 data')

    # specify resolution
    parser.add_argument("--res", type=str, help="can't be larger than 3", nargs='?', default= "0")

    # specify orientation
    parser.add_argument("--orient", type = int, nargs='?', default=0)

    # save home directory
    parser.add_argument('--save_home', type=str, nargs='?', default="")

    # save file option
    parser.add_argument('--saveftype', type=str, nargs='?', default="")

    args = parser.parse_args()

    h5path = args.h5path
    res = args.res
    orient = args.orient

    # if save home directory not specified, return the h5file home directory
    if args.save_home == "":
        save_home = os.path.dirname(args.h5path)
    else: 
        save_home = args.save_home

    savetiff = True
    savehdf5 = True
    if args.saveftype == "tiff":
        savehdf5 = False
    elif args.saveftype == "h5":
        savetiff = False

    channel = ["s00", "s01"] 

    for ch in channel:

        print(ch)

        fd = FixImage3d(h5path=h5path, 
                        res=res, 
                        orient = orient,
                        chan = ch,
                        savehome=save_home)

        img = fd.readH5Data(res)
        img = fd.rescale(img)
        p2, p98 = np.percentile(img, (2, 98), axis = (1,2))
        p98_max = p98.max()
        p2_min = p2.min()
        p2_max = p2.max()

        img_corrected = np.zeros_like(img)
        for i in tqdm(range(len(img)), desc="Converting..."):

            img_corrected[i] = fd.stripe_fix(img[i])
            img_corrected[i] = fd.contrast_fix(img_corrected[i], 
                                                p98[i], p2[i], 
                                                p98_max, p2_min, p2_max)

        if savetiff == True:
            fd.savetif(img, img_corrected)

        if savehdf5 == True:
            fd.savehdf5(img_corrected)


if __name__ == '__main__':
    start = time.time()
    main()
    elapsed = time.time() - start
    if elapsed > 60 and elapsed < 3600:
        elapsed = elapsed/60
        print("Done. \nruntime = %.2f min" %elapsed)
    elif elapsed > 3600:
        elapsed = elapsed/3600
        print("Done. \nruntime = %.2f hr" %elapsed)
    else:
        print("Done. \nruntime = %.3f s" %elapsed)
