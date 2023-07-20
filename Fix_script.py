import os
import numpy as np
from tqdm import tqdm
import argparse
from FixImage3D import FixImage3d
import plot as pm
import time


"""
====================================================================================================================
Run: 

python Fix_script.py --h5path [--res]  [--save_home]


Example:

python Fix_script.py DataExample\\fused.h5 --res 0

====================================================================================================================

"""

def main():

    parser = argparse.ArgumentParser()

    # directory containing hdf5 data
    parser.add_argument("h5path", help='directory with hdf5 data')

    # specify resolution
    parser.add_argument("--res", type=str, help="can't be larger than 3", nargs='?', default= 0)

    # save home directory
    parser.add_argument('--save_home', type=str, nargs='?', default="")

    # save file option
    parser.add_argument('saveftype', type=str, nargs='?', default="")

    args = parser.parse_args()

    h5path = args.h5path
    res = args.res

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
                        chan = ch,
                        savehome=save_home)

        img = fd.readH5Data(res)

        img_corrected = np.zeros_like(img)
        for i in tqdm(range(len(img)), desc="Converting..."):

            img_corrected[i] = fd.stripe_fix(img[i])
            img_corrected[i] = fd.contrast_fix(img_corrected[i], i)

        if savetiff == True:
            fd.savetif(img, img_corrected)

        if savehdf5 == True:
            fd.savehdf5(img_corrected)

        pm.plot_corrected(img_corrected, save_home, ch)
        pm.plot_original(img, save_home, ch)


if __name__ == '__main__':
    start = time.time()
    main()
    elapsed = time.time() - start
    print("Done. \nruntime = %.5f s" %elapsed)
