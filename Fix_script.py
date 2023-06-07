import os
import numpy as np
from tqdm import tqdm
import argparse
from Fix_depth_stripe import FixImage3d
import plot as pm
import time


"""
====================================================================================================================
Run: 

python .py file -h5path -res -zxy -savehome



Example:

python depthfix\\Fix_script.py W:\\UPenn_Clinical\\OTLS4_NODO_2-24-23_AFM010_well_1\\fused_041223\\fused.h5 --res 3

====================================================================================================================

"""


def process_fix(h5path, 
                chan, 
                res, 
                zxy, 
                save_home):
    """
    Return: 
    - img: 3D array, original volume
    - img_corrected: 3D array, stripe and depth corrected volume
    """

    fd = FixImage3d(h5path=h5path, 
                res=res, 
                zxy=zxy, 
                savehome=save_home)

    print("reading 8x downsampled volume...")
    img_8x = fd.readH5Data(chan[0], 3)

    print("reading selected res volume...")
    img = fd.readH5Data(chan[0], res)

    p2, p98, min, mean_prof, global_max  = fd.calculate_rescale_lim(img_8x, 
                                                        img_length = img.shape[0]
                                                        )

    img_corrected = np.zeros_like(img)
    for i in tqdm(range(len(img)), desc="Converting..."):

        img_corrected[i] = fd.stripe_fix(img[i])
        img_corrected[i] = fd.contrast_fix(img_corrected[i], 
                                           p2[i], 
                                           p98[i], 
                                           min[i],
                                           global_max
                                           )

    tiffname, tiffname_corrected, h5name_corrected = fd.saveFileName(chan[1])

    print("saving..")
    fd.savetif(img, tiffname)
    fd.savetif(img_corrected, tiffname_corrected)
    fd.savehdf5(img_corrected, h5name_corrected, chan[0])

    return img, img_corrected



def main():

    parser = argparse.ArgumentParser()

    # directory containing hdf5 data
    parser.add_argument("h5path", help='directory with hdf5 data')

    # specify resolution
    parser.add_argument("--res", type=str, help="can't be larger than 3", nargs='?', default= 0)

    # specify the axis 
    parser.add_argument('--zxy', type=bool, nargs='?', default= True)

    # save home directory
    parser.add_argument('--save_home', type=str, nargs='?', default="")

    args = parser.parse_args()

    h5path = args.h5path
    res = args.res
    zxy = args.zxy

    # if save home directory, not specified, return the h5file home directory
    if args.save_home == "":
        save_home = os.path.dirname(args.h5path)
    else: 
        save_home = args.save_home

    channel = [("s00", 'nuc'),
               ("s01", 'cyto')] 

    # loop through the two channels
    for chan in channel:

        img, img_corrected = process_fix(h5path,
                                         chan, 
                                         res,
                                         zxy,
                                         save_home)
    

        pm.plot_corrected(img_corrected, save_home, chan[1])
        pm.plot_original(img, save_home, chan[1])


if __name__ == '__main__':
    start = time.time()
    main()
    elapsed = time.time() - start
    print("done. \nruntime = %.5f s" %elapsed)
