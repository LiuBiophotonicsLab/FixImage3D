import os
import numpy as np
from matplotlib import pyplot as plt
import numpy as np



def plot_corrected(img, save_home, ch):
    mean = img.mean(axis = (1,2))
    # p2, p98 = np.percentile(img,
    #                         (2, 98), 
    #                         axis = (1,2)
    #                         # ,method='linear'
    #                         )

    fname = save_home + os.sep + "mean_" + ch + "_corrected.png"

    ylim2 = mean.max()*1.05
    
    n = len(mean)
    plt.plot(np.linspace(1,n,n),mean)
    plt.title('Mean across depth after correction')
    plt.xlabel('depth level')
    plt.ylabel('Intensity')
    plt.ylim([0, ylim2])

    plt.savefig(fname)
    plt.close()

def plot_original(img, save_home, ch):
    mean = img.mean(axis = (1,2))
    # p2, p98 = np.percentile(img,
    #                         (2, 98), 
    #                         axis = (1,2)
    #                         # ,method='linear'
    #                         )

    fname = save_home + os.sep + "mean_" + ch + "_.png"

    ylim2 = mean.max()*1.05
    
    n = len(mean)
    plt.plot(np.linspace(1,n,n),mean)
    plt.title('Mean across depth before correction')
    plt.xlabel('depth level')
    plt.ylabel('Intensity')
    plt.ylim([0, ylim2])

    plt.savefig(fname)
    plt.close()
