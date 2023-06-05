import matplotlib.pyplot as plt
import numpy as np

def multi_slice_viewer(volume, index=0, figsize=(6, 6)):
    """
    volume : 3D numpy array
        3D image for viewing plane-wise

    index : int
        Image index to display, defaults to zero
    """
    volume = np.flip(volume)
    fig, ax = plt.subplots(figsize=figsize)
    plt.tick_params(axis='both', which='both', top=False,
                    bottom=False, left=False, right=False,
                    labelbottom=False, labelleft=False)
    ax.volume = volume
    ax.index = index  # volume.shape[0] // 2q
    ax.imshow(volume[ax.index], cmap = "gray", 
            #   vmin = vmin, 
            #   vmax = vmax
              )
    __remove_keymap_conflicts({'q','w'})
    fig.canvas.mpl_connect('key_press_event', __process_key)
    plt.show()

def __process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'q':
        __previous_slice(ax)
    elif event.key == 'w':
        __next_slice(ax)
    fig.canvas.draw()

def __previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 10) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def __next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 10) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])

def __remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)
