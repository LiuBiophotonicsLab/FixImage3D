# FixImage3D

### Contents

- [Overview](#overview)
- [System requirements](#system-requirements)
- [Installation](#installation)
- [Demo](#demo)
- [Results](#results)

## Overview

FixImage3D is an image processing tool designed to to normalize signal levels in a two-channel fluorescence 3D image volume, i.e. nuclear and cytoplasmic stain. This software corrects two types of variations found in Open-top light-sheet microscopy (OTLS) images:

1. Lateral variations: Vertical stripes that caused by photobleaching of overlapping regions between adjacent tiles.

2. Depth variations: Fluorescence signal intensity decreases with increasing imaging depth due to light scattering or absorption effects.

## System requirements

This package is written in Python. Users should install Anaconda (tested on Conda 4.12.0, Python 3.8.8, Window 10)

In addition, the following packages are required, many of which come with Python/Anaconda. This code has been tested with the version number indicated, though other versions may also work.

- Numpy 1.20.0
- h5py 2.10.0
- Scikit-image 0.21.0
- Matplotlib 3.3.4

No specific computing hardware is required.

## Installation

After installing the required packages above, run the following command in Anaconda to clone this repository:

```bash
git clone https://github.com/sarahrahsl/FixImage3D.git
```

Typical installation time (after setting up Anaconda environment): less than 5 minutes.

## Demo

For a quick start, run the following command:

```bash
python Fix_script.py --h5path 
```

For a step-by-step guide or to only run stripe-fix, refer to the Jupyter notebook.

## Results

The output for quick start command includes: 

- An HDF5 file saved with the name:  *sample_corrected.h5*
- 4 TIFF files for cyto channel and nuc channel, before and after correction, with the name: *sample_corrected_nuc.tiff* , *sample_corrected_cyto.tiff* , *sample_nuc.tiff* , *sample_cyto.tiff*
- 4 PNG files plotting the mean for each channel before and after correction.
