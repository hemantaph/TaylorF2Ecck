# TaylorF2Ecck

This repository contains stored Bilby result files and plotting notebooks
for the TaylorF2Ecck eccentric-waveform paper.

Paper: https://arxiv.org/abs/2508.12697

It is mainly for inspecting the stored parameter-estimation results and
regenerating plots from those results.

## Data

All `.hdf` and `.hdf5` files are tracked with Git LFS.

After cloning, run:

```bash
git lfs install
git lfs pull
```

The HDF5 result files contain the posterior samples, priors, sampler
information, labels, and available metadata needed by the notebooks.

## Requirements

Use Python 3.10 with the usual scientific Python and gravitational-wave
analysis packages used by the notebooks, including:

```text
numpy scipy pandas matplotlib seaborn corner h5py tqdm jupyter
bilby bilby_pipe dynesty gwpy pycbc gwsnr
```

Waveform generation and parameter-estimation reruns using
`TaylorF2Ecck` or `TaylorF2Ecch` require the custom LALSuite fork used
in the paper:

```text
https://git.ligo.org/hemantakumar.phurailatpam/lalsuite
```

The pyEFPE comparison also requires:

```bash
python -m pip install git+https://github.com/gmorras/pyEFPE.git
```

## Parameter Estimation

This repository does not include every input/configuration file needed
to rerun all parameter-estimation jobs.

The included `.ini` files should be treated as templates. To rerun PE,
users should update the input and output paths and provide their own
required strain/frame data, PSDs, and run-specific settings.
