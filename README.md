# MRTwin Reconstruct Core

Python reconstruction core for **Siemens twix raw MRI data** (`.dat`). It reads
the raw k-space with [twixtools](https://github.com/pehses/twixtools) and
reconstructs it with a centred FFT, POCS partial Fourier or GRAPPA. Results are
returned as SimpleITK images.

The package contains no 3D Slicer code, so it can be used and tested on its own.

| | |
|---|---|
| Distribution name | `mrtwin-reconstruct-core` |
| Import package | `MRTwinReconstructCoreLib` |
| Sister package | `mrzero-reconstruct-core` (MRzero simulated signals, same layout and API) |

> **Naming.** Despite the "MRTwin" name, this package reads Siemens **twix**
> files. It is unrelated to the PyPI package `mrtwin`. The import name is kept
> for compatibility with the Slicer extension `MRTwixReconstruct`.

## Used by

The 3D Slicer extension **MRTwixReconstruct** (`SlicerExtensions/MRTwixReconstruct`,
module `MRTwinReconstruct`). It imports
`MRTwinReconstructCoreLib.MRTwinReconstructLogic` and calls
`runReconstruction(path, method)`. If the package is missing, the extension
installs it from this GitHub repository with `slicer.util.pip_install`.

## Reconstruction methods

| `method` | What it does | Needs |
|---|---|---|
| `"FFT"` | Zero-pads k-space so that the scanner-reported centre is at `n // 2`, then applies a unitary centred inverse FFT. | — |
| `"POCS"` | Completes one-sided partial-Fourier k-space by POCS (smooth-phase constraint) along `pocs_dim`, then FFT. | — |
| `"GRAPPA"` | Calibrates a GRAPPA kernel on the ACS lines (separate `refscan` if present and compatible, otherwise the fully sampled centre of the image data) and fills the missing lines (vendored `ggrappa`), then FFT. | — (GPU used when available) |

Method names are case-insensitive. The image is the root-sum-of-squares (RSS)
of the coil images.

## Installation

```bash
# from GitHub
python -m pip install "git+https://github.com/CyprienBouton/MRTwinReconstructCore.git"

# from a clone (editable, with test dependencies)
python -m pip install -e ".[test]"
```

Requirements: Python ≥ 3.9, `numpy`, `torch`, `scipy`, `tqdm`, `SimpleITK`,
`gt-twixtools` (installed automatically; this is the PyPI distribution of
`pehses/twixtools`, import name `twixtools`). Do not install the older
`twixtools` distribution alongside it: both provide the same `twixtools`
module.

**In 3D Slicer**, install from the Slicer Python console
(`slicer.util.pip_install("git+https://github.com/CyprienBouton/MRTwinReconstructCore.git")`)
or let the extension do it. Installing into your system Python does not make
the package available inside Slicer.

### BART

Not needed by any method of this package. The vendored BART bindings
(`MRTwinReconstructCoreLib.bart_python`) are kept so that both cores share the
same `reco_utils`; to use them (`reco_utils.run_bart`), install the
[BART](https://mrirecon.github.io/bart/) binary and set `BART_TOOLBOX_PATH`
(or install it in `/usr/local/bin`, `/usr/bin`, or inside WSL on Windows).

## Quick start

```python
import SimpleITK as sitk
from MRTwinReconstructCoreLib import MRTwinReconstructLogic

logic = MRTwinReconstructLogic()
image, kspace, raw_kspace = logic.runReconstruction(
    "meas_MID00042.dat",
    "POCS",
    pocs_dim=1,            # partial Fourier along Lin
)
sitk.WriteImage(image, "image.nii.gz")
```

Command line: `python examples/reconstruct_twix.py meas_*.dat --method FFT -o out/ [--plot]`.

## API

### `MRTwinReconstructLogic` (`MRTwinReconstructCoreLib.MRTwinReconstructCore`)

`runReconstruction(twixPath, method="FFT", apply_centering=True, pocs_iterations=30, pocs_device="cpu", pocs_dim=0, kernelSize=(4, 1, 5), reg=1e-4, af=None)`

| Argument | Meaning |
|---|---|
| `twixPath` | Siemens `.dat` file (the last measurement of a multi-RAID file is used) |
| `method` | `"FFT"`, `"POCS"` or `"GRAPPA"` |
| `apply_centering` | Zero-pad so that the scanner-reported k-space centre is at `n // 2` |
| `pocs_iterations`, `pocs_device`, `pocs_dim` | POCS iterations, torch device, partial-Fourier axis (0 = Col, 1 = Lin, 2 = Par) |
| `kernelSize`, `reg`, `af` | GRAPPA kernel `(ky, kz, kx)`, Tikhonov regularisation, acceleration `(af_y, af_z)` (`None` = detected) |

Returns `(image_sitk, reconstruction_kspace_sitk, raw_kspace_sitk)`: the RSS
magnitude image (unit spacing, Lin axis flipped for display), the k-space that
produced it and the raw k-space. The two k-space volumes are log-compressed RSS
magnitudes normalised to [0, 1], for display.

`runMultipleReconstructions(twixPaths, method="FFT", **kwargs)` runs several
files and returns a list of `{"filename", "result", "error"}`.

Static helpers: `kspace_to_sitk`, `make_sitk_image`, `center_origin`.

### `RecoMRTwin` (lower level)

| Method | Purpose |
|---|---|
| `get_kspace(apply_centering=False)` | Image k-space `[Col, Lin, Par, Cha]` |
| `get_refscan_kspace(image_shape)` | Separate GRAPPA reference scan, or `None` |
| `center_kspace(kspace)` | Zero-pad so that the twix centre is at `n // 2` |
| `fft_reconstruct(kspace)` | Unitary centred inverse FFT → coil images |
| `pocs_partial_fourier(kspace, dim_pf, number_of_iterations, center_index)` | POCS → completed k-space |
| `get_acs_from_kspace(kspace, center)` | Largest fully sampled block around the centre |
| `grappa_reconstruct(kspace, acs, af, kernel_size, reg)` | GRAPPA → filled k-space |
| `get_spacing(shape)` | Voxel size in mm (currently always 1 mm) |

### `reco_utils` (shared with the MRzero package)

Centred FFTs, centring by zero-padding, ACS / acceleration detection, GRAPPA,
POCS partial Fourier, BART calls and SimpleITK conversion. This file is kept
identical in both reconstruction cores.

## Conventions and methods

* **Arrays.** twixtools returns `[Par, Lin, Cha, Col]` after squeezing; the
  package works on `[Col, Lin, Par, Cha]` = `[nx, ny, nz, ncoils]`.
  Unacquired samples are exactly zero.
* **Centring.** The scanner-reported centre (`kspace_center_col/lin/par`) is
  moved to index `n // 2` by zero-padding (the `fftshift` convention).
  Already-centred axes are unchanged; asymmetric axes are padded to an even
  size; nothing is cropped.
* **POCS.** A real object has a Hermitian-symmetric k-space; MR images have a
  smooth phase instead. The phase `φ` is estimated from the symmetric,
  Gaussian-weighted centre of k-space, then the algorithm alternates
  `x ← |x|·e^{iφ}` in image space and "put the acquired samples back" in
  k-space. Only the partial-Fourier axis is iterated (the other axes stay in
  image space). Acquired samples are returned unchanged.
* **GRAPPA.** Each missing sample is a linear combination of the acquired
  neighbours of all coils, `s_j(k) = Σ_c Σ_d w_{j,c,d} s_c(k + d)`. The
  weights are fitted on the ACS by Tikhonov-regularised least squares,
  `W = (SᴴS + λ²σ_max I)⁻¹ SᴴT` with `λ = reg`.

## Repository structure

```text
MRTwixReconstructCore/
├── README.md
├── pyproject.toml
├── .gitignore
├── MRTwinReconstructCoreLib/
│   ├── __init__.py               # public API, __version__
│   ├── MRTwinReconstructCore.py  # MRTwinReconstructLogic (used by Slicer)
│   ├── RecoMRTwin.py             # twix reading + reconstruction back-ends
│   ├── reco_utils.py             # shared with the MRzero package (keep identical)
│   ├── ggrappa/                  # vendored GRAPPA (BSD-3-Clause, see LICENSE)
│   └── bart_python/              # vendored BART bindings (BSD-3-Clause, see LICENSE)
├── examples/
│   └── reconstruct_twix.py
└── tests/                        # synthetic-data tests, no .dat file needed
```

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The tests replace the twix reader with a synthetic multi-coil phantom, so no
raw data are needed. Never commit `.dat` files: they may contain patient data
(`.gitignore` excludes them).

## Licences

* **This package:** no licence has been chosen yet. Add a `LICENSE` file
  before publishing.
* **`ggrappa/`** — vendored from [gGRAPPA](https://github.com/mbertrait/ggrappa)
  (Maxime Bertrait and gGRAPPA collaborators) through the fork
  [CyprienBouton/ggrappa](https://github.com/CyprienBouton/ggrappa), commit
  `39cfab0`. BSD 3-Clause, `ggrappa/LICENSE` (upstream's `pyproject.toml`
  says MIT but its `LICENSE` file is BSD-3-Clause). Local change: the kernel
  sliding range in `application.py` (`- sbl + 1` instead of `- sbl - 1`), which
  left the last k-space lines/partitions unfilled.
* **`bart_python/`** — vendored from the Python bindings of
  [BART](https://github.com/mrirecon/bart) (`python/bart.py`, `cfl.py`,
  `wslsupport.py`). BSD 3-Clause, `bart_python/LICENSE`. Local changes: module
  renamed to `bart_python.py`, package-relative imports, WSL path helpers.
* Runtime dependencies: twixtools / `gt-twixtools` (MIT), PyTorch
  (BSD-style), NumPy/SciPy (BSD-3-Clause), SimpleITK (Apache-2.0), tqdm
  (MIT/MPL-2.0).

## Known limitations

* Only single-slice-group 3D data that squeeze to `[Par, Lin, Cha, Col]` are
  supported (no multi-slice 2D, echoes, repetitions, segments or averages).
* The voxel size is not read from the twix header (always 1 mm) and readout
  oversampling is not removed.
* GRAPPA has only been tested on synthetic data; acceleration detection
  assumes a regular Cartesian pattern (no CAIPIRINHA shift).
