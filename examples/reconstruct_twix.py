"""Reconstruct Siemens twix files and save the results as NIfTI.

Example::

    python examples/reconstruct_twix.py meas_MID00042.dat --method POCS --pocs-dim 1 -o out/

Writes ``<name>_<method>_image.nii.gz``, ``<name>_<method>_kspace.nii.gz``
and ``<name>_raw_kspace.nii.gz`` for every input file, and an optional PNG of
the raw sampling pattern (``--plot``, requires matplotlib).
"""

import argparse
import os

import SimpleITK as sitk

from MRTwinReconstructCoreLib import METHODS, MRTwinReconstructLogic


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("twix", nargs="+", help="Siemens .dat file(s)")
    parser.add_argument("--method", default="FFT", choices=METHODS, type=str.upper)
    parser.add_argument("--pocs-dim", type=int, default=0,
                        help="partial-Fourier axis for POCS (0=Col, 1=Lin, 2=Par)")
    parser.add_argument("-o", "--output", default=".", help="output directory")
    parser.add_argument("--plot", action="store_true",
                        help="save a PNG of the raw k-space sampling pattern")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results = MRTwinReconstructLogic().runMultipleReconstructions(
        args.twix, method=args.method, pocs_dim=args.pocs_dim)

    for entry in results:
        name = os.path.splitext(os.path.basename(entry["filename"]))[0]
        if entry["error"] is not None:
            print(f"[ERROR] {name}: {entry['error']}")
            continue
        image, kspace, raw_kspace = entry["result"]
        prefix = os.path.join(args.output, name)
        sitk.WriteImage(image, f"{prefix}_{args.method}_image.nii.gz")
        sitk.WriteImage(kspace, f"{prefix}_{args.method}_kspace.nii.gz")
        sitk.WriteImage(raw_kspace, f"{prefix}_raw_kspace.nii.gz")
        print(f"[OK] {name} -> {prefix}_*.nii.gz")

        if args.plot:
            import matplotlib.pyplot as plt

            raw = sitk.GetArrayFromImage(raw_kspace)       # [Par, Lin, Col]
            sampled = raw.sum(axis=0) > 0                  # [Lin, Col]
            plt.imshow(sampled, cmap="gray", aspect="auto")
            plt.xlabel("Col")
            plt.ylabel("Lin")
            plt.title(f"{name}: acquired k-space positions")
            plt.savefig(f"{prefix}_sampling.png", dpi=150)
            plt.close()


if __name__ == "__main__":
    main()
