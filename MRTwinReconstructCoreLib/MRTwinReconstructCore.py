"""High-level reconstruction entry point for Siemens twix raw data.

:class:`MRTwinReconstructLogic` is the API used by the 3D Slicer extension
``MRTwixReconstruct`` (module ``MRTwinReconstruct``). It reads a Siemens
``.dat`` file, reconstructs it and returns SimpleITK images ready to be
loaded in Slicer.
"""

from typing import Optional, Sequence

import SimpleITK as sitk
import torch

from . import reco_utils
from .RecoMRTwin import RecoMRTwin

#: Reconstruction methods accepted by ``runReconstruction`` (case-insensitive).
METHODS = ("FFT", "POCS", "GRAPPA")


class MRTwinReconstructLogic:
    """Reconstruct Siemens twix raw data into SimpleITK volumes."""

    # ================================================================
    # HELPERS (kept as static methods for backward compatibility)
    # ================================================================

    @staticmethod
    def kspace_to_sitk(kspace: torch.Tensor) -> sitk.Image:
        """Multi-coil k-space -> log-compressed RSS display volume (unit spacing)."""
        return reco_utils.kspace_to_sitk(kspace, center_volume=False)

    @staticmethod
    def center_origin(sitk_img: sitk.Image) -> sitk.Image:
        """Put physical (0, 0, 0) at the centre of the volume."""
        return reco_utils.center_origin(sitk_img)

    @staticmethod
    def make_sitk_image(volume: torch.Tensor,
                        spacing: Sequence[float] = (1.0, 1.0, 1.0),
                        origin: Sequence[float] = (0.0, 0.0, 0.0),
                        direction: Sequence[float] = reco_utils.IDENTITY_DIRECTION,
                        center_volume: bool = False) -> sitk.Image:
        """Real ``[z, y, x]`` tensor -> float32 SimpleITK image."""
        return reco_utils.make_sitk_image(volume, spacing=spacing, origin=origin,
                                          direction=direction, center_volume=center_volume)

    # ================================================================
    # MAIN RECONSTRUCTION
    # ================================================================

    def runReconstruction(
        self,
        twixPath: str,
        method: str = "FFT",
        apply_centering: bool = True,
        pocs_iterations: int = 30,
        pocs_device: str = "cpu",
        pocs_dim: int = 0,
        kernelSize: Sequence[int] = (4, 1, 5),
        reg: float = 1e-4,
        af: Optional[Sequence[int]] = None,
    ):
        """Reconstruct one twix file.

        Args:
            twixPath: Siemens ``.dat`` file.
            method: ``"FFT"``, ``"POCS"`` or ``"GRAPPA"`` (case-insensitive).
            apply_centering: zero-pad k-space so that the scanner-reported
                centre is at index ``n // 2`` before the FFT (recommended for
                partial Fourier / asymmetric echo).
            pocs_iterations: POCS: number of iterations.
            pocs_device: POCS: torch device (``"cpu"`` or ``"cuda"``).
            pocs_dim: POCS: partial-Fourier axis (0 = Col / asymmetric echo,
                1 = Lin, 2 = Par).
            kernelSize: GRAPPA kernel ``(ky, kz, kx)``.
            reg: GRAPPA Tikhonov regularisation.
            af: GRAPPA acceleration ``(af_y, af_z)``; ``None`` = estimated
                from the sampling pattern.

        Returns:
            ``(image_sitk, reconstruction_kspace_sitk, raw_kspace_sitk)``:
            the magnitude image (RSS over coils, unit spacing, Lin axis
            flipped), the k-space used for that image and the raw k-space,
            both as log-magnitude display volumes.
        """
        method_key = method.strip().upper()
        if method_key not in METHODS:
            raise ValueError(
                f"Unknown reconstruction method: {method}. "
                f"Available methods: {', '.join(METHODS)}."
            )

        reco = RecoMRTwin(twixPath)

        raw_kspace = reco.get_kspace(apply_centering=False)
        raw_kspace_sitk = self.kspace_to_sitk(raw_kspace)

        if apply_centering:
            reconstruction_kspace, reco.kspace_center = reco.center_kspace(raw_kspace)
        else:
            reconstruction_kspace = raw_kspace.clone()
            reco.kspace_center = reco.twix_center

        # ------------------------------------------------------------
        # Reconstruction
        # ------------------------------------------------------------
        if method_key == "POCS":
            reconstruction_kspace = reco.pocs_partial_fourier(
                reconstruction_kspace,
                dim_enc=(0, 1, 2),
                dim_pf=pocs_dim,
                number_of_iterations=pocs_iterations,
                center_index=int(reco.kspace_center[pocs_dim]),
                device=pocs_device,
            )

        elif method_key == "GRAPPA":
            refscan = reco.get_refscan_kspace(raw_kspace.shape)
            if refscan is None:
                acs_source = reconstruction_kspace          # integrated ACS lines
            elif apply_centering:
                acs_source, _ = reco.center_kspace(refscan)  # same padding as the image
            else:
                acs_source = refscan
            acs = reco.get_acs_from_kspace(acs_source, reco.kspace_center)
            reconstruction_kspace = reco.grappa_reconstruct(
                reconstruction_kspace, acs, af=af, kernel_size=kernelSize, reg=reg
            )

        output = reco.fft_reconstruct(reconstruction_kspace)

        # ------------------------------------------------------------
        # Coil combination and conversion
        # ------------------------------------------------------------
        magnitude = reco_utils.rss(output, dim=-1)
        volume = torch.flip(magnitude.permute(2, 1, 0), dims=(1,))
        image_sitk = self.make_sitk_image(volume, spacing=reco.get_spacing(magnitude.shape))
        reconstruction_kspace_sitk = self.kspace_to_sitk(reconstruction_kspace)

        return image_sitk, reconstruction_kspace_sitk, raw_kspace_sitk

    # ================================================================
    # MULTIPLE-FILE RECONSTRUCTION
    # ================================================================

    def runMultipleReconstructions(self, twixPaths, method="FFT", **kwargs):
        """Run :meth:`runReconstruction` on several files.

        Extra keyword arguments are forwarded to :meth:`runReconstruction`.
        Errors are caught per file.

        Returns:
            list of ``{"filename", "result", "error"}`` dicts, where ``result``
            is the tuple returned by :meth:`runReconstruction` (or None) and
            ``error`` the error message (or None).
        """
        results = []
        for twixPath in twixPaths:
            try:
                result = self.runReconstruction(twixPath, method, **kwargs)
                results.append({"filename": twixPath, "result": result, "error": None})
            except Exception as exc:  # keep going with the other files
                results.append({"filename": twixPath, "result": None, "error": str(exc)})
        return results
