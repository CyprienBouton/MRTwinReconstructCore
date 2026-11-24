"""Low-level reconstruction of Siemens twix raw data.

:class:`RecoMRTwin` reads a Siemens ``.dat`` (twix) file with ``twixtools``,
exposes its k-space as ``[Col, Lin, Par, Cha]`` (= ``[nx, ny, nz, ncoils]``)
and provides the reconstruction back-ends used by
:class:`MRTwinReconstructCoreLib.MRTwinReconstructLogic`:

* centred inverse FFT,
* POCS partial-Fourier reconstruction,
* GRAPPA (vendored ``ggrappa``).

K-space arrays follow the conventions documented in :mod:`.reco_utils`.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from . import reco_utils

#: Twix dimension order after ``np.squeeze``: [Par, Lin, Cha, Col] ->
#: internal [Col, Lin, Par, Cha].
_TWIX_TO_INTERNAL = (3, 1, 0, 2)


class RecoMRTwin:
    """Reconstruction of one Siemens twix measurement.

    Args:
        filename: path to the ``.dat`` file. The last measurement of a
            multi-RAID file is used (the first ones are usually noise
            adjustment scans).
    """

    def __init__(self, filename):
        import twixtools  # provided by the gt-twixtools distribution

        self.filename = filename
        self.twixobj = twixtools.read_twix(filename)[-1]
        self.twixmap = twixtools.map_twix(self.twixobj)
        if "image" not in self.twixmap:
            raise RuntimeError("No 'image' dataset found in the twix file.")
        self.image_data = self.twixmap["image"]

        #: k = 0 indices (Col, Lin, Par) reported by the scanner.
        self.twix_center = (
            self.image_data.kspace_center_col,
            self.image_data.kspace_center_lin,
            self.image_data.kspace_center_par,
        )
        #: k = 0 indices of the array currently being reconstructed.
        self.kspace_center = self.twix_center

    # ================================================================
    # K-SPACE
    # ================================================================

    @staticmethod
    def _to_internal(data: np.ndarray) -> torch.Tensor:
        """Squeezed twix array [Par, Lin, Cha, Col] -> tensor [Col, Lin, Par, Cha]."""
        data = np.squeeze(data)
        if data.ndim != 4:
            raise RuntimeError(
                f"Expected 4D twix data [Par, Lin, Cha, Col] after squeeze, got {data.shape}. "
                "Multi-slice, multi-echo, segmented or averaged data are not supported."
            )
        return torch.from_numpy(np.ascontiguousarray(np.transpose(data, _TWIX_TO_INTERNAL)))

    def get_kspace(self, apply_centering: bool = False) -> torch.Tensor:
        """Image k-space ``[Col, Lin, Par, Cha]`` (unacquired samples are zero).

        Args:
            apply_centering: zero-pad so that the twix k-space centre lands at
                index ``n // 2`` (see :meth:`center_kspace`). Updates
                :attr:`kspace_center` accordingly.
        """
        kspace = self._to_internal(self.image_data[:])
        self.kspace_center = self.twix_center
        if apply_centering:
            kspace, self.kspace_center = self.center_kspace(kspace)
        return kspace

    def get_refscan_kspace(self, image_shape: Sequence[int]) -> Optional[torch.Tensor]:
        """Separate GRAPPA reference scan ``[Col, Lin, Par, Cha]``, or None.

        Returns None when the file has no ``refscan`` dataset or when its
        shape differs from ``image_shape`` (the raw image k-space shape), in
        which case the indexing cannot be assumed to match the image data.
        """
        if "refscan" not in self.twixmap:
            return None
        try:
            refscan = self._to_internal(self.twixmap["refscan"][:])
        except RuntimeError:
            return None
        return refscan if tuple(refscan.shape) == tuple(image_shape) else None

    def center_kspace(self, kspace: torch.Tensor):
        """Zero-pad so that the twix centre is at index ``n // 2`` (no wrapping).

        Returns ``(centered_kspace, new_center)``; see
        :func:`.reco_utils.center_kspace`.
        """
        return reco_utils.center_kspace(kspace, self.twix_center)

    # ================================================================
    # RECONSTRUCTION BACK-ENDS
    # ================================================================

    def fft_reconstruct(self, kspace: torch.Tensor) -> torch.Tensor:
        """Unitary centred inverse FFT over Col, Lin and Par.

        Returns complex coil images ``[nx, ny, nz, ncoils]``.
        """
        return reco_utils.centered_ifft(kspace, dims=(0, 1, 2))

    @staticmethod
    def pocs_partial_fourier(kspace, dim_enc=(0, 1, 2), dim_pf=0, number_of_iterations=10,
                             gaussian_std=10.0, center_index=None, device="cpu"):
        """POCS partial-Fourier k-space completion; see
        :func:`.reco_utils.pocs_partial_fourier`."""
        return reco_utils.pocs_partial_fourier(
            kspace, dim_enc=dim_enc, dim_pf=dim_pf,
            number_of_iterations=number_of_iterations, gaussian_std=gaussian_std,
            center_index=center_index, device=device,
        )

    def get_acs_from_kspace(self, kspace: torch.Tensor,
                            center: Optional[Sequence[int]] = None) -> torch.Tensor:
        """Largest fully sampled block around ``center`` (default ``n // 2``)."""
        return reco_utils.get_acs_from_kspace(kspace, center)

    def grappa_reconstruct(self, kspace: torch.Tensor, acs: torch.Tensor,
                           af: Optional[Sequence[int]] = None,
                           kernel_size: Sequence[int] = (4, 4, 5),
                           reg: float = 1e-4, batch_size: int = 1,
                           cuda: Optional[bool] = None, quiet: bool = True) -> torch.Tensor:
        """GRAPPA k-space filling; returns k-space ``[nx, ny, nz, ncoils]``.

        See :func:`.reco_utils.grappa_reconstruct`. ``af=None`` estimates the
        acceleration from the sampling pattern; ``cuda=None`` uses the GPU
        when available.
        """
        return reco_utils.grappa_reconstruct(kspace, acs, af=af, kernel_size=kernel_size,
                                             reg=reg, batch_size=batch_size, cuda=cuda,
                                             quiet=quiet)

    # ================================================================
    # GEOMETRY
    # ================================================================

    def get_spacing(self, shape: Optional[Sequence[int]] = None) -> Tuple[float, float, float]:
        """Voxel size [mm]. Currently always ``(1, 1, 1)``.

        The physical resolution is not read from the twix header yet
        (``shape`` is accepted for API symmetry with ``RecoMRzero``).
        """
        return (1.0, 1.0, 1.0)
