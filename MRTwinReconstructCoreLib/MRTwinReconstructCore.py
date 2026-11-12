
import numpy as np
import torch
import SimpleITK as sitk

from .RecoMRTwin import RecoMRTwin


class MRTwinReconstructLogic:

    # ================================================================
    # KSPACE -> SLICER
    # ================================================================

    @staticmethod
    def kspace_to_sitk(kspace):

        magnitude = torch.sqrt(
            torch.sum(
                torch.abs(kspace) ** 2,
                dim=-1
            )
        )

        display = torch.log1p(magnitude)

        display = display - display.min()

        maximum = display.max()

        if maximum > 0:
            display = display / maximum

        volume = display.permute(2, 1, 0).float()

        return MRTwinReconstructLogic.make_sitk_image(volume)

    # ================================================================
    # IMAGE -> SLICER
    # ================================================================

    @staticmethod
    def make_sitk_image(volume, spacing=(1.0, 1.0, 1.0)):

        data = (
            volume
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        image = sitk.GetImageFromArray(data)

        image.SetSpacing(spacing)

        image.SetDirection(
            (
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
            )
        )

        return image

    # ================================================================
    # CENTERING
    # ================================================================

    @staticmethod
    def center_reconstruction_kspace(kspace, center):

        """
        Zero-pad a separate reconstruction copy of k-space.

        Input:
            kspace: torch.Tensor [Col, Lin, Par, Coils]
            center: Twix center (Col, Lin, Par)

        Returns:
            centered_kspace
            updated_center

        The input tensor is never modified.
        """

        padding, target_center = RecoMRTwin.get_centering_geometry(
            kspace.shape,
            center
        )

        kspace_np = (
            kspace
            .detach()
            .cpu()
            .numpy()
        )

        centered_np = np.pad(
            kspace_np,
            pad_width=padding,
            mode="constant",
            constant_values=0
        )

        centered_kspace = torch.from_numpy(
            np.ascontiguousarray(centered_np)
        )

        return centered_kspace, target_center

    # ================================================================
    # MAIN RECONSTRUCTION
    # ================================================================

    def runReconstruction(
        self,
        twixPath,
        method="FFT",
        apply_centering=True,
        pocs_iterations=30,
        pocs_device="cpu"
    ):

        method = method.upper()

        reco = RecoMRTwin(twixPath)

        # ------------------------------------------------------------
        # Load original k-space
        # ------------------------------------------------------------

        raw_kspace = reco.get_kspace(
            apply_centering=False
        )

        # Keep raw k-space unchanged.
        raw_kspace_sitk = self.kspace_to_sitk(
            raw_kspace
        )

        # Work on a separate reconstruction copy.
        reconstruction_kspace = raw_kspace.clone()

        # ------------------------------------------------------------
        # Optional zero-padding centering
        # ------------------------------------------------------------

        if apply_centering:

            reconstruction_kspace, target_center = (
                self.center_reconstruction_kspace(
                    reconstruction_kspace,
                    reco.twix_center
                )
            )

            reco.kspace_center = target_center

        else:

            reco.kspace_center = reco.twix_center

        # ------------------------------------------------------------
        # Reconstruction
        # ------------------------------------------------------------

        if method == "FFT":

            reconstruction_kspace = reco.prepare_kspace(
                reconstruction_kspace
            )

            output = reco.fft_reconstruct(
                reconstruction_kspace
            )

        elif method == "POCS":

            reconstruction_kspace = reco.pocs_partial_fourier(
                reconstruction_kspace,
                dim_enc=(0, 1, 2),
                dim_pf=0,
                number_of_iterations=pocs_iterations,
                center_index=int(reco.kspace_center[0]),
                device=pocs_device
            )

            output = reco.fft_reconstruct(
                reconstruction_kspace
            )

        elif method == "GRAPPA":

            reconstruction_kspace = reco.prepare_kspace(
                reconstruction_kspace
            )

            acs = reco.get_acs_from_kspace(
                reconstruction_kspace
            )

            output = reco.grappa_reconstruct(
                reconstruction_kspace,
                acs
            )

        else:

            raise ValueError(
                f"Unknown reconstruction method: {method}. "
                "Available methods: FFT, POCS, GRAPPA."
            )

        # ------------------------------------------------------------
        # Coil combination
        # ------------------------------------------------------------

        if output.dim() == 4:

            magnitude = output.abs().mean(dim=-1)

        elif output.dim() == 3:

            magnitude = output.abs()

        else:

            raise ValueError(
                f"Unexpected reconstruction shape: {output.shape}"
            )

        # ------------------------------------------------------------
        # Convert reconstructed image to SimpleITK
        # ------------------------------------------------------------

        volume = magnitude.permute(2, 1, 0)

        volume = torch.flip(
            volume,
            dims=(1,)
        )

        image_sitk = self.make_sitk_image(
            volume,
            spacing=reco.get_spacing()
        )

        # ------------------------------------------------------------
        # Convert reconstructed k-space to SimpleITK
        # ------------------------------------------------------------

        reconstruction_kspace_sitk = self.kspace_to_sitk(
            reconstruction_kspace
        )

        return (
            image_sitk,
            reconstruction_kspace_sitk,
            raw_kspace_sitk
        )

    # ================================================================
    # MULTIPLE-FILE RECONSTRUCTION
    # ================================================================

    def runMultipleReconstructions(
        self,
        twixPaths,
        method="FFT",
        apply_centering=True,
        pocs_iterations=10,
        pocs_device="cpu"
    ):

        results = []

        for twixPath in twixPaths:

            try:

                result = self.runReconstruction(
                    twixPath,
                    method=method,
                    apply_centering=apply_centering,
                    pocs_iterations=pocs_iterations,
                    pocs_device=pocs_device
                )

                results.append({
                    "filename": twixPath,
                    "result": result,
                    "error": None
                })

            except Exception as exc:

                results.append({
                    "filename": twixPath,
                    "result": None,
                    "error": str(exc)
                })

        return results
