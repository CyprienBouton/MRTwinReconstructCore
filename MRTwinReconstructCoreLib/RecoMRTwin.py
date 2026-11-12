
import numpy as np
import torch
import twixtools
import matplotlib.pyplot as plt


class RecoMRTwin:

    def __init__(self, filename):

        self.filename = filename

        self.twixobj = twixtools.read_twix(filename)[-1]
        self.twixmap = twixtools.map_twix(self.twixobj)

        if "image" not in self.twixmap:
            raise RuntimeError("No 'image' dataset found in twixmap.")

        self.image_data = self.twixmap["image"]

        self.twix_center = (
            self.image_data.kspace_center_col,
            self.image_data.kspace_center_lin,
            self.image_data.kspace_center_par,
        )

        self.kspace_center = self.twix_center

    # ================================================================
    # CENTERING GEOMETRY
    # ================================================================

    @staticmethod
    def get_centering_geometry(kspace_shape, center):

        """
        Calculate zero-padding required to put the Twix center
        at the geometric center of each encoding dimension.

        Input:
            kspace_shape: [Col, Lin, Par, Coils]
            center: (Col, Lin, Par)

        Returns:
            padding
            target_center
        """

        padding = []
        target_center = []

        for axis, current in enumerate(center):

            if current is None:
                raise RuntimeError(
                    f"Twix center coordinate for axis {axis} is None."
                )

            current = int(current)
            n = int(kspace_shape[axis])

            if not 0 <= current < n:
                raise RuntimeError(
                    f"Twix center {current} is outside axis {axis} "
                    f"with size {n}."
                )

            radius = max(
                current,
                n - 1 - current
            )

            output_size = 2 * radius + 1
            target = radius

            before = target - current
            after = output_size - n - before

            padding.append((before, after))
            target_center.append(target)

        padding.append((0, 0))

        return padding, tuple(target_center)

    # ================================================================
    # GET KSPACE
    # ================================================================

    def get_kspace(self, apply_centering=True):

        data = self.image_data[:]

        data = np.squeeze(data)

        if data.ndim != 4:
            raise RuntimeError(
                f"Expected 4D Twix data after squeeze, got {data.shape}"
            )

        # Twix [Par, Lin, Cha, Col]
        # Internal [Col, Lin, Par, Cha]

        kspace = np.transpose(
            data,
            (3, 1, 0, 2)
        )

        kspace = np.ascontiguousarray(kspace)

        self.kspace_center = self.twix_center

        if apply_centering:

            kspace = self.center_kspace(
                kspace,
                self.twix_center
            )

            _, self.kspace_center = self.get_centering_geometry(
                np.transpose(data, (3, 1, 0, 2)).shape,
                self.twix_center
            )

        return torch.from_numpy(
            np.ascontiguousarray(kspace)
        )

    # ================================================================
    # CENTER KSPACE
    # ================================================================

    @staticmethod
    def center_kspace(kspace: np.ndarray, center):

        """
        Zero-pad k-space so that the Twix-reported center
        is at the geometric center of the output.

        Input:
            kspace: [Col, Lin, Par, Coils]
            center: (Col, Lin, Par)

        No circular shifting or interpolation.
        """

        padding, _ = RecoMRTwin.get_centering_geometry(
            kspace.shape,
            center
        )

        return np.pad(
            kspace,
            pad_width=padding,
            mode="constant",
            constant_values=0
        )

    # ================================================================
    # CENTERED FOURIER TRANSFORMS
    # ================================================================

    @staticmethod
    def centered_ifft(data, dims):

        dims = tuple(sorted(set(dims)))

        if not dims:
            return data

        data = torch.fft.ifftshift(
            data,
            dim=dims
        )

        data = torch.fft.ifftn(
            data,
            dim=dims
        )

        data = torch.fft.fftshift(
            data,
            dim=dims
        )

        return data

    @staticmethod
    def centered_fft(data, dims):

        dims = tuple(sorted(set(dims)))

        if not dims:
            return data

        data = torch.fft.ifftshift(
            data,
            dim=dims
        )

        data = torch.fft.fftn(
            data,
            dim=dims
        )

        data = torch.fft.fftshift(
            data,
            dim=dims
        )

        return data

    # ================================================================
    # POCS MASK
    # ================================================================

    @staticmethod
    def get_partial_fourier_mask(kspace, dim_pf):

        """
        Detect acquired positions along the partial-Fourier dimension.

        This assumes that unacquired positions are zero across
        all other dimensions and coils.
        """

        reduction_dims = tuple(
            axis
            for axis in range(kspace.ndim)
            if axis != dim_pf
        )

        return torch.any(
            torch.abs(kspace) > 0,
            dim=reduction_dims
        )

    # ================================================================
    # POCS PARTIAL FOURIER
    # ================================================================

    @staticmethod
    def pocs_partial_fourier(
        kspace,
        dim_enc=(0, 1, 2),
        dim_pf=0,
        number_of_iterations=10,
        gaussian_std=10.0,
        center_index=None,
        device="cpu"
    ):

        """
        Partial Fourier reconstruction using POCS.

        Input:
            kspace: [Col, Lin, Par, Coils]

        dim_pf:
            Partial Fourier dimension.

        dim_enc:
            Fourier encoding dimensions, excluding coils.

        center_index:
            Actual DC index in the CURRENT k-space array.

        Assumptions:
            - PF is along one dimension.
            - Missing PF samples occur at one edge.
            - There are no internal acceleration gaps.
            - A symmetric central region is acquired.

        Acquired samples are preserved.
        Estimated missing samples are retained.
        """

        if not isinstance(kspace, torch.Tensor):
            kspace = torch.as_tensor(kspace)

        if not torch.is_complex(kspace):
            raise TypeError("POCS requires complex-valued k-space.")

        if kspace.ndim != 4:
            raise ValueError(
                f"Expected 4D k-space, got {tuple(kspace.shape)}"
            )

        if number_of_iterations < 1:
            raise ValueError(
                "number_of_iterations must be >= 1."
            )

        if gaussian_std <= 0:
            raise ValueError("gaussian_std must be positive.")

        original_device = kspace.device
        original_dtype = kspace.dtype

        kspace = kspace.to(device)

        ndim = kspace.ndim
        dim_pf = dim_pf % ndim

        dim_enc = tuple(
            sorted(set(int(d) % ndim for d in dim_enc))
        )

        if dim_pf not in dim_enc:
            raise ValueError(
                "dim_pf must belong to dim_enc."
            )

        nonpf_enc_dims = tuple(
            d for d in dim_enc if d != dim_pf
        )

        if ndim - 1 in dim_enc:
            raise ValueError(
                "The coil dimension must not be Fourier transformed."
            )

        n_pf = kspace.shape[dim_pf]

        if center_index is None:
            center_index = n_pf // 2

        center_index = int(center_index)

        if not 0 <= center_index < n_pf:
            raise ValueError("Invalid PF center index.")

        # ------------------------------------------------------------
        # 1. Acquisition mask
        # ------------------------------------------------------------

        mask_original = RecoMRTwin.get_partial_fourier_mask(
            kspace,
            dim_pf
        )

        acquired_indices = torch.where(mask_original)[0]

        if acquired_indices.numel() == 0:
            raise RuntimeError(
                "No acquired PF positions were detected."
            )

        first_acquired = int(acquired_indices[0].item())
        last_acquired = int(acquired_indices[-1].item())

        acquired_count = int(acquired_indices.numel())

        acquired_fraction = acquired_count / n_pf

        if acquired_fraction >= 1.0:
            raise RuntimeError(
                "No missing PF positions detected."
            )

        if acquired_fraction <= 0.5:
            raise RuntimeError(
                "POCS requires more than 50% of PF positions."
            )

        if not mask_original[center_index]:
            raise RuntimeError(
                "PF center is not acquired; phase estimation is unsafe."
            )

        # ------------------------------------------------------------
        # 2. Determine contiguous acquisition support
        # ------------------------------------------------------------

        support_mask = torch.zeros(
            n_pf,
            dtype=torch.bool,
            device=kspace.device
        )

        support_mask[first_acquired:last_acquired + 1] = True

        internal_holes = torch.where(
            ~mask_original[first_acquired:last_acquired + 1]
        )[0]

        if internal_holes.numel() > 0:
            raise RuntimeError(
                "Internal missing PF positions were detected. "
                "This POCS implementation assumes contiguous "
                "one-dimensional partial Fourier acquisition."
            )

        missing_left = first_acquired > 0
        missing_right = last_acquired < n_pf - 1

        if missing_left and missing_right:
            raise RuntimeError(
                "Missing samples detected on both sides. "
                "Only one-sided partial Fourier is supported."
            )

        if not missing_left and not missing_right:
            raise RuntimeError(
                "No one-sided partial-Fourier region detected."
            )

        # ------------------------------------------------------------
        # 3. Symmetric central region
        # ------------------------------------------------------------

        indices = torch.arange(
            n_pf,
            device=kspace.device
        )

        reflected_indices = 2 * center_index - indices

        valid_reflections = (
            (reflected_indices >= 0)
            & (reflected_indices < n_pf)
        )

        symmetric_mask = torch.zeros(
            n_pf,
            dtype=torch.bool,
            device=kspace.device
        )

        valid_indices = indices[valid_reflections]
        valid_reflected = reflected_indices[valid_reflections]

        symmetric_mask[valid_indices] = (
            support_mask[valid_indices]
            & support_mask[valid_reflected]
        )

        if symmetric_mask.sum() < 3:
            raise RuntimeError(
                "Insufficient symmetric central region."
            )

        # ------------------------------------------------------------
        # 4. Gaussian weighting of symmetric k-space
        # ------------------------------------------------------------

        coordinates = torch.arange(
            n_pf,
            dtype=torch.float32,
            device=kspace.device
        )

        gaussian = torch.exp(
            -0.5 * (
                (coordinates - center_index) / gaussian_std
            ) ** 2
        )

        gaussian = gaussian * symmetric_mask.float()

        gaussian_shape = [1] * ndim
        gaussian_shape[dim_pf] = n_pf

        gaussian = gaussian.reshape(gaussian_shape)

        kspace_symmetric = kspace * gaussian

        # ------------------------------------------------------------
        # 5. Phase estimation in HYBRID SPACE
        # ------------------------------------------------------------

        # First transform the non-PF encoding dimensions.
        # The PF dimension remains in k-space.

        kspace_symmetric_hybrid = RecoMRTwin.centered_ifft(
            kspace_symmetric,
            dims=nonpf_enc_dims
        )

        # Then transform the PF dimension.
        # Result: image along PF, k-space along non-PF dimensions.
        # This is the same hybrid representation used in the POCS loop.

        phase_estimate = RecoMRTwin.centered_ifft(
            kspace_symmetric_hybrid,
            dims=(dim_pf,)
        )

        phase_magnitude = torch.abs(phase_estimate)

        maximum_phase_magnitude = phase_magnitude.max()

        if maximum_phase_magnitude <= 0:
            raise RuntimeError(
                "Phase estimation failed: symmetric k-space is zero."
            )

        phase_threshold = maximum_phase_magnitude * 1e-6

        valid_phase = phase_magnitude > phase_threshold

        phase_factor = torch.ones_like(phase_estimate)

        phase_factor[valid_phase] = (
            phase_estimate[valid_phase]
            / phase_magnitude[valid_phase]
        )

        # ------------------------------------------------------------
        # 6. Hybrid representation of original k-space
        # ------------------------------------------------------------

        kspace_hybrid = RecoMRTwin.centered_ifft(
            kspace,
            dims=nonpf_enc_dims
        )

        kspace_hybrid_original = kspace_hybrid.clone()

        mask_shape = [1] * ndim
        mask_shape[dim_pf] = n_pf

        mask_view = mask_original.reshape(mask_shape)

        # ------------------------------------------------------------
        # 7. POCS iterations
        # ------------------------------------------------------------

        for _ in range(number_of_iterations):

            # Hybrid k-space -> image space along PF dimension.

            image_hybrid = RecoMRTwin.centered_ifft(
                kspace_hybrid,
                dims=(dim_pf,)
            )

            # Impose estimated phase while preserving magnitude.

            image_hybrid = (
                torch.abs(image_hybrid)
                * phase_factor
            )

            # Image space -> hybrid k-space.

            kspace_hybrid = RecoMRTwin.centered_fft(
                image_hybrid,
                dims=(dim_pf,)
            )

            # Restore acquired samples exactly.

            kspace_hybrid = torch.where(
                mask_view,
                kspace_hybrid_original,
                kspace_hybrid
            )

        # ------------------------------------------------------------
        # 8. Return to fully encoded k-space
        # ------------------------------------------------------------

        kspace_reconstructed = RecoMRTwin.centered_fft(
            kspace_hybrid,
            dims=nonpf_enc_dims
        )

        # ------------------------------------------------------------
        # 9. Verify acquired-data consistency
        # ------------------------------------------------------------

        mask_full = mask_view.expand_as(kspace)

        consistency_error = torch.max(
            torch.abs(
                kspace_reconstructed[mask_full]
                - kspace[mask_full]
            )
        )

        tolerance = 1e-5 * max(
            float(torch.abs(kspace[mask_full]).max().item()),
            1.0
        )

        if float(consistency_error.item()) > tolerance:
            raise RuntimeError(
                "POCS data consistency check failed: "
                f"maximum difference = {consistency_error.item():.6e}"
            )

        # Missing samples are intentionally retained.
        # Do not apply the original acquisition mask again.

        return kspace_reconstructed.to(
            device=original_device,
            dtype=original_dtype
        )

    # ================================================================
    # PREPARE KSPACE
    # ================================================================

    def prepare_kspace(self, kspace):

        return kspace

    # ================================================================
    # FFT
    # ================================================================

    def fft_reconstruct(self, kspace):

        kspace = torch.fft.ifftshift(
            kspace,
            dim=(0, 1, 2)
        )

        image = torch.fft.ifftn(
            kspace,
            dim=(0, 1, 2)
        )

        image = torch.fft.fftshift(
            image,
            dim=(0, 1, 2)
        )

        return image

    # ================================================================
    # ACS
    # ================================================================

    def get_acs_from_kspace(self, kspace):

        mask = (
            kspace.abs().sum(dim=-1) > 0
        )

        full_x = mask.any(dim=(1, 2))
        full_y = mask.any(dim=(0, 2))
        full_z = mask.any(dim=(0, 1))

        def get_bounds(values):

            indices = torch.where(values)[0]

            if len(indices) == 0:
                return None, None

            return (
                indices[0].item(),
                indices[-1].item() + 1
            )

        x0, x1 = get_bounds(full_x)
        y0, y1 = get_bounds(full_y)
        z0, z1 = get_bounds(full_z)

        if None in (x0, x1, y0, y1, z0, z1):
            raise RuntimeError(
                "Could not determine ACS region."
            )

        return kspace[x0:x1, y0:y1, z0:z1, :]

    # ================================================================
    # SPACING
    # ================================================================

    def get_spacing(self):

        return (1.0, 1.0, 1.0)

    # ================================================================
    # GRAPPA
    # ================================================================

    def grappa_reconstruct(
        self,
        kspace,
        acs,
        kernel_size=(4, 1, 5),
        reg=1e-4
    ):

        raise NotImplementedError(
            "GRAPPA is not implemented."
        )
