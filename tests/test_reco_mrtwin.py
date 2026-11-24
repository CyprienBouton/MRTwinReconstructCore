"""End-to-end tests of MRTwinReconstructLogic with a fake twix reader.

No ``.dat`` file is needed: :class:`FakeRecoMRTwin` serves a synthetic
phantom with the same array layout as twixtools ([Par, Lin, Cha, Col]).
"""

import SimpleITK as sitk
import torch

from MRTwinReconstructCoreLib import MRTwinReconstructCore as core
from MRTwinReconstructCoreLib import MRTwinReconstructLogic, RecoMRTwin, reco_utils
from phantom import make_phantom, relative_error

FAKE_FILES = {}


class FakeRecoMRTwin(RecoMRTwin):
    def __init__(self, filename):
        kspace = FAKE_FILES[filename]                       # [Col, Lin, Par, Cha]
        twix = kspace.numpy().transpose(2, 1, 3, 0)         # [Par, Lin, Cha, Col]
        self.filename = filename
        self.twixmap = {"image": twix}
        self.image_data = twix
        self.twix_center = tuple(n // 2 for n in kspace.shape[:3])
        self.kspace_center = self.twix_center


def _run(kspace, method, **kwargs):
    FAKE_FILES["fake.dat"] = kspace
    original = core.RecoMRTwin
    core.RecoMRTwin = FakeRecoMRTwin
    try:
        return MRTwinReconstructLogic().runReconstruction("fake.dat", method, **kwargs)
    finally:
        core.RecoMRTwin = original


def _expected_image(images):
    volume = torch.flip(reco_utils.rss(images).permute(2, 1, 0), dims=(1,))
    return volume.numpy()


def _image_error(image_sitk, images):
    array = torch.from_numpy(sitk.GetArrayFromImage(image_sitk))
    return relative_error(array, torch.from_numpy(_expected_image(images)))


def test_fft():
    images, kspace = make_phantom()
    image, recon_kspace, raw_kspace = _run(kspace, "fft")
    assert image.GetSize() == (32, 32, 4)
    assert raw_kspace.GetSize() == (32, 32, 4)
    assert _image_error(image, images) < 1e-5


def test_pocs_partial_fourier_on_lin():
    images, kspace = make_phantom()
    partial = kspace.clone()
    partial[:, :10] = 0
    zero_filled, *_ = _run(partial, "FFT")
    pocs, *_ = _run(partial, "POCS", pocs_dim=1, pocs_iterations=20)
    assert _image_error(pocs, images) < _image_error(zero_filled, images)


def test_grappa_integrated_acs():
    images, kspace = make_phantom()
    undersampled = kspace.clone()
    undersampled[:, 1::2] = 0
    undersampled[:, 10:22] = kspace[:, 10:22]
    image, *_ = _run(undersampled, "GRAPPA", kernelSize=(4, 1, 5))
    assert _image_error(image, images) < 0.1


def test_unknown_method():
    _, kspace = make_phantom()
    try:
        _run(kspace, "SENSE")
    except ValueError:
        return
    raise AssertionError("ValueError expected")
