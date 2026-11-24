"""MRTwin reconstruction core (Siemens twix raw data).

Reconstructs Siemens ``.dat`` raw data (FFT, POCS partial Fourier, GRAPPA)
into SimpleITK volumes. Used by the 3D Slicer extension ``MRTwixReconstruct``.

Public API::

    from MRTwinReconstructCoreLib import MRTwinReconstructLogic
    image, kspace, raw_kspace = MRTwinReconstructLogic().runReconstruction(
        "meas.dat", "FFT")
"""

__version__ = "0.2.0"

from .MRTwinReconstructCore import METHODS, MRTwinReconstructLogic
from .RecoMRTwin import RecoMRTwin

__all__ = ["MRTwinReconstructLogic", "RecoMRTwin", "METHODS", "__version__"]
