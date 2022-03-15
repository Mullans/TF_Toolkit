# NOTE - requires scipy, scikit-image, numpy, opencv2, tensorflow, and tqdm
# from .blurig import BlurIG
# from .saliencymask import SaliencyMask
# from .xrai import XRAI
import warnings
already_warned = False
try:
    import tensorflow
except ImportError:
    warnings.warn("Missing tensorflow install - most of this won't work")
    already_warned = True

try:
    from .gradcam import GradCAM, GradCAMPlus
    # from .contribution import twophase_attribution, multilayer_attribution
    from .occlusion import Occlusion
except ImportError as e:
    if not already_warned:
        raise e
