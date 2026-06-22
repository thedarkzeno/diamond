from .denoiser import Denoiser, DenoiserConfig, SigmaDistributionConfig
from .inner_model import InnerModelConfig
from .diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig

__all__ = [
    "Denoiser",
    "DenoiserConfig",
    "SigmaDistributionConfig",
    "InnerModelConfig",
    "DiffusionSampler",
    "DiffusionSamplerConfig",
    "FlowDenoiser",
    "FlowDenoiserConfig",
    "SanaInnerModelConfig",
    "LatentSampler",
    "LatentSamplerConfig",
]

_LAZY_IMPORTS = {
    "FlowDenoiser": (".flow_denoiser", "FlowDenoiser"),
    "FlowDenoiserConfig": (".flow_denoiser", "FlowDenoiserConfig"),
    "SanaInnerModelConfig": (".sana_inner_model", "SanaInnerModelConfig"),
    "LatentSampler": (".latent_sampler", "LatentSampler"),
    "LatentSamplerConfig": (".latent_sampler", "LatentSamplerConfig"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module_name, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
