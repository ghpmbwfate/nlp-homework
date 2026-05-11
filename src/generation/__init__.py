def __getattr__(name):
    if name == "VLMGenerator":
        from .generator import VLMGenerator as _VLMGenerator
        return _VLMGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
