def __getattr__(name):
    if name == "VLMGenerator":
        from .generator import VLMGenerator as _VLMGenerator
        return _VLMGenerator
    if name == "PostProcessor":
        from .postprocess import PostProcessor
        return PostProcessor
    if name == "SelfRAG":
        from .self_rag import SelfRAG
        return SelfRAG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
