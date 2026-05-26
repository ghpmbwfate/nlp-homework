def __getattr__(name):
    if name == "LLMGenerator":
        from .generator import LLMGenerator as _LLMGenerator
        return _LLMGenerator
    if name == "PostProcessor":
        from .postprocess import PostProcessor
        return PostProcessor
    if name == "SelfRAG":
        from .self_rag import SelfRAG
        return SelfRAG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
