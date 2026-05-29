def __getattr__(name):
    if name == "ChartExtractor":
        from .chart_extractor import ChartExtractor
        return ChartExtractor
    if name == "merge_with_page_content":
        from .chart_extractor import merge_with_page_content
        return merge_with_page_content
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

