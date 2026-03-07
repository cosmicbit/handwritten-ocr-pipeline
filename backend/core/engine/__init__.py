__all__ = [
    "DEFAULT_TEACHER_PDF",
    "DEFAULT_STUDENT_PDF",
    "run_pipeline",
]


def __getattr__(name):
    if name in __all__:
        from . import main

        return getattr(main, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
