from importlib.metadata import PackageNotFoundError, version

from . import dspec

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # pragma: no cover
    try:
        __version__ = version("hera_filters")
    except PackageNotFoundError:
        # package is not installed
        __version__ = "unknown"
