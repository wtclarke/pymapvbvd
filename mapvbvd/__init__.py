from importlib.metadata import PackageNotFoundError, version

from mapvbvd.mapVBVD import mapVBVD

try:
    __version__ = version("pyMapVBVD")
except PackageNotFoundError:
    __version__ = "0+unknown"
