"""Support methods and classes for the Biodiversity studies on Mgnify metagenomes."""

__version__ = "0.1.0"

from . import utils, metacomp
from .mgnify_oop import TaxonomyTable, AbundanceTable
from . import stats, taxonomy


def hello() -> str:
    """Return a greeting message."""
    return "Hello from mgnify-methods!"
