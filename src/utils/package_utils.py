"""
Utility module to handle external packages with warnings or issues.
"""

import warnings
import importlib

def import_pretty_midi():
    """
    Import pretty_midi while suppressing the pkg_resources deprecation warning.
    
    Returns:
        The pretty_midi module
    """
    # Suppress the specific pkg_resources deprecation warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", 
                               message="pkg_resources is deprecated as an API", 
                               category=UserWarning)
        import pretty_midi
        return pretty_midi