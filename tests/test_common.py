"""Tests for common.py utilities."""

import sys
import os

# Add the parent directory to the path so we can import common.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.common import packages, PACKAGES

def test_packages_returns_versioned_packages():
    """Test that packages() returns all the versioned packages."""
    result = packages()
    
    # Check that all versioned packages are included
    for pkg, version in PACKAGES.items():
        assert f"{pkg}={version}" in result
    
    # Check that the length is correct (no extras)
    assert len(result) == len(PACKAGES)

def test_packages_includes_additional_packages():
    """Test that packages() includes additional packages."""
    additional = ["pytest", "flask"]
    result = packages(additional)
    
    # Check that all versioned packages are included
    for pkg, version in PACKAGES.items():
        assert f"{pkg}={version}" in result
        
    # Check that additional packages are included
    for pkg in additional:
        assert pkg in result
    
    # Check that the length is correct
    assert len(result) == len(PACKAGES) + len(additional)