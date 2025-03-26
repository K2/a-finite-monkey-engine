"""
Utilities for managing package dependencies
"""

import importlib
import subprocess
import sys
from typing import List, Dict, Tuple, Optional, Set

from .logger import logger

def is_package_installed(package_name: str) -> bool:
    """
    Check if a package is installed
    
    Args:
        package_name: Name of the package
        
    Returns:
        True if the package is installed, False otherwise
    """
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def ensure_packages(required_packages: List[str], raise_error: bool = False) -> bool:
    """
    Ensure that required packages are installed
    
    Args:
        required_packages: List of required package names
        raise_error: Whether to raise an error if packages are missing
        
    Returns:
        True if all packages are installed, False otherwise
    """
    missing = []
    for package in required_packages:
        if not is_package_installed(package):
            missing.append(package)
            
    if not missing:
        return True
        
    # Log the missing packages
    logger.warning(f"Missing required packages: {', '.join(missing)}")
    
    if raise_error:
        raise ImportError(f"Missing required packages: {', '.join(missing)}")
    
    # Ask if user wants to install packages
    print(f"The following packages are required but not installed: {', '.join(missing)}")
    response = input("Do you want to install them? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        try:
            # Install missing packages
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            logger.info("Successfully installed missing packages")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install packages: {e}")
            if raise_error:
                raise
            return False
    else:
        logger.warning("Required packages were not installed")
        return False
