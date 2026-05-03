"""
Validation utilities for Vnstock Installer
Argument parsers and validators
"""

import argparse
import json
import logging
import os


def argparse_file_exists(filename):
    """Validate that a file exists"""
    if not os.path.isfile(filename):
        raise argparse.ArgumentTypeError(
            f'File does not exist: {filename}'
        )
    return filename


def argparse_file_json(filename):
    """Validate that a file exists and is valid JSON"""
    if filename is None:
        return None
    
    if not os.path.isfile(filename):
        raise argparse.ArgumentTypeError(
            f'File does not exist: {filename}'
        )
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            json.load(f)
        return filename
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(
            f'File is not valid JSON: {filename} - {e}'
        )


def argparse_logging_level(level_str):
    """Validate logging level"""
    level_str = level_str.upper()
    if level_str not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        raise argparse.ArgumentTypeError(
            f'Invalid logging level: {level_str}'
        )
    return level_str


def validate_python_version(version_str):
    """Validate Python version string (e.g., '3.10')"""
    try:
        major, minor = version_str.split('.')[:2]
        major = int(major)
        minor = int(minor)
        
        if major != 3 or minor < 10:
            return False
        
        return True
    except (ValueError, IndexError):
        return False


def validate_api_key(api_key):
    """Validate API key format"""
    if not api_key:
        return False, "API key is empty"
    
    if len(api_key) < 10:
        return False, "API key is too short"
    
    # Add more validation as needed
    return True, "Valid"


def validate_venv_path(path_str):
    """Validate virtual environment path"""
    if not path_str:
        return True, "Using default venv path"
    
    expanded_path = os.path.expanduser(path_str)
    
    # Check if parent directory exists
    parent_dir = os.path.dirname(expanded_path) or os.getcwd()
    if not os.path.exists(parent_dir):
        return False, f"Parent directory does not exist: {parent_dir}"
    
    return True, "Valid path"
