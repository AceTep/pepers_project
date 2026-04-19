"""
Utility functions for PDF parsing pipeline.
"""

import os
import logging
from pathlib import Path
from typing import List


def setup_logging(log_level=logging.INFO) -> logging.Logger:
    """
    Configure logging with console output.
    
    Args:
        log_level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("pdf_parser")
    logger.setLevel(log_level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def ensure_directory_exists(directory: str) -> Path:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Path to directory
        
    Returns:
        Path object of the directory
        
    Raises:
        IOError: If directory cannot be created
    """
    dir_path = Path(directory)
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    except Exception as e:
        raise IOError(f"Failed to create directory {directory}: {e}")


def get_pdf_files(input_dir: str) -> List[Path]:
    """
    Discover all PDF files in a directory.
    
    Args:
        input_dir: Path to directory containing PDFs
        
    Returns:
        List of Path objects for all .pdf files found
        
    Raises:
        FileNotFoundError: If input directory doesn't exist
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    if not input_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {input_dir}")
    
    pdf_files = sorted(input_path.glob("*.pdf"))
    return pdf_files


def get_relative_path(absolute_path: str, base_dir: str = None) -> str:
    """
    Convert absolute path to relative path from project root or specified base.
    
    Args:
        absolute_path: Absolute path to convert
        base_dir: Base directory for relative path (default: current working directory)
        
    Returns:
        Relative path as string
    """
    if base_dir is None:
        base_dir = os.getcwd()
    
    try:
        return str(Path(absolute_path).relative_to(base_dir))
    except ValueError:
        # Path is not relative to base_dir, return absolute
        return absolute_path


def sanitize_filename(filename: str) -> str:
    """
    Remove file extension and return clean base name.
    
    Args:
        filename: Original filename (e.g., 'document.pdf')
        
    Returns:
        Clean filename without extension (e.g., 'document')
    """
    return Path(filename).stem
