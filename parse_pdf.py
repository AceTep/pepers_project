#!/usr/bin/env python3
"""
Main script for PDF parsing pipeline.

Processes all PDFs in the input directory using IBM Docling,
extracts structured content and metadata, and saves results
to JSON and TXT files in the output directory.

Usage:
    python parse_pdf.py
"""

import os
# Disable CUDA to avoid GPU compatibility issues (use CPU instead)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
import logging
from pathlib import Path

from src.utils import (
    setup_logging,
    ensure_directory_exists,
    get_pdf_files,
)
from src.docling_parser import parse_and_save_pdf


def main():
    """
    Main entry point for PDF parsing pipeline.
    
    Processes all PDFs from 'data/' directory and saves
    parsed outputs to 'parsed/' directory.
    """
    # Setup logging
    logger = setup_logging(log_level=logging.INFO)
    logger.info("PDF Parsing Pipeline Started")
    
    # Define paths relative to project root
    project_root = Path(__file__).parent
    input_dir = project_root / "data"
    output_dir = project_root / "parsed"
    
    try:
        # Ensure output directory exists
        ensure_directory_exists(str(output_dir))
        logger.info(f"Output directory ready: {output_dir}")
        
        # Discover PDF files
        pdf_files = get_pdf_files(str(input_dir))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return 0
        
        logger.info(f"Found {len(pdf_files)} PDF(s) to process")
        print()
        
        # Process each PDF
        successful = 0
        failed = 0
        failed_files = []
        
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
            
            success, error_msg = parse_and_save_pdf(
                str(pdf_path),
                str(output_dir)
            )
            
            if success:
                successful += 1
                logger.info(f"✓ Successfully parsed {pdf_path.name}")
            else:
                failed += 1
                failed_files.append((pdf_path.name, error_msg))
                logger.error(f"✗ Failed to parse {pdf_path.name}: {error_msg}")
            
            print()
        
        # Print summary
        logger.info("=" * 60)
        logger.info("PARSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total PDFs processed: {len(pdf_files)}")
        logger.info(f"Successfully parsed:  {successful}")
        logger.info(f"Failed:               {failed}")
        logger.info(f"Output directory:     {output_dir}")
        
        if failed_files:
            logger.warning("\nFailed files:")
            for filename, error in failed_files:
                logger.warning(f"  - {filename}: {error}")
        
        logger.info("=" * 60)
        logger.info("PDF Parsing Pipeline Completed")
        
        return 0 if failed == 0 else 1
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
