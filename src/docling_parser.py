"""
PDF parsing module using IBM Docling.

Handles conversion of PDF files to structured format with metadata extraction.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Disable CUDA to avoid GPU compatibility issues (use CPU instead)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from docling.document_converter import DocumentConverter
from docling.document_converter import ConversionStatus
from .content_cleaner import clean_content_for_rag, get_cleaning_stats


logger = logging.getLogger("pdf_parser")


class DoclingParseError(Exception):
    """Custom exception for Docling parsing errors."""
    pass


def parse_single_pdf(pdf_path: str) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Parse a single PDF file using Docling.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Tuple of (ConversionResult, metadata_dict)
        If parsing fails, returns (None, error_info)
        
    Raises:
        DoclingParseError: If file doesn't exist or is not a PDF
    """
    pdf_path_obj = Path(pdf_path)
    
    if not pdf_path_obj.exists():
        raise DoclingParseError(f"PDF file not found: {pdf_path}")
    
    if pdf_path_obj.suffix.lower() != ".pdf":
        raise DoclingParseError(f"File is not a PDF: {pdf_path}")
    
    try:
        logger.info(f"Parsing PDF: {pdf_path_obj.name}")
        
        converter = DocumentConverter()
        result = converter.convert(source=pdf_path)
        
        if result.status == ConversionStatus.SUCCESS:
            logger.info(f"Successfully parsed: {pdf_path_obj.name}")
            return result, None
        else:
            error_msg = f"Docling conversion failed with status {result.status}"
            logger.error(f"{pdf_path_obj.name}: {error_msg}")
            return None, {"error": error_msg, "status": str(result.status)}
    
    except Exception as e:
        error_msg = f"Error parsing PDF: {str(e)}"
        logger.error(f"{pdf_path_obj.name}: {error_msg}")
        return None, {"error": error_msg, "exception_type": type(e).__name__}


def extract_text_and_metadata(doc_result: Any, pdf_path: str) -> Dict[str, Any]:
    """
    Extract text content and metadata from parsed Docling document.
    
    Args:
        doc_result: ConversionResult from Docling
        pdf_path: Original PDF file path (for metadata)
        
    Returns:
        Dictionary with:
            - metadata: title, author, page_count, creation_date, source_file
            - content: markdown-formatted text (cleaned for RAG)
            - content_stats: information about cleaning performed
    """
    pdf_path_obj = Path(pdf_path)
    
    try:
        # Extract markdown content
        markdown_content = doc_result.document.export_to_markdown()
        
        # Clean content for RAG (remove references, metadata, etc.)
        cleaned_content = clean_content_for_rag(markdown_content, verbose=True)
        cleaning_stats = get_cleaning_stats(markdown_content, cleaned_content)
        
        # Extract metadata
        doc_meta = doc_result.document.meta if hasattr(doc_result.document, 'meta') else {}
        
        # Get page count
        page_count = None
        if hasattr(doc_result.document, 'pages'):
            page_count = len(doc_result.document.pages)
        
        # Build metadata dictionary
        metadata = {
            "source_file": pdf_path_obj.name,
            "source_path": str(pdf_path_obj),
            "parsed_timestamp": datetime.now().isoformat(),
            "title": getattr(doc_meta, 'title', None) if hasattr(doc_meta, 'title') else None,
            "author": getattr(doc_meta, 'author', None) if hasattr(doc_meta, 'author') else None,
            "creation_date": getattr(doc_meta, 'creation_date', None) if hasattr(doc_meta, 'creation_date') else None,
            "page_count": page_count,
        }
        
        return {
            "metadata": metadata,
            "content": cleaned_content,
            "content_stats": cleaning_stats,
        }
    
    except Exception as e:
        logger.error(f"Error extracting content from {pdf_path_obj.name}: {str(e)}")
        raise DoclingParseError(f"Failed to extract content: {str(e)}")


def convert_markdown_to_plain_text(markdown_content: str) -> str:
    """
    Convert markdown content to plain text.
    
    Removes markdown formatting while preserving readable structure.
    
    Args:
        markdown_content: Markdown-formatted text
        
    Returns:
        Plain text version
    """
    text = markdown_content
    
    # Remove markdown links [text](url) -> text
    import re
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove markdown bold/italic markers
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)  # **bold** -> bold
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)      # *italic* -> italic
    text = re.sub(r'__([^_]+)__', r'\1', text)       # __bold__ -> bold
    text = re.sub(r'_([^_]+)_', r'\1', text)         # _italic_ -> italic
    
    # Remove code block markers but keep content
    text = re.sub(r'```[a-z]*\n', '', text)          # Remove opening code fence
    text = re.sub(r'\n```', '', text)                 # Remove closing code fence
    
    # Clean up heading markers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Clean up list markers
    text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove excessive whitespace but preserve paragraph breaks
    lines = text.split('\n')
    cleaned_lines = [line.rstrip() for line in lines]
    text = '\n'.join(cleaned_lines)
    
    # Replace multiple blank lines with single blank line
    text = re.sub(r'\n\n\n+', '\n\n', text)
    
    return text.strip()


def save_parsed_output(
    parsed_data: Dict[str, Any],
    output_dir: str,
    base_filename: str
) -> Tuple[bool, Optional[str]]:
    """
    Save parsed content to JSON and TXT files.
    
    Args:
        parsed_data: Dictionary with 'metadata', 'content', and 'content_stats' keys
        output_dir: Directory to save output files
        base_filename: Base filename without extension
        
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_file = output_path / f"{base_filename}.json"
    txt_file = output_path / f"{base_filename}.txt"
    
    try:
        # Save JSON with full structure including cleaning stats
        json_data = {
            "metadata": parsed_data["metadata"],
            "content": parsed_data["content"],
            "cleaning_stats": parsed_data.get("content_stats", {}),
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved JSON output: {json_file.name}")
        
        # Save TXT with plain text (already cleaned)
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(parsed_data["content"])
        
        logger.info(f"Saved TXT output: {txt_file.name}")
        
        return True, None
    
    except Exception as e:
        error_msg = f"Failed to save output for {base_filename}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def parse_and_save_pdf(pdf_path: str, output_dir: str) -> Tuple[bool, Optional[str]]:
    """
    Parse a PDF and save results to JSON and TXT files.
    
    Orchestrates the full parsing and saving workflow.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save parsed outputs
        
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    pdf_path_obj = Path(pdf_path)
    base_filename = pdf_path_obj.stem
    
    try:
        # Parse PDF
        doc_result, parse_error = parse_single_pdf(pdf_path)
        
        if doc_result is None:
            error_msg = parse_error.get("error", "Unknown parsing error") if parse_error else "Unknown error"
            return False, error_msg
        
        # Extract text and metadata
        parsed_data = extract_text_and_metadata(doc_result, pdf_path)
        
        # Save outputs
        success, save_error = save_parsed_output(parsed_data, output_dir, base_filename)
        
        if not success:
            return False, save_error
        
        return True, None
    
    except Exception as e:
        error_msg = f"Unexpected error processing {pdf_path_obj.name}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
