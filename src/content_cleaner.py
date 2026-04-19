"""
Content cleaning module for RAG pipeline.

Removes junk data from parsed PDF content including:
- References/bibliography sections
- Author affiliations and metadata
- Headers with DOI, citations, publication info
- Repetitive formatting elements
- OCR artifacts and placeholders
"""

import re
import logging
from typing import List, Tuple

logger = logging.getLogger("pdf_parser")


# Patterns to identify and remove junk sections
JUNK_PATTERNS = [
    # References sections with various names
    (r"(?:^|\n)(?:REFERENCES|BIBLIOGRAPHY|CITATIONS|WORKS CITED|REFERENCES AND NOTES|FURTHER READING)[\s\n].*?(?=(?:^|\n)[A-Z][A-Z]+[\s\n]|$)", 
     "references_section"),
    
    # DOI, URLs, and citation links (can appear as standalone)
    (r"https?://doi\.org/[^\s\n]+", "doi_url"),
    (r"DOI[:\s]+(?:https://)?[\w\./]+", "doi_label"),
    (r"v\d+z\d+s\d+", "arxiv_like_id"),
    
    # Author and affiliation blocks
    (r"(?:^|\n)(?:\d+\s+)?(?:Universidad|University|Institute|School|Department)[\w\s&,\.]*(?:\n|$)",
     "affiliation"),
    (r"Correspondence:\s*[^\n]+\n", "correspondence"),
    (r"^(?:\d+\s+)?[A-Z][a-z]+(?:,\s+[A-Z][a-z]+)*\s+\d+\s*(?:\(|\[)?[A-Za-z]{2,}\s*(?:,\s*\d+|Province|State|Region)", 
     "author_affiliation"),
    
    # Academic publication metadata
    (r"(?:Received|Accepted|Published|Submitted)(?:\s+on?-?line)?[\s:]+[A-Za-z]+ \d+,? (?:202[0-9]|20\d\d|[0-9]+)\.?",
     "publication_date"),
    (r"(?:Academic Editor|Associate Editor|Guest Editor|Editor)\s*:?\s*[^\n]*\n",
     "editor_info"),
    (r"(?:Special|Theme|Issue)\s+(?:Issue|Section|Editor)\s*:?\s*[^\n]*\n",
     "issue_info"),
    
    # License and copyright blocks
    (r"(?:This work is licensed|Licensed under|© \d+|Copyright © \d+)[^\n]*(?:\n[^\n]*)*",
     "license_block"),
    (r"Creative Commons[^\n]*(?:\n[^\n]*)*",
     "cc_license"),
    
    # Acknowledgments section
    (r"(?:^|\n)(?:ACKNOWLEDGMENTS?|THANKS?|ACKNOWLEDGEMENTS?|AGRADECIMIENTOS)[\s\n].*?(?=(?:^|\n)[A-Z][A-Z]+[\s\n]|$)",
     "acknowledgments"),
    
    # Funding information
    (r"(?:Funding|Grant|Financed by|Supported by)[\s:]+[^\n]*(?:\n[^\n]*)*",
     "funding_info"),
    
    # Conflict of interest statements
    (r"(?:^|\n)(?:CONFLICT OF INTEREST|CONFLICTS OF INTEREST|DISCLOSURE)[\s\n].*?(?=(?:^|\n)[A-Z][A-Z]+[\s\n]|$)",
     "conflict_of_interest"),
    
    # Image/Figure/Table placeholder patterns
    (r"<!--\s*image\s*-->", "image_placeholder"),
    (r"\[image\]|\[figure\]|\[Figure \d+\]|\[Figure\]", "image_tag"),
    (r"\[Table \d+\]|\[TABLE\]", "table_tag"),
    
    # Page numbers and footers
    (r"^— \d+ —$", "page_separator"),
    (r"(?:pp|Pages?)\s*:?\s*\d+[–-]\d+", "page_range"),
    
    # Empty or near-empty lines with excessive spacing (OCR artifacts)
    (r"\n{3,}", "\n\n"),
    
    # Author initials and very short lines that are likely metadata
    (r"^[A-Z]\. [A-Z]\.(?:\s+[A-Z]\.)*$", "author_initials"),
    
    # Abstract markers in various languages
    (r"(?:^|\n)(?:ABSTRACT|RESUMEN|RÉSUMÉ|ZUSAMMENFASSUNG|ABSTRACT AND KEY WORDS)[\s\n]",
     "abstract_marker"),
    
    # Agro/journal specific metadata
    (r"Agro Productividad.*?(?:\n|$)", "journal_name"),
    (r"\d+\s*\(\s*\d+\s*\).*?pp\s*:?\s*\d+[–-]\d+", "journal_issue"),
]


def remove_header_metadata(text: str) -> str:
    """
    Remove header metadata that appears before main content.
    
    Removes content from beginning until we find a main content marker
    (like ABSTRACT, INTRODUCTION, etc.)
    
    Args:
        text: Full extracted text from PDF
        
    Returns:
        Text with leading metadata removed
    """
    # Markers that indicate start of actual content
    content_start_markers = [
        r"^#+\s*ABSTRACT\b",
        r"^ABSTRACT\b",
        r"^#+\s*RESUMEN\b",
        r"^RESUMEN\b",
        r"^#+\s*INTRODUCTION\b",
        r"^INTRODUCTION\b",
        r"^#+\s*BACKGROUND\b",
        r"^BACKGROUND\b",
        r"^#+\s*METHODS?\b",
        r"^METHODS?\b",
    ]
    
    lines = text.split('\n')
    start_index = 0
    
    # Find first occurrence of content marker
    for i, line in enumerate(lines):
        for marker in content_start_markers:
            if re.match(marker, line.strip(), re.IGNORECASE):
                start_index = i
                logger.debug(f"Found content start marker at line {i}: {line.strip()[:50]}")
                return '\n'.join(lines[i:])
    
    # If no marker found, return original (might be a document without abstract)
    return text


def extract_main_sections(text: str) -> str:
    """
    Extract main content sections and discard references/metadata.
    
    Strategy: Remove header metadata, keep content before major section breaks 
    (like "REFERENCES"), and remove trailing metadata sections.
    
    Args:
        text: Full extracted text from PDF
        
    Returns:
        Cleaned text with junk removed
    """
    # First remove header metadata
    text = remove_header_metadata(text)
    
    # Find common section breaks that indicate start of junk content
    # Updated to handle markdown headings (## REFERENCES, etc.)
    junk_section_markers = [
        r"^#+\s*REFERENCES\b",
        r"^REFERENCES\b",
        r"^#+\s*BIBLIOGRAPHY\b",
        r"^BIBLIOGRAPHY\b",
        r"^#+\s*BIBLIOGRAPHY\s+CHAPTER\s+\d+\b",
        r"^BIBLIOGRAPHY\s+CHAPTER\s+\d+\b",
        r"^#+\s*CITATIONS\b",
        r"^CITATIONS\b",
        r"^#+\s*WORKS\s+CITED\b",
        r"^WORKS\s+CITED\b",
        r"^#+\s*REFERENCES\s+AND\s+NOTES\b",
        r"^REFERENCES\s+AND\s+NOTES\b",
        r"^#+\s*CONFLICT\s+OF\s+INTEREST\b",
        r"^CONFLICT\s+OF\s+INTEREST\b",
        r"^#+\s*ACKNOWLEDGMENTS?\b",
        r"^ACKNOWLEDGMENTS?\b",
        r"^#+\s*FUNDING\s+INFORMATION\b",
        r"^FUNDING\s+INFORMATION\b",
        r"^#+\s*SUPPLEMENTARY\s+MATERIAL\b",
        r"^SUPPLEMENTARY\s+MATERIAL\b",
        r"^#+\s*APPENDIX\b",
        r"^APPENDIX\b",
        r"^#+\s*EDITORIAL\s+STAFF\b",
        r"^EDITORIAL\s+STAFF\b",
        r"^#+\s*AUTHOR\s+(?:INFORMATION|DETAILS|AFFILIATIONS)\b",
        r"^AUTHOR\s+(?:INFORMATION|DETAILS|AFFILIATIONS)\b",
        r"^#+\s*CONTRIBUTORS\b",
        r"^CONTRIBUTORS\b",
    ]
    
    # Split by sections and keep only the main content
    lines = text.split('\n')
    cutoff_index = len(lines)
    
    for i, line in enumerate(lines):
        for marker in junk_section_markers:
            if re.match(marker, line.strip(), re.IGNORECASE):
                cutoff_index = i
                logger.debug(f"Found junk section marker at line {i}: {line.strip()[:50]}")
                break
        if cutoff_index < len(lines):
            break
    
    # Keep content up to cutoff
    main_content = '\n'.join(lines[:cutoff_index])
    
    return main_content


def remove_junk_patterns(text: str) -> str:
    """
    Remove specific junk patterns from text.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    cleaned = text
    
    # Apply removal patterns (in order of priority)
    patterns_to_apply = [
        # Remove image/table placeholders first
        (r"<!--\s*image\s*-->\s*\n?", ""),
        (r"\[(?:image|figure|Figure|TABLE|Table)\s*\d*\]\s*\n?", ""),
        
        # Remove metadata headers
        (r"DOI\s*\n?", ""),
        (r"https?://doi\.org/[^\s\n]*", ""),
        (r"v\d+z\d+s\d+", ""),
        
        # Remove publication info headers (Creative Commons, journal info)
        (r"(?:This work is licensed|Creative Commons[^\n]*\n?)", ""),
        (r"(?:Agro Productividad.*?\n?)", ""),
        (r"Academic Editor.*?\n", ""),
        (r"Received\s*:?.*?\n", ""),
        (r"Accepted\s*:?.*?\n", ""),
        (r"Published.*?\n", ""),
        (r"(?:Associate|Guest)?\s*Editor\s*:.*?\n", ""),
        
        # Remove author affiliation lines (university addresses)
        (r"^\d+\s+(?:Universidad|University|Institute|Department|School).*?\n", ""),
        (r"Correspondence:.*?\n", ""),
        
        # Remove standalone author initials/names
        (r"^[A-Z]\.\s+[A-Z]\.(?:\s+[A-Z]\.)*\s*\n", ""),
        
        # Remove license blocks
        (r"This publication is.*?Creative Commons.*?\n", ""),
        
        # Remove multiple consecutive blank lines
        (r"\n{3,}", "\n\n"),
    ]
    
    for pattern, replacement in patterns_to_apply:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.MULTILINE | re.IGNORECASE)
    
    return cleaned


def remove_trailing_metadata(text: str) -> str:
    """
    Remove trailing metadata sections that commonly appear at document end.
    
    Args:
        text: Text to clean
        
    Returns:
        Text with trailing metadata removed
    """
    # Identify sections that are typically metadata at the end
    # Updated to handle both markdown headings and plain section headers
    trailing_markers = {
        "references": r"(?:^|\n)#{0,6}\s*(?:REFERENCES|BIBLIOGRAPHY|WORKS CITED|CITATIONS)\b",
        "acknowledgments": r"(?:^|\n)#{0,6}\s*(?:ACKNOWLEDGMENTS?|THANKS?)\b",
        "conflict": r"(?:^|\n)#{0,6}\s*(?:CONFLICT OF INTEREST|CONFLICTS OF INTEREST)\b",
        "funding": r"(?:^|\n)#{0,6}\s*(?:FUNDING|GRANT|FINANCIAL SUPPORT)\b",
        "supplementary": r"(?:^|\n)#{0,6}\s*(?:SUPPLEMENTARY|APPENDIX|APPENDICES)\b",
    }
    
    # Find the earliest occurrence of any trailing marker
    earliest_pos = len(text)
    earliest_marker = None
    
    for marker_name, pattern in trailing_markers.items():
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()
            earliest_marker = marker_name
    
    # If we found a marker, keep only content before it
    if earliest_marker and earliest_pos > 0:
        text = text[:earliest_pos]
        logger.debug(f"Removed trailing section: {earliest_marker}")
    
    return text


def clean_whitespace(text: str) -> str:
    """
    Clean up excessive whitespace and formatting artifacts.
    
    Args:
        text: Text to clean
        
    Returns:
        Text with clean whitespace
    """
    # Remove leading/trailing whitespace from lines
    lines = [line.rstrip() for line in text.split('\n')]
    
    # Remove empty lines at start
    while lines and not lines[0].strip():
        lines.pop(0)
    
    # Remove empty lines at end
    while lines and not lines[-1].strip():
        lines.pop()
    
    # Join and collapse multiple blank lines
    text = '\n'.join(lines)
    text = re.sub(r'\n\n\n+', '\n\n', text)
    
    return text


def clean_content_for_rag(text: str, verbose: bool = False) -> str:
    """
    Main cleaning function that applies all cleaning strategies.
    
    Args:
        text: Raw extracted text from PDF
        verbose: Log cleaning steps taken
        
    Returns:
        Cleaned text suitable for RAG
    """
    original_length = len(text)
    
    # Step 1: Extract main sections (remove major junk blocks)
    text = extract_main_sections(text)
    if verbose:
        removed = original_length - len(text)
        if removed > 0:
            logger.debug(f"Extracted main sections: removed {removed} chars")
    
    # Step 2: Remove trailing metadata
    text = remove_trailing_metadata(text)
    if verbose:
        removed = original_length - len(text)
        if removed > 0:
            logger.debug(f"Removed trailing metadata: removed ~{removed} chars total")
    
    # Step 3: Remove junk patterns
    text = remove_junk_patterns(text)
    
    # Step 4: Clean whitespace
    text = clean_whitespace(text)
    
    final_length = len(text)
    percent_removed = ((original_length - final_length) / original_length * 100) if original_length > 0 else 0
    
    logger.info(f"Content cleaned: {original_length} → {final_length} chars ({percent_removed:.1f}% removed)")
    
    return text


def get_cleaning_stats(original: str, cleaned: str) -> dict:
    """
    Get statistics about what was cleaned.
    
    Args:
        original: Original text
        cleaned: Cleaned text
        
    Returns:
        Dictionary with cleaning statistics
    """
    return {
        "original_chars": len(original),
        "cleaned_chars": len(cleaned),
        "removed_chars": len(original) - len(cleaned),
        "percent_removed": ((len(original) - len(cleaned)) / len(original) * 100) if len(original) > 0 else 0,
        "original_lines": len(original.splitlines()),
        "cleaned_lines": len(cleaned.splitlines()),
        "lines_removed": len(original.splitlines()) - len(cleaned.splitlines()),
    }
