#!/usr/bin/env python3
"""
Document Chunking Script - Phase 1 of RAG Pipeline

Splits cleaned parsed documents into semantic chunks with metadata.
Prepares chunks for embedding and vector storage in Chroma.
"""

import os
import json
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    content: str
    source_file: str
    chunk_id: int
    chunk_count: int
    start_line: int
    end_line: int


class DocumentChunker:
    """Chunks documents from parsed/ directory for RAG pipeline."""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        input_dir: str = "parsed",
        output_dir: str = "chunks"
    ):
        """
        Initialize chunker with splitting parameters.
        
        Args:
            chunk_size: Target size of each chunk in tokens (approx 4 chars per token)
            chunk_overlap: Overlap between consecutive chunks for context preservation
            input_dir: Directory containing parsed .txt files
            output_dir: Directory to store chunked output
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize splitter - uses newlines, spaces as separators for semantic boundaries
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],  # Prefer semantic boundaries
            length_function=self._count_tokens,  # Use token approximation for sizing
        )
    
    @staticmethod
    def _count_tokens(text: str) -> int:
        """Approximate token count (4 chars ≈ 1 token)."""
        return len(text) // 4
    
    def chunk_single_file(self, filepath: Path) -> List[Chunk]:
        """
        Split a single document into chunks.
        
        Args:
            filepath: Path to .txt file to chunk
            
        Returns:
            List of Chunk objects with metadata
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into chunks using LangChain splitter
        text_chunks = self.splitter.split_text(content)
        
        chunks = []
        current_line = 1
        
        for chunk_id, chunk_text in enumerate(text_chunks):
            # Calculate approximate line positions
            lines_in_chunk = chunk_text.count('\n')
            end_line = current_line + lines_in_chunk
            
            chunk = Chunk(
                content=chunk_text,
                source_file=filepath.name,
                chunk_id=chunk_id,
                chunk_count=len(text_chunks),
                start_line=current_line,
                end_line=end_line
            )
            chunks.append(chunk)
            current_line = end_line + 1
        
        return chunks
    
    def chunk_all_documents(self) -> Dict[str, List[Chunk]]:
        """
        Process all parsed documents and return chunked results.
        
        Returns:
            Dictionary mapping source filenames to their chunks
        """
        all_chunks = {}
        txt_files = sorted(self.input_dir.glob("*.txt"))
        
        if not txt_files:
            print(f"⚠️  No .txt files found in {self.input_dir}/")
            return all_chunks
        
        print(f"📄 Chunking {len(txt_files)} documents...\n")
        
        for filepath in tqdm(txt_files, desc="Processing files"):
            try:
                chunks = self.chunk_single_file(filepath)
                all_chunks[filepath.name] = chunks
                print(f"  ✓ {filepath.name}: {len(chunks)} chunks")
            except Exception as e:
                print(f"  ✗ {filepath.name}: {e}")
        
        return all_chunks
    
    def save_chunks(self, all_chunks: Dict[str, List[Chunk]]) -> None:
        """
        Save chunks to JSONL format (one chunk per line).
        
        Args:
            all_chunks: Dictionary of chunks from chunk_all_documents()
        """
        output_file = self.output_dir / "chunks.jsonl"
        total_chunks = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for source_file, chunks in all_chunks.items():
                for chunk in chunks:
                    f.write(json.dumps(asdict(chunk)) + '\n')
                    total_chunks += 1
        
        print(f"\n✅ Saved {total_chunks} chunks to {output_file}")
        return output_file
    
    def save_chunk_metadata(self, all_chunks: Dict[str, List[Chunk]]) -> None:
        """
        Save summary metadata about chunks for reference.
        
        Args:
            all_chunks: Dictionary of chunks from chunk_all_documents()
        """
        metadata = {
            "total_documents": len(all_chunks),
            "total_chunks": sum(len(chunks) for chunks in all_chunks.values()),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "documents": {}
        }
        
        for source_file, chunks in all_chunks.items():
            total_tokens = sum(self._count_tokens(c.content) for c in chunks)
            metadata["documents"][source_file] = {
                "chunk_count": len(chunks),
                "total_tokens": total_tokens,
                "avg_chunk_size": total_tokens // len(chunks) if chunks else 0
            }
        
        metadata_file = self.output_dir / "chunk_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📊 Saved metadata to {metadata_file}\n")
        
        # Print summary
        print("Chunking Summary:")
        print(f"  Total documents: {metadata['total_documents']}")
        print(f"  Total chunks: {metadata['total_chunks']}")
        for doc, stats in metadata['documents'].items():
            print(f"    {doc}: {stats['chunk_count']} chunks (~{stats['avg_chunk_size']} tokens avg)")


def main():
    """Main entry point for chunking documents."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Chunk parsed documents for RAG pipeline"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="parsed",
        help="Input directory with parsed .txt files (default: parsed/)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="chunks",
        help="Output directory for chunks (default: chunks/)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Target chunk size in tokens (~4 chars per token, default: 500)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between chunks in tokens (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Create chunker and process documents
    chunker = DocumentChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        input_dir=args.input,
        output_dir=args.output
    )
    
    # Chunk all documents
    all_chunks = chunker.chunk_all_documents()
    
    if all_chunks:
        # Save chunks and metadata
        chunker.save_chunks(all_chunks)
        chunker.save_chunk_metadata(all_chunks)
        print("🚀 Ready for embedding phase!")
    else:
        print("❌ No chunks created - check input directory")


if __name__ == "__main__":
    main()
