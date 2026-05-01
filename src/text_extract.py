from pathlib import Path
from typing import Optional
import hashlib
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using PyMuPDF."""
    chunks = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            chunks.append(f"\n\n--- PAGE {page_num + 1} ---\n\n{page_text}")
    return "".join(chunks)

def _text_cache_key(path: Path) -> str:
    stat = path.stat()
    key = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def load_text(path: Path, cache_dir: Optional[Path] = None, use_cache: bool = True) -> str:
    """Load text from PDF or TXT file."""
    if path.suffix.lower() == '.pdf':
        if cache_dir is not None and use_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{_text_cache_key(path)}.txt"
            if cache_path.exists():
                return cache_path.read_text(encoding='utf-8')

            text = extract_text_from_pdf(path)
            cache_path.write_text(text, encoding='utf-8')
            return text
        return extract_text_from_pdf(path)
    elif path.suffix.lower() == '.txt':
        return path.read_text(encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Supported: .pdf, .txt")
