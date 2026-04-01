import os
import json
import asyncio
import re
import math
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from .config import ChunkingConfig, ChunkingStrategy

# Library imports
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
try:
    import mammoth
except ImportError:
    mammoth = None
try:
    import openpyxl
except ImportError:
    openpyxl = None

try:
    import pysbd
    _segmenter = pysbd.Segmenter(language="en", clean=False)
except ImportError:
    _segmenter = None

class DocumentProcessor:
    def __init__(self, config: ChunkingConfig, agentic_llm=None):
        self.config = config
        self.agentic_llm = agentic_llm
        self._executor = ThreadPoolExecutor()
        self._last_pages = None

    async def load_document(self, file_path: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._load_sync, file_path)

    def _load_sync(self, file_path: str) -> str:
        self._last_pages = None
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            if not PdfReader: raise ImportError("pypdf not installed")
            reader = PdfReader(file_path)
            pages = [page.extract_text() or "" for page in reader.pages]
            self._last_pages = [p for p in pages if p.strip()]
            return "\n".join(pages)
        
        elif ext == '.docx':
            if not mammoth: raise ImportError("mammoth not installed")
            with open(file_path, "rb") as docx_file:
                result = mammoth.extract_raw_text(docx_file)
                return result.value
        
        elif ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif ext in ['.xlsx', '.xls']:
            if not openpyxl: raise ImportError("openpyxl not installed")
            wb = openpyxl.load_workbook(file_path, data_only=True)
            sheet = wb.active
            text = []
            for row in sheet.iter_rows(values_only=True):
                text.append(" ".join([str(c) for c in row if c is not None]))
            return "\n".join(text)
            
        raise ValueError(f"Unsupported file extension: {ext}")

    async def process(self, text: str) -> List[str]:
        if self.config.strategy == ChunkingStrategy.AGENTIC:
            return await self.agentic_split(text)
        return self.recursive_split(text)

    def _entropy(self, s: str) -> float:
        if not s:
            return 0.0
        freq = {}
        for ch in s:
            freq[ch] = freq.get(ch, 0) + 1
        length = len(s)
        H = 0.0
        for count in freq.values():
            p = count / length
            H += -p * math.log2(p)
        return H

    def recursive_split(self, text: str) -> List[str]:
        size_chars = max(500, int(self.config.chunk_size) if self.config.chunk_size else 1000)
        base_overlap = max(0, int(self.config.chunk_overlap) if self.config.chunk_overlap else 200)
        
        # 1. Page Splitting (for PDFs)
        if self._last_pages:
            top_level_splits = self._last_pages
        else:
            # 2. Markdown/Format Splitting
            # Split by headers, code blocks, or tables
            top_level_splits = re.split(r'(?m)^(?:#{1,6}\s+.*|```[\s\S]*?```|\|.*\|.*\|)', text)
            top_level_splits = [s.strip() for s in top_level_splits if s.strip()]
            if not top_level_splits:
                top_level_splits = [text]

        final_chunks = []
        current_chunk = ""

        for block in top_level_splits:
            # 3. Sentence Splitting
            if _segmenter:
                sentences = _segmenter.segment(block)
            else:
                # Fallback to improved regex that avoids decimals and common abbreviations
                sentences = re.split(r'(?<!\d)\.(?!\d)(?=\s+[A-Z])|(?<=[!?])\s+', block)

            for s in sentences:
                candidate = f"{current_chunk} {s}".strip() if current_chunk else s.strip()
                if len(candidate) >= size_chars:
                    if current_chunk:
                        entropy = self._entropy(current_chunk)
                        overlap = min(base_overlap + math.floor(entropy * 50), math.floor(size_chars / 3))
                        final_chunks.append(current_chunk)
                        current_chunk = current_chunk[-overlap:] + " " + s.strip()
                    else:
                        # Single sentence longer than chunk size - force split or keep as is
                        final_chunks.append(s.strip())
                        current_chunk = ""
                else:
                    current_chunk = candidate

        if current_chunk:
            final_chunks.append(current_chunk)

        return final_chunks

    def compute_chunk_metadata(self, file_path: str, raw_text: str, chunks: List[str]) -> List[dict]:
        ext = os.path.splitext(file_path)[1].lower()
        title = os.path.basename(file_path)
        positions = []
        cursor = 0
        for c in chunks:
            idx = raw_text.find(c, cursor)
            start = idx if idx >= 0 else 0
            end = start + len(c)
            positions.append((start, end))
            cursor = end
        pages_meta = None
        if ext == '.pdf' and isinstance(self._last_pages, list):
            lens = [len(p) for p in self._last_pages]
            cum = []
            acc = 0
            for l in lens:
                acc += l
                cum.append(acc)
            tmp = []
            for (start, end) in positions:
                pf = next((i for i, x in enumerate(cum) if x >= start), -1) + 1
                pt = next((i for i, x in enumerate(cum) if x >= end), -1) + 1
                tmp.append({'pageFrom': pf or 1, 'pageTo': pt or (pf or 1)})
            pages_meta = tmp
        sections = None
        if ext in ['.md', '.txt']:
            lines = raw_text.split('\n')
            heads = []
            offset = 0
            for ln in lines:
                if ln.strip().startswith('#'):
                    heads.append({'pos': offset, 'text': ln.strip().lstrip('#').strip()})
                offset += len(ln) + 1
            tmp = []
            for (start, _) in positions:
                cand = [h for h in heads if h['pos'] <= start]
                h = cand[-1]['text'] if cand else None
                tmp.append(h)
            sections = tmp
        out = []
        for i, (start, end) in enumerate(positions):
            m = {
                'fileType': ext,
                'docTitle': title,
                'chunkIndex': i,
                'pageFrom': pages_meta[i]['pageFrom'] if pages_meta else None,
                'pageTo': pages_meta[i]['pageTo'] if pages_meta else None,
                'section': sections[i] if sections else None
            }
            out.append(m)
        return out

    async def agentic_split(self, text: str) -> List[str]:
        if not self.agentic_llm:
            raise ValueError("Agentic LLM not configured.")
        
        windows = self.recursive_split(text)
        final_chunks = []
        
        for window in windows:
            prompt = f'Split this text into semantically complete propositions. Return a VALID JSON list of strings. Do not include Markdown formatting.\nText: "{window}"'
            try:
                response = await self.agentic_llm.generate(prompt)
                # Cleanup JSON
                clean_json = response.replace("```json", "").replace("```", "").strip()
                # Find the first [ and last ]
                start = clean_json.find('[')
                end = clean_json.rfind(']')
                if start != -1 and end != -1:
                    clean_json = clean_json[start:end+1]
                
                parsed = json.loads(clean_json)
                if isinstance(parsed, list):
                    final_chunks.extend(parsed)
                else:
                    final_chunks.append(window)
            except Exception:
                final_chunks.append(window)
                
        return final_chunks
