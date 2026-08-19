from dataclasses import dataclass
from pathlib import Path

from tree_sitter_languages import get_parser

from terminus.observability.logging import get_logger

logger = get_logger(__name__)


EXTENSION_TO_LANGUAGE = {
    # Python
    '.py': 'python',
    '.pyi': 'python',
    
    # JavaScript / TypeScript
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'tsx',
    
    # Java / Kotlin / Scala
    '.java': 'java',
    '.kt': 'kotlin',
    '.scala': 'scala',
    
    # C / C++ / Objective-C
    '.c': 'c',
    '.cpp': 'cpp',
    '.h': 'c',
    '.hpp': 'cpp',
    '.m': 'objc',
    '.mm': 'objc',
    
    # C# / F# / VB.NET
    '.cs': 'c_sharp',
    '.fs': 'f_sharp',
    '.vb': 'vb_net',
    
    # Go
    '.go': 'go',
    
    # Rust
    '.rs': 'rust',
    
    # Web
    '.html': 'html',
    '.css': 'css',
    '.scss': 'scss',
    '.sass': 'scss',
    '.less': 'less',
    
    # Markup
    '.md': 'markdown',
    '.markdown': 'markdown',
    '.xml': 'xml',
    '.json': 'json',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    
    # Shell / Batch
    '.sh': 'bash',
    '.bash': 'bash',
    '.zsh': 'bash',
    '.fish': 'bash',
    '.ps1': 'powershell',
    '.bat': 'batch',
    '.cmd': 'batch',
    
    # PHP
    '.php': 'php',
    
    # Ruby
    '.rb': 'ruby',
    
    # Config / Other
    '.ini': 'ini',
    '.conf': 'ini',
    '.toml': 'toml',
}

TEXT_EXTENSIONS = {".txt", ".log", ".md", ".json", ".xml", ".ini", ".yaml", ".yml", ".csv", ".cfg", ".conf", ".jsonl", ".text"}

ALL_EXTENSION = set(EXTENSION_TO_LANGUAGE.keys()) | TEXT_EXTENSIONS

CHUNK_SIZE=50
CHUNK_OVERLAP=10


BLOCK_NODE_TYPES={
    "function_definition": "function",
    "class_definition": "class",
    "interface_declaration": "interface",
    "struct_declaration": "struct",
    "enum_declaration": "enum",
    "union_declaration": "union",
    "type_alias_declaration": "type_alias",
    "trait_declaration": "trait",
    "impl_block": "implementation",
    "method_declaration": "method",
    "constructor_declaration": "constructor",
    "destructor_declaration": "destructor",
    
}

@dataclass
class ParsedChunk:
    name:str
    type:str
    content:str
    source:str
    start_line:int
    end_line:int

def parse_file(filepath:str)->list[ParsedChunk]:
    """Entry point - routes to AST or text parser based on file extension"""
    ext=Path(filepath).suffix.lower()
    if not Path(filepath).exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError

    if ext in TEXT_EXTENSIONS:
        lines = Path(filepath).read_text(encoding="utf-8",errors="ignore").splitlines()
        return _sliding_window(lines, filepath)
    language_name = EXTENSION_TO_LANGUAGE.get(ext)
    if not language_name:
        logger.warning(f"Unsupported file type: {filepath}")
        raise ValueError(f"Unsupported file type: {filepath}")
    source = Path(filepath).read_text(encoding="utf-8",errors="ignore")
    return _parse_with_treesitter(source, filepath,language_name)

def _parse_with_treesitter(source:str,filepath:str,language_name:str)->list[ParsedChunk]:
    """ Parse source with the appropriate tree-sitter grammar and extract named block."""
    logger.info(f"Parsing {filepath} with {language_name} grammar")
    parser = get_parser(language_name)
    tree = parser.parse(source.encode())
    lines = source.splitlines()
    chunks = []
    _walk(tree.root_node,source,filepath,chunks,depth=0)

    if not chunks:
        logger.warning(f"No structured code blocks found in {filepath}")
        return _sliding_window(lines,filepath)

    logger.info(f"Parsed {filepath} into {len(chunks)} blocks using AST")
    return chunks


def _walk(node,source:str,filepath:str,chunks:list[ParsedChunk],depth:int):
    """ Recursively walk the AST and extract named blocks."""
    if node.type in BLOCK_NODE_TYPES:
        name = _extract_name(node, source)
        content = source[node.start_byte: node.end_byte]
        chunk_type = "class" if "class" in node.type else "function"
        chunks.append(ParsedChunk(
            name=name,
            type=chunk_type,
            content=content,
            source=filepath,
            start_line=node.start_point[0]+1,
            end_line=node.end_point[0]+1
        ))
        logger.debug(f"Extracted {chunk_type}: {name}")

    for child in node.children:
        _walk(child,source,filepath,chunks,depth+1)
        

def _extract_name(node, source:str)->str:
     """Find the identifier child of a block node"""
     for child in node.children:
         if child.type in ["identifier","name", "proper_identifier"]:
             return source[child.start_byte:child.end_byte]
     return node.type


def _sliding_window(lines: list[str], filepath:str)-> list[ParsedChunk]:
    """Fallback to sliding window chunking if AST parsing fails or is not applicable """
    if not lines:
        raise ValueError(f"Empty files: {filepath}")
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i,start_line in enumerate(range(0, len(lines), step)):
        end_line = min(start_line + CHUNK_SIZE, len(lines))
        chunk_text = "\n".join(lines[start_line:end_line]).strip()
        if chunk_text:
            chunks.append(
                ParsedChunk(
                    name=f"window_{i}",
                    type="chunk",
                    content=chunk_text,
                    source=filepath,
                    start_line=start_line+1,
                    end_line=end_line
                )
            )
            if end_line == len(lines):
                break
            logger.debug(f"Created window chunk {i}")
    logger.info(f"Split {filepath} into {len(chunks)} sliding window chunks")
    return chunks

def get_source_files(repo_path:str, skips_dirs: list[str] | None = None)->list[str]:
    """Recursively  """
    skip = set(skips_dirs or [".venv","venv", ".git", "__pycache__","node_modules","dist", "build", "target"])
    files = [
        str(path) for path in Path(repo_path).rglob('*')
        if path.suffix.lower() in ALL_EXTENSION
        and not any(part in skip for part in path.parts)
        
    ]
    logger.info(f"Found {len(files)} source files in {repo_path}")
    return files
        
    
        
    

    
    