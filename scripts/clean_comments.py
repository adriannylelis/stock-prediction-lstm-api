#!/usr/bin/env python3
"""
Script para remover comentários excessivos dos arquivos Python.
Mantém apenas comentários essenciais (TODOs, FIXMEs, explicações não-óbvias).
"""

import re
import sys
from pathlib import Path


def should_keep_comment(line: str, prev_line: str = "") -> bool:
    """Decide se um comentário deve ser mantido."""
    comment = line.strip()
    
    if not comment.startswith("#"):
        return True
    
    comment_text = comment[1:].strip().lower()
    
    keep_patterns = [
        "todo",
        "fixme",
        "hack",
        "note:",
        "warning:",
        "important:",
        "bug",
        "issue",
        "deprecated",
        "pylint:",
        "type:",
        "noqa",
        "pragma",
    ]
    
    if any(pattern in comment_text for pattern in keep_patterns):
        return True
    
    obvious_patterns = [
        r"^# (load|import|create|initialize|set|get|return|call|check|verify|validate)",
        r"^# \w+ (para|for|to|of|from|with|in|on|at)",
        r"^#.*\u00e9 \w+$",
        r"^#.*example",
        r"^#.*default",
        r"^#.*fallback",
    ]
    
    if any(re.match(pattern, comment, re.IGNORECASE) for pattern in obvious_patterns):
        return False
    
    if len(comment_text) < 5:
        return False
    
    return True


def clean_file(filepath: Path) -> int:
    """Remove comentários excessivos de um arquivo."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        cleaned_lines = []
        removed_count = 0
        prev_line = ""
        
        for line in lines:
            if should_keep_comment(line, prev_line):
                cleaned_lines.append(line)
            else:
                if line.strip():
                    removed_count += 1
            prev_line = line
        
        if removed_count > 0:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(cleaned_lines)
            print(f"✓ {filepath.relative_to(Path.cwd())}: {removed_count} comentários removidos")
        
        return removed_count
    
    except Exception as e:
        print(f"✗ {filepath}: {e}", file=sys.stderr)
        return 0


def main():
    """Processa todos os arquivos Python no projeto."""
    project_root = Path(__file__).parent.parent
    
    patterns = [
        "src/**/*.py",
        "cli/**/*.py",
        "tests/**/*.py",
    ]
    
    total_removed = 0
    total_files = 0
    
    for pattern in patterns:
        for filepath in project_root.glob(pattern):
            if "__pycache__" in str(filepath):
                continue
            
            removed = clean_file(filepath)
            if removed > 0:
                total_files += 1
                total_removed += removed
    
    print(f"\n📊 Resumo:")
    print(f"   Arquivos modificados: {total_files}")
    print(f"   Comentários removidos: {total_removed}")


if __name__ == "__main__":
    main()
