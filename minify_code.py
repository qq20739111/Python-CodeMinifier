#!/usr/bin/env python3
"""
Python代码精简工具 - Code Minifier v1.0.0

一个高效的Python代码压缩工具，用于去除注释、文档字符串和多余空行，
减小文件体积，适用于代码发布和部署场景。

主要功能:
- 智能去除注释 (保留字符串内的#字符)
- 精确删除文档字符串 (基于AST解析)
- 清理多余空行 (可选保留代码结构)
- 支持批量处理目录
- 多种编码格式兼容

基本用法:
    # 精简单个文件 (去除所有内容)
    python minify_code.py input.py
    
    # 只去除注释
    python minify_code.py input.py --rm-c

    # 去除注释、文档字符串和空行
    python minify_code.py input.py --rm-cde
    
    # 批量处理目录
    python minify_code.py -d src_directory

作者: LeiLei
日期: 2025-08-25
版本: v1.0.0
许可: GPL-3.0 License
"""

import sys
import argparse
import tokenize
import io
import ast
from pathlib import Path
from typing import Set, Tuple, Optional, Dict, Any


class CodeMinifier:
    """代码精简器类 - 重构版本"""
    
    # ============================================================================
    # 1. 初始化和配置方法
    # ============================================================================
    
    def __init__(self):
        """初始化代码精简器"""
        self.stats = {
            'original_size': 0,
            'minified_size': 0,
            'lines_removed': 0,
            'comments_removed': 0,
            'docstrings_removed': 0,
            'empty_lines_removed': 0
        }
    
    def _reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'original_size': 0,
            'minified_size': 0,
            'lines_removed': 0,
            'comments_removed': 0,
            'docstrings_removed': 0,
            'empty_lines_removed': 0
        }
    
    # ============================================================================
    # 2. 主要公共接口方法
    # ============================================================================
    
    def process_content(self, content: str, remove_comments: bool = True, 
                       remove_docstrings: bool = True, remove_empty_lines: bool = True, 
                       preserve_structure: bool = False, stats: Optional[Dict[str, Any]] = None) -> str:
        """
        处理代码内容的主方法
        
        Args:
            content: 原始代码内容
            remove_comments: 是否去除注释
            remove_docstrings: 是否去除文档字符串
            remove_empty_lines: 是否去除空行
            preserve_structure: 是否保留代码结构
            stats: 统计信息字典，如果为None则使用实例的stats
            
        Returns:
            处理后的代码内容
        """
        # 使用传入的stats或实例stats，提高线程安全性
        if stats is None:
            stats = self.stats
        
        # 早期返回优化：空内容直接返回
        if not content.strip():
            return content
        
        lines = content.split('\n')
        original_line_count = len(lines)
        
        # 按顺序应用处理方法
        if remove_docstrings:
            lines = self.remove_docstrings(lines, stats)
        
        if remove_comments:
            lines = self.remove_comments(lines, stats)
        
        if remove_empty_lines:
            lines = self.remove_empty_lines(lines, preserve_structure, stats)
        
        stats['lines_removed'] = original_line_count - len(lines)
        return '\n'.join(lines)
    
    def minify_file(self, input_file: str, output_file: Optional[str] = None, 
                   rm_option: str = "cde", preserve_structure: bool = False) -> bool:
        """
        精简单个文件
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径，默认为原文件名_min.py
            rm_option: 移除选项 (c=注释, d=文档字符串, e=空行)
            preserve_structure: 是否保留代码结构
            
        Returns:
            是否成功处理
        """
        try:
            # 输入验证
            if not self._validate_input_file(input_file):
                return False
            
            # 重置统计信息
            self._reset_stats()
            
            # 读取输入文件
            content = self._read_file_with_encoding_detection(input_file)
            if content is None:
                return False
            
            self.stats['original_size'] = len(content)
            
            # 解析处理选项
            options = self.parse_remove_options(rm_option)
            
            # 处理内容
            minified_content = self.process_content(
                content, 
                options['remove_comments'],
                options['remove_docstrings'], 
                options['remove_empty_lines'], 
                preserve_structure
            )
            
            self.stats['minified_size'] = len(minified_content)
            
            # 确定输出文件路径
            if output_file is None:
                input_path = Path(input_file)
                output_file = input_path.parent / f"{input_path.stem}_min{input_path.suffix}"
            
            # 验证输出路径
            if not self._validate_output_path(output_file):
                return False
            
            # 写入输出文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(minified_content)
            
            return True
            
        except Exception as e:
            print(f"处理文件时出错: {e}")
            return False  # 修复bug：明确返回False
    
    def minify_directory(self, input_dir: str, output_dir: Optional[str] = None, 
                        pattern: str = "*.py", rm_option: str = "cde", 
                        preserve_structure: bool = False, skip_existing: bool = True) -> list:
        """
        批量处理目录中的文件
        
        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径
            pattern: 文件匹配模式
            rm_option: 移除选项 (c=注释, d=文档字符串, e=空行)
            preserve_structure: 是否保留代码结构
            skip_existing: 是否跳过已存在的精简文件
            
        Returns:
            处理结果列表
        """
        input_path = Path(input_dir)
        if not input_path.exists() or not input_path.is_dir():
            print(f"错误: 输入目录不存在或不是目录: {input_dir}")
            return []
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = input_path
        
        results = []
        
        for file_path in input_path.rglob(pattern):
            if file_path.is_file():
                # 如果启用跳过现有文件选项，检查是否为已精简的文件
                if skip_existing and file_path.stem.endswith('_min'):
                    continue
                
                relative_path = file_path.relative_to(input_path)
                if output_dir:
                    output_file = output_path / relative_path
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                else:
                    # 保持目录结构，在相应子目录中生成 _min 文件
                    relative_dir = relative_path.parent
                    output_subdir = output_path / relative_dir
                    output_subdir.mkdir(parents=True, exist_ok=True)
                    output_file = output_subdir / f"{file_path.stem}_min{file_path.suffix}"
                
                # 重置当前实例的统计信息并处理文件（避免创建多个实例）
                self._reset_stats()
                success = self.minify_file(
                    str(file_path), str(output_file), rm_option, preserve_structure
                )
                
                results.append({
                    'input': str(file_path),
                    'output': str(output_file),
                    'success': success,
                    'stats': self.stats.copy()  # 复制当前统计信息
                })
        
        return results
    
    # ============================================================================
    # 3. 核心处理方法
    # ============================================================================
    
    def remove_comments(self, lines: list, stats: Optional[Dict[str, Any]] = None) -> list:
        """
        去除注释的独立方法 - 重新设计版本（使用更可靠的算法）
        
        Args:
            lines: 代码行列表
            stats: 统计信息字典，如果为None则使用实例的stats
            
        Returns:
            处理后的代码行列表
        """
        if stats is None:
            stats = self.stats
        
        # 早期返回优化：空列表直接返回
        if not lines:
            return lines
        
        result_lines = []
        
        # 首先尝试对整个代码块进行tokenize，获取准确的字符串和注释位置
        full_text = '\n'.join(lines)
        string_ranges, comment_ranges = self._analyze_code_structure(full_text)
        
        # 处理每一行
        current_pos = 0
        for line_num, line in enumerate(lines):
            line_start_pos = current_pos
            line_end_pos = current_pos + len(line)
            
            # 检查是否为shebang行或编码声明行（保留这些特殊注释）
            stripped_line = line.strip()
            is_special_comment = self._is_special_comment(stripped_line)
            
            # 首先检查这一行是否完全在多行字符串内
            if self._is_entirely_in_multiline_string(line_start_pos, line_end_pos, string_ranges):
                # 在多行字符串内，保留原样
                result_lines.append(line)
            elif stripped_line.startswith('#') and not is_special_comment:
                # 不在多行字符串内的普通注释行，删除（但保留特殊注释）
                stats['comments_removed'] += 1
            else:
                # 特殊注释行或普通代码行
                if is_special_comment:
                    # 保留特殊注释行
                    result_lines.append(line)
                else:
                    # 不在多行字符串内的普通行，去除行内注释
                    processed_line = self._remove_line_comments(line, line_start_pos, comment_ranges)
                    if processed_line != line:
                        stats['comments_removed'] += 1
                    result_lines.append(processed_line)
            
            current_pos = line_end_pos + 1  # +1 for newline
        
        return result_lines
    
    def remove_docstrings(self, lines: list, stats: Optional[Dict[str, Any]] = None) -> list:
        """
        去除文档字符串的新方法 - 基于AST的精准算法
        
        Args:
            lines: 代码行列表
            stats: 统计信息字典，如果为None则使用实例的stats
            
        Returns:
            处理后的代码行列表
        """
        if stats is None:
            stats = self.stats
        
        # 早期返回优化：空列表直接返回
        if not lines:
            return lines
        
        # 重新组合成完整代码用于AST分析
        full_code = '\n'.join(lines)
        
        # 使用AST找到所有文档字符串位置
        docstring_ranges = self._find_docstrings_ast(full_code)
        
        # 创建要删除的行的集合（转换为0-based索引）
        lines_to_remove = set()
        docstring_count = 0  # 正确计数文档字符串个数
        for start_line, end_line in docstring_ranges:
            docstring_count += 1  # 每个文档字符串只计数一次，而不是每行计数
            for line_num in range(start_line, end_line + 1):
                lines_to_remove.add(line_num)
        
        # 更新统计信息
        stats['docstrings_removed'] = docstring_count
        
        # 构建结果
        result_lines = []
        for i, line in enumerate(lines):
            if i not in lines_to_remove:
                result_lines.append(line)
        
        return result_lines
    
    def remove_empty_lines(self, lines: list, preserve_structure: bool = False, 
                          stats: Optional[Dict[str, Any]] = None) -> list:
        """
        去除空行的独立方法
        
        Args:
            lines: 代码行列表
            preserve_structure: 是否保留代码结构
            stats: 统计信息字典，如果为None则使用实例的stats
            
        Returns:
            处理后的代码行列表
        """
        if stats is None:
            stats = self.stats
        
        result_lines = []
        
        for line in lines:
            if not line.strip():
                if preserve_structure:
                    # 保留结构时，连续空行只保留一行
                    if result_lines and result_lines[-1].strip() == '':
                        stats['empty_lines_removed'] += 1
                        continue
                    else:
                        result_lines.append('')
                        continue
                else:
                    stats['empty_lines_removed'] += 1
                    continue
            
            result_lines.append(line)
        
        return result_lines
    
    # ============================================================================
    # 4. 注释处理相关方法
    # ============================================================================
    
    def _analyze_code_structure(self, code: str) -> Tuple[list, list]:
        """
        分析代码结构，获取所有字符串和注释的准确位置
        
        Args:
            code: 完整的代码文本
            
        Returns:
            (string_ranges, comment_ranges)
                string_ranges: [(start, end, is_multiline), ...]
                comment_ranges: [(start, end), ...]
        """
        string_ranges = []
        comment_ranges = []
        
        try:
            # 使用tokenize获取所有token
            tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
            
            for token in tokens:
                if token.type == tokenize.STRING:
                    # 计算字符在整个文本中的位置
                    start_pos = self._line_col_to_pos(code, token.start)
                    end_pos = self._line_col_to_pos(code, token.end)
                    
                    # 判断是否是多行字符串 (修复：处理raw字符串前缀)
                    token_str = token.string
                    # 移除可能的前缀 (r, u, b, f 及其组合)
                    clean_token = token_str
                    if clean_token.lower().startswith(('r"""', "r'''", 'u"""', "u'''", 'b"""', "b'''", 'f"""', "f'''",
                                                      'rb"""', "rb'''", 'rf"""', "rf'''", 'br"""', "br'''", 'bf"""', "bf'''")):
                        # 找到引号的实际开始位置
                        quote_start = len(clean_token) - len(clean_token.lstrip('rRuUbBfF'))
                        clean_token = clean_token[quote_start:]
                    
                    is_multiline = (clean_token.startswith('"""') or clean_token.startswith("'''")) and '\n' in token_str
                    
                    string_ranges.append((start_pos, end_pos, is_multiline))
                
                elif token.type == tokenize.COMMENT:
                    # 计算注释在整个文本中的位置
                    start_pos = self._line_col_to_pos(code, token.start)
                    end_pos = self._line_col_to_pos(code, token.end)
                    comment_ranges.append((start_pos, end_pos))
        
        except (tokenize.TokenError, SyntaxError, IndentationError, UnicodeDecodeError):
            # 更全面的异常捕获，如果tokenize完全失败，使用保守的备用方法
            return self._fallback_analyze_structure(code)
        
        return string_ranges, comment_ranges
    
    def _line_col_to_pos(self, text: str, line_col: Tuple[int, int]) -> int:
        """
        将行列坐标转换为文本中的绝对位置（改进版本）
        
        Args:
            text: 文本内容
            line_col: (line, col) 坐标 (1-based行, 0-based列)
            
        Returns:
            绝对位置
        """
        line, col = line_col
        lines = text.split('\n')
        
        # 更严格的边界检查
        if line < 1 or line > len(lines):
            return len(text)
        
        # 考虑文件末尾是否有换行符
        pos = 0
        for i in range(line - 1):
            pos += len(lines[i])
            if i < len(lines) - 1 or text.endswith('\n'):
                pos += 1  # 换行符
        
        # 确保col不会超出当前行的长度
        current_line_length = len(lines[line - 1]) if line <= len(lines) else 0
        safe_col = min(col, current_line_length)
        
        return min(pos + safe_col, len(text))
    
    def _is_entirely_in_multiline_string(self, line_start: int, line_end: int, 
                                        string_ranges: list) -> bool:
        """
        检查指定范围是否完全在多行字符串内
        
        Args:
            line_start: 行开始位置
            line_end: 行结束位置
            string_ranges: 字符串范围列表
            
        Returns:
            是否完全在多行字符串内
        """
        for start, end, is_multiline in string_ranges:
            if is_multiline and start <= line_start and line_end <= end:
                return True
        return False
    
    def _remove_line_comments(self, line: str, line_start_pos: int, comment_ranges: list) -> str:
        """
        去除行内注释，但保留字符串中的#字符
        
        Args:
            line: 当前行内容
            line_start_pos: 行在整个文本中的起始位置
            comment_ranges: 注释范围列表
            
        Returns:
            处理后的行内容
        """
        if '#' not in line:
            return line
        
        # 找到这一行中的第一个注释（最靠前的）
        earliest_comment_pos = len(line)  # 初始化为行末尾
        
        for comment_start, comment_end in comment_ranges:
            # 检查注释是否在当前行内
            relative_start = comment_start - line_start_pos
            if 0 <= relative_start < len(line):
                # 找到行内注释，记录最早的位置
                earliest_comment_pos = min(earliest_comment_pos, relative_start)
        
        # 如果找到注释，在该位置截断
        if earliest_comment_pos < len(line):
            return line[:earliest_comment_pos].rstrip()
        
        return line
    
    def _is_special_comment(self, line: str) -> bool:
        """
        检查是否为特殊注释行（需要保留的注释）
        
        Args:
            line: 已经strip()的行内容
            
        Returns:
            是否为特殊注释
        """
        if not line.startswith('#'):
            return False
        
        # shebang行
        if line.startswith('#!'):
            return True
        
        # 编码声明行
        if line.startswith('#') and ('coding' in line.lower() or 'encoding' in line.lower()):
            # 匹配常见的编码声明格式
            import re
            encoding_patterns = [
                r'#.*?coding[:=]\s*([-\w.]+)',
                r'#.*?encoding[:=]\s*([-\w.]+)',
                r'#.*?-\*-.*?coding[:=]\s*([-\w.]+).*?-\*-',
                r'#.*?vim:.*?fileencoding=\s*([-\w.]+)'
            ]
            for pattern in encoding_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    return True
        
        return False

    def _fallback_analyze_structure(self, code: str) -> Tuple[list, list]:
        """
        当tokenize失败时的备用结构分析方法
        
        Args:
            code: 代码文本
            
        Returns:
            (string_ranges, comment_ranges)
        """
        # 非常保守的实现：只处理简单的行注释
        string_ranges = []
        comment_ranges = []
        
        lines = code.split('\n')
        pos = 0
        
        for line in lines:
            # 查找简单的行注释（不在字符串内的#）
            in_string = False
            string_char = None
            i = 0
            
            while i < len(line):
                char = line[i]
                
                if not in_string:
                    if char in ['"', "'"]:
                        # 检查是否是三引号
                        if i + 2 < len(line) and line[i:i+3] in ['"""', "'''"]:
                            # 跳过三引号处理，太复杂
                            i += 3
                            continue
                        else:
                            in_string = True
                            string_char = char
                    elif char == '#':
                        # 找到注释
                        comment_start = pos + i
                        comment_end = pos + len(line)
                        comment_ranges.append((comment_start, comment_end))
                        break
                else:
                    if char == string_char:
                        # 检查是否被转义
                        escape_count = 0
                        check_pos = i - 1
                        while check_pos >= 0 and line[check_pos] == '\\':
                            escape_count += 1
                            check_pos -= 1
                        
                        if escape_count % 2 == 0:  # 没有被转义
                            in_string = False
                            string_char = None
                
                i += 1
            
            pos += len(line) + 1  # +1 for newline
        
        return string_ranges, comment_ranges
    
    # ============================================================================
    # 5. 文档字符串处理相关方法
    # ============================================================================
    
    def _find_docstrings_ast(self, code: str) -> Set[Tuple[int, int]]:
        """
        使用AST准确找到所有文档字符串的位置
        
        Args:
            code: 源代码字符串
            
        Returns:
            文档字符串的 (start_line, end_line) 集合（0-based）
        """
        docstring_ranges = set()
        
        try:
            tree = ast.parse(code)
        except (SyntaxError, IndentationError, UnicodeDecodeError):
            # 更全面的异常处理，AST解析失败，回退到原有方法
            return self._find_docstrings_fallback(code)
        
        def extract_docstrings(node, parent=None):
            """递归提取文档字符串"""
            
            # 如果当前节点是Module，遍历其子节点
            if isinstance(node, ast.Module):
                for child in ast.iter_child_nodes(node):
                    extract_docstrings(child, node)
                return
            
            # 检查是否是文档字符串 - 兼容不同Python版本
            def is_string_expr(node):
                """检查是否是字符串表达式，兼容Python 3.7及以下版本"""
                if not isinstance(node, ast.Expr):
                    return False
                # Python 3.8+
                if hasattr(ast, 'Constant') and isinstance(node.value, ast.Constant):
                    return isinstance(node.value.value, str)
                # Python 3.7及以下
                elif hasattr(ast, 'Str') and isinstance(node.value, ast.Str):
                    return True
                return False
            
            if is_string_expr(node):
                # 检查是否在函数/类/模块的开头
                is_docstring = False
                
                if isinstance(parent, ast.Module):  # 模块级文档字符串
                    # 检查是否是模块的第一个语句（排除shebang和编码声明）
                    if parent.body and parent.body[0] == node:
                        # 额外检查：确保这确实是文档字符串而不是普通字符串
                        # 检查前面是否只有shebang和编码声明等特殊行
                        lines = code.split('\n')
                        string_start_line = node.lineno - 1  # 转为0-based
                        
                        # 检查文档字符串前面的行是否都是特殊注释或空行
                        is_valid_module_docstring = True
                        for i in range(string_start_line):
                            line = lines[i].strip()
                            if line and not self._is_special_comment(line):
                                is_valid_module_docstring = False
                                break
                        
                        if is_valid_module_docstring:
                            is_docstring = True
                elif isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # 检查是否是函数/类的第一个语句
                    if parent.body and parent.body[0] == node:
                        is_docstring = True
                
                if is_docstring:
                    # AST提供的行号是1-based，转换为0-based
                    start_line = node.lineno - 1
                    # 安全获取end_lineno，兼容不同Python版本
                    if hasattr(node, 'end_lineno') and node.end_lineno is not None:
                        end_line = node.end_lineno - 1
                    elif hasattr(node.value, 'end_lineno') and node.value.end_lineno is not None:
                        end_line = node.value.end_lineno - 1
                    else:
                        # 备用方法：根据字符串内容估算（改进版本）
                        try:
                            if hasattr(node.value, 'value'):
                                string_content = node.value.value
                            else:  # Python 3.7及以下版本的ast.Str
                                string_content = node.value.s
                            # 改进：更准确地计算结束行
                            end_line = start_line + string_content.count('\n')
                        except (AttributeError, TypeError):
                            end_line = start_line
                    docstring_ranges.add((start_line, end_line))
            
            # 递归处理子节点
            for child in ast.iter_child_nodes(node):
                extract_docstrings(child, node)
        
        extract_docstrings(tree)
        return docstring_ranges
    
    def _find_docstrings_fallback(self, code: str) -> Set[Tuple[int, int]]:
        """
        AST失败时的备用文档字符串查找方法
        """
        lines = code.split('\n')
        docstring_ranges = set()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # 查找三引号开始的行
            if line.startswith('"""') or line.startswith("'''"):
                delimiter = '"""' if line.startswith('"""') else "'''"
                
                # 简单判断是否为文档字符串
                if self._is_likely_docstring_simple(lines, i):
                    # 查找结束位置
                    end_line = self._find_docstring_end_simple(lines, i, delimiter)
                    if end_line >= i:
                        docstring_ranges.add((i, end_line))
                        i = end_line + 1
                        continue
            
            i += 1
        
        return docstring_ranges
    
    def _is_likely_docstring_simple(self, lines: list, line_index: int) -> bool:
        """简单的文档字符串判断（保守策略）"""
        # 模块级（文件开头）
        if line_index <= 3:
            return True
        
        # 检查前面几行是否有函数/类定义
        for i in range(max(0, line_index - 3), line_index):
            line = lines[i].strip()
            if line.startswith('def ') or line.startswith('class ') or line.startswith('async def '):
                return True
            if ':' in line and ('def ' in line or 'class ' in line):
                return True
        
        return False
    
    def _find_docstring_end_simple(self, lines: list, start_line: int, delimiter: str) -> int:
        """查找文档字符串的结束行"""
        start_line_content = lines[start_line].strip()
        
        # 单行文档字符串
        if start_line_content.count(delimiter) >= 2:
            return start_line
        
        # 多行文档字符串
        for i in range(start_line + 1, len(lines)):
            if delimiter in lines[i]:
                return i
        
        return len(lines) - 1  # 如果没找到结束，返回文件末尾
    
    # ============================================================================
    # 6. 工具和验证方法
    # ============================================================================
    
    def parse_remove_options(self, rm_option: str) -> Dict[str, bool]:
        """
        解析用户的移除选项
        
        Args:
            rm_option: 移除选项字符串，如 "cd", "de", "cde" 等
            
        Returns:
            包含各项处理选项的字典
        """
        if not rm_option:
            # 默认处理所有内容
            return {
                'remove_comments': True,
                'remove_docstrings': True,
                'remove_empty_lines': True
            }
        
        rm_option = rm_option.lower()
        return {
            'remove_comments': 'c' in rm_option,
            'remove_docstrings': 'd' in rm_option,
            'remove_empty_lines': 'e' in rm_option
        }
    
    def _validate_input_file(self, input_file: str) -> bool:
        """验证输入文件"""
        try:
            input_path = Path(input_file)
            if not input_path.exists():
                print(f"错误: 输入文件不存在: {input_file}")
                return False
            if not input_path.is_file():
                print(f"错误: 输入路径不是文件: {input_file}")
                return False
            if input_path.suffix.lower() not in ['.py', '.pyw']:
                print(f"警告: 文件可能不是Python文件: {input_file}")
            return True
        except Exception as e:
            print(f"错误: 验证输入文件时出错: {e}")
            return False
    
    def _validate_output_path(self, output_file: str) -> bool:
        """验证输出路径"""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"错误: 创建输出目录时出错: {e}")
            return False
    
    def _read_file_with_encoding_detection(self, file_path: str) -> Optional[str]:
        """
        读取文件，支持编码检测
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件内容，如果失败返回None
        """
        encodings_to_try = ['utf-8', 'gbk', 'latin-1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"错误: 读取文件时出错: {e}")
                return None
        
        print(f"错误: 无法用任何已知编码读取文件: {file_path}")
        return None
    
    # ============================================================================
    # 7. 统计和输出方法
    # ============================================================================
    
    def print_stats(self, input_file: str, output_file: str, rm_option: str = "cde"):
        """打印处理统计信息"""
        original_size = self.stats['original_size']
        minified_size = self.stats['minified_size']
        saved_size = original_size - minified_size
        saved_percent = (saved_size / original_size * 100) if original_size > 0 else 0
        
        options = self.parse_remove_options(rm_option)
        
        print(f"\n处理完成: {input_file}")
        print(f"输出文件: {output_file}")
        print(f"原始大小: {original_size:,} 字节")
        print(f"精简大小: {minified_size:,} 字节")
        print(f"节省空间: {saved_size:,} 字节 ({saved_percent:.1f}%)")
        print(f"移除行数: {self.stats['lines_removed']}")
        
        # 详细显示处理的内容
        if options['remove_comments'] and self.stats['comments_removed'] > 0:
            print(f"移除注释: {self.stats['comments_removed']}")
        if options['remove_docstrings'] and self.stats['docstrings_removed'] > 0:
            print(f"移除文档字符串: {self.stats['docstrings_removed']}")
        if options['remove_empty_lines'] and self.stats['empty_lines_removed'] > 0:
            print(f"移除空行: {self.stats['empty_lines_removed']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Python代码精简工具 - 去除注释、文档字符串和空行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s input.py                         # 去除所有内容（注释、文档字符串、空行）
  %(prog)s input.py -o output.py            # 精简到指定输出文件
  
  单项操作:
  %(prog)s input.py --rm-c                  # 只去除注释
  %(prog)s input.py --rm-d                  # 只去除文档字符串
  %(prog)s input.py --rm-e                  # 只去除空行
  
  双项组合:
  %(prog)s input.py --rm-cd                 # 去除注释和文档字符串
  %(prog)s input.py --rm-dc                 # 去除文档字符串和注释
  %(prog)s input.py --rm-ce                 # 去除注释和空行
  %(prog)s input.py --rm-ec                 # 去除空行和注释
  %(prog)s input.py --rm-de                 # 去除文档字符串和空行
  %(prog)s input.py --rm-ed                 # 去除空行和文档字符串
  
  三项组合（任意顺序）:
  %(prog)s input.py --rm-cde                # 去除注释、文档字符串、空行（默认）
  %(prog)s input.py --rm-ced                # 去除注释、空行、文档字符串
  %(prog)s input.py --rm-dce                # 去除文档字符串、注释、空行
  %(prog)s input.py --rm-dec                # 去除文档字符串、空行、注释
  %(prog)s input.py --rm-ecd                # 去除空行、注释、文档字符串
  %(prog)s input.py --rm-edc                # 去除空行、文档字符串、注释
  
  其他选项:
  %(prog)s input.py --preserve-struct       # 去除所有内容但保留代码结构
  %(prog)s -d src_dir                       # 批量处理目录
  %(prog)s -d src_dir --rm-c                # 批量只去除注释
  %(prog)s -d src_dir --rm-edc              # 批量去除空行、文档字符串、注释
        """
    )
    
    # 位置参数
    parser.add_argument('input', nargs='?', help='输入文件路径')
    
    # 可选参数
    parser.add_argument('-o', '--output', help='输出文件路径')
    parser.add_argument('-d', '--directory', help='处理整个目录')
    parser.add_argument('-od', '--output-dir', help='输出目录路径')
    parser.add_argument('-p', '--pattern', default='*.py', help='文件匹配模式 (默认: *.py)')
    
    # 移除选项 - 统一的 --rm-xxx 格式
    # 单项选项
    parser.add_argument('--rm-c', action='store_const', const='c', dest='rm_option',
                       help='只移除注释 (comments)')
    parser.add_argument('--rm-d', action='store_const', const='d', dest='rm_option',
                       help='只移除文档字符串 (docstrings)')
    parser.add_argument('--rm-e', action='store_const', const='e', dest='rm_option',
                       help='只移除空行 (empty lines)')
    
    # 双项组合选项
    parser.add_argument('--rm-cd', action='store_const', const='cd', dest='rm_option',
                       help='移除注释和文档字符串')
    parser.add_argument('--rm-dc', action='store_const', const='dc', dest='rm_option',
                       help='移除文档字符串和注释')
    parser.add_argument('--rm-ce', action='store_const', const='ce', dest='rm_option',
                       help='移除注释和空行')
    parser.add_argument('--rm-ec', action='store_const', const='ec', dest='rm_option',
                       help='移除空行和注释')
    parser.add_argument('--rm-de', action='store_const', const='de', dest='rm_option',
                       help='移除文档字符串和空行')
    parser.add_argument('--rm-ed', action='store_const', const='ed', dest='rm_option',
                       help='移除空行和文档字符串')
    
    # 三项组合选项（全排列）
    parser.add_argument('--rm-cde', action='store_const', const='cde', dest='rm_option',
                       help='移除注释、文档字符串、空行')
    parser.add_argument('--rm-ced', action='store_const', const='ced', dest='rm_option',
                       help='移除注释、空行、文档字符串')
    parser.add_argument('--rm-dce', action='store_const', const='dce', dest='rm_option',
                       help='移除文档字符串、注释、空行')
    parser.add_argument('--rm-dec', action='store_const', const='dec', dest='rm_option',
                       help='移除文档字符串、空行、注释')
    parser.add_argument('--rm-ecd', action='store_const', const='ecd', dest='rm_option',
                       help='移除空行、注释、文档字符串')
    parser.add_argument('--rm-edc', action='store_const', const='edc', dest='rm_option',
                       help='移除空行、文档字符串、注释')
    
    # 其他选项
    parser.add_argument('--preserve-struct', action='store_true', help='保留代码结构（部分空行）')
    parser.add_argument('--include-existing', action='store_true', help='包含已存在的_min文件（重新处理）')
    parser.add_argument('-q', '--quiet', action='store_true', help='静默模式')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 参数验证
    if not args.input and not args.directory:
        parser.print_help()
        sys.exit(1)
    
    # 创建代码压缩器（提前创建以便验证）
    minifier = CodeMinifier()
    
    # 处理移除选项
    if args.rm_option:
        rm_option = args.rm_option
    else:
        rm_option = 'cde'  # 默认移除所有
    
    # 处理参数
    preserve_struct = args.preserve_struct
    quiet = args.quiet
    verbose = args.verbose
    
    if args.directory:
        # 处理目录
        source_dir = args.directory
        output_dir = args.output_dir
        
        results = minifier.minify_directory(
            input_dir=source_dir,
            output_dir=output_dir,
            pattern=args.pattern,
            rm_option=rm_option,
            preserve_structure=preserve_struct,
            skip_existing=not args.include_existing
        )
        
        if not quiet:
            success_count = sum(1 for r in results if r['success'])
            error_count = len(results) - success_count
            total = len(results)
            success_rate = (success_count / total * 100) if total > 0 else 0
            print(f"\n批量处理完成:")
            print(f"  成功处理: {success_count} 个文件")
            print(f"  处理失败: {error_count} 个文件")
            print(f"  成功率: {success_rate:.1f}%")
    else:
        # 处理单个文件
        source_file = Path(args.input)
        if args.output:
            output_file = Path(args.output)
        else:
            output_file = source_file.parent / f"{source_file.stem}_min{source_file.suffix}"
        
        try:
            success = minifier.minify_file(
                input_file=str(source_file),
                output_file=str(output_file),
                rm_option=rm_option,
                preserve_structure=preserve_struct
            )
            if success:
                if not quiet:
                    minifier.print_stats(str(source_file), str(output_file), rm_option)
            else:
                print(f"错误: 文件处理失败 {source_file}")
                sys.exit(1)
        except Exception as e:
            print(f"错误: 无法处理文件 {source_file}: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
