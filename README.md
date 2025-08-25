# Python Code Minifier

一个高效的 Python 代码压缩工具，用于去除注释、文档字符串和多余空行，减小文件体积，适用于代码发布和部署场景。

## 主要特性

- ✨ **智能去除注释** - 保留字符串内的 `#` 字符，精确识别真实注释
- 📝 **精确删除文档字符串** - 基于 AST 解析，准确识别模块、类、函数文档字符串
- 🧹 **清理多余空行** - 可选保留代码结构，避免破坏可读性
- 📁 **批量处理目录** - 支持递归处理整个项目目录
- 🌐 **多种编码兼容** - 自动检测 UTF-8、GBK、Latin-1 等编码格式
- ⚙️ **灵活配置选项** - 支持单独或组合去除不同类型内容
- 📊 **详细统计信息** - 显示处理前后文件大小对比和压缩率

## 快速开始

### 基本用法

```bash
# 精简单个文件（去除所有注释、文档字符串、空行）
python minify_code.py input.py

# 指定输出文件
python minify_code.py input.py -o output.py

# 批量处理目录
python minify_code.py -d src_directory
```

### 选择性处理

```bash
# 只去除注释
python minify_code.py input.py --rm-c

# 只去除文档字符串
python minify_code.py input.py --rm-d

# 只去除空行
python minify_code.py input.py --rm-e

# 组合处理：去除注释和文档字符串
python minify_code.py input.py --rm-cd
```

## 详细使用教程

### 1. 单文件处理

#### 基本精简
```bash
# 去除所有内容（默认行为）
python minify_code.py example.py
# 输出：example_min.py
```

#### 自定义输出路径
```bash
python minify_code.py src/main.py -o dist/main.py
```

#### 保留代码结构
```bash
# 保留部分空行以维持代码结构
python minify_code.py input.py --preserve-struct
```

### 2. 选择性处理选项

| 选项 | 说明 | 示例 |
|------|------|------|
| `--rm-c` | 只移除注释 | `python minify_code.py file.py --rm-c` |
| `--rm-d` | 只移除文档字符串 | `python minify_code.py file.py --rm-d` |
| `--rm-e` | 只移除空行 | `python minify_code.py file.py --rm-e` |
| `--rm-cd` | 移除注释和文档字符串 | `python minify_code.py file.py --rm-cd` |
| `--rm-ce` | 移除注释和空行 | `python minify_code.py file.py --rm-ce` |
| `--rm-de` | 移除文档字符串和空行 | `python minify_code.py file.py --rm-de` |
| `--rm-cde` | 移除所有内容（默认） | `python minify_code.py file.py --rm-cde` |

### 3. 批量处理

#### 处理整个目录
```bash
# 处理当前目录下所有 .py 文件
python minify_code.py -d .

# 处理指定目录
python minify_code.py -d src_directory
```

#### 指定输出目录
```bash
# 将精简后的文件输出到不同目录
python minify_code.py -d src -od dist
```

#### 自定义文件匹配模式
```bash
# 只处理特定模式的文件
python minify_code.py -d src -p "*.py" --rm-c
```

#### 包含已存在的精简文件
```bash
# 重新处理已经精简过的文件
python minify_code.py -d src --include-existing
```

### 4. 高级选项

```bash
# 静默模式（不显示统计信息）
python minify_code.py input.py -q

# 详细输出模式
python minify_code.py input.py -v

# 查看帮助信息
python minify_code.py --help
```

## 处理示例

### 原始代码
```python
#!/usr/bin/env python3
"""
这是一个示例模块
用于演示代码精简功能
"""

class Calculator:
    """计算器类"""
    
    def __init__(self):
        """初始化计算器"""
        self.result = 0  # 存储结果
    
    def add(self, x, y):
        """加法运算"""
        return x + y  # 返回两数之和


# 主程序入口
if __name__ == "__main__":
    calc = Calculator()  # 创建计算器实例
    print(calc.add(1, 2))  # 输出计算结果
```

### 精简后代码
```python
#!/usr/bin/env python3
class Calculator:
    def __init__(self):
        self.result = 0
    def add(self, x, y):
        return x + y
if __name__ == "__main__":
    calc = Calculator()
    print(calc.add(1, 2))
```

### 处理统计
```
处理完成: example.py
输出文件: example_min.py
原始大小: 456 字节
精简大小: 187 字节
节省空间: 269 字节 (59.0%)
移除行数: 12
移除注释: 4
移除文档字符串: 3
移除空行: 5
```

## 技术特性

### 智能字符串处理
- 使用 Python `tokenize` 模块精确识别字符串边界
- 正确处理原始字符串（r-strings）、f-strings、三引号字符串
- 保留字符串内的 `#` 字符和其他特殊字符

### AST 文档字符串检测
- 基于抽象语法树（AST）准确识别文档字符串
- 区分普通字符串和文档字符串
- 支持模块、类、函数、异步函数的文档字符串

### 错误处理机制
- 语法错误时自动回退到保守算法
- 支持多种文件编码自动检测
- 完善的异常处理和错误提示

## 兼容性

- **Python 版本**: 3.7+
- **操作系统**: Windows, macOS, Linux
- **编码支持**: UTF-8, GBK, Latin-1, CP1252

## 性能表现

| 项目规模 | 处理时间 | 平均压缩率 |
|----------|----------|------------|
| 小型项目 (< 10 文件) | < 1秒 | 40-60% |
| 中型项目 (10-100 文件) | 1-5秒 | 35-55% |
| 大型项目 (> 100 文件) | 5-30秒 | 30-50% |

## 注意事项

### 代码完整性
- 精简后的代码在功能上与原代码完全相同
- 只移除注释、文档字符串和空行，不影响程序逻辑
- 建议在重要项目中先测试精简后的代码

### 版本控制
- 建议保留原始代码文件
- 精简文件默认添加 `_min` 后缀
- 可以在 `.gitignore` 中忽略精简文件

### 使用场景
- ✅ 生产环境部署
- ✅ 嵌入式设备代码
- ✅ 代码混淆预处理
- ❌ 开发调试阶段
- ❌ 代码审查和维护

## 常见问题

### Q: 会不会破坏代码功能？
A: 不会。工具只移除注释、文档字符串和空行，不修改任何程序逻辑。

### Q: 支持哪些 Python 语法特性？
A: 支持所有标准 Python 语法，包括 f-strings、async/await、类型注解等。

### Q: 如何处理语法错误的文件？
A: 遇到语法错误时会自动回退到保守算法，尽可能处理文件内容。

### Q: 可以恢复精简前的代码吗？
A: 无法直接恢复，建议保留原始文件或使用版本控制系统。

## 开发相关

### 依赖项
```bash
# 核心依赖（Python 标准库）
import ast
import tokenize
import argparse
import pathlib
```

### 运行测试
```bash
# 测试单个文件
python minify_code.py test_example.py

# 测试复杂字符串场景
python minify_code.py complex_string_test.py
```

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/new-feature`
3. 提交更改：`git commit -am 'Add new feature'`
4. 推送分支：`git push origin feature/new-feature`
5. 创建 Pull Request

## 更新日志

### v1.0.0 (2025-08-25)
- 🎉 初始版本发布
- ✨ 基于 AST 的文档字符串检测
- 🔧 智能注释处理算法
- 📁 批量目录处理功能
- 🌐 多编码格式支持

## 许可证

本项目采用 GPL-3.0 License 许可证

## 联系作者

- **作者**: LeiLei
- **邮箱**: 20739111@qq.com
- **GitHub**: [@your-username](https://github.com/qq20739111)

---

如果这个项目对你有帮助，请给个 ⭐ Star 支持一下！