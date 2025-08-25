#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复杂字符串和注释测试文件
测试各种引号组合和#字符在Python中的所有可能场景
这个文件用于验证minify_code.py的字符串处理能力

这是模块的第一个字符串表达式语句，将被认定为模块文档字符串
"""

"""
这不是第一个多行字符串表达式语句
不是模块文档字符串，不会被删除！
"""

# ========================================
# 第一部分：基础字符串测试
# ========================================

# 单引号字符串
simple_single = 'This is a simple single quote string'  # 行内注释1
simple_single_with_hash = 'String with # inside'  # 这个注释应该被删除

# 双引号字符串
simple_double = "This is a simple double quote string"  # 行内注释2
simple_double_with_hash = "String with # inside"  # 这个注释应该被删除

# 空字符串
empty_single = ''  # 空字符串单引号
empty_double = ""  # 空字符串双引号

# ========================================
# 第二部分：三引号字符串测试
# ========================================

# 三单引号字符串
triple_single = '''This is a triple single quote string
可以包含多行
和各种字符 # 这个#在字符串内，不是注释
'''  # 这个注释应该被删除

# 三双引号字符串
triple_double = """This is a triple double quote string
可以包含多行
和各种字符 # 这个#在字符串内，不是注释
"""  # 这个注释应该被删除

# 包含问题的复杂字符串
complex_code = '''regex_pattern_3 = r"""
#               # #字符
[a-z]+          # 小写字母
\\d+             # 数字
"""

# 普通注释
x = 1  # 行内注释
'''

# 复杂引号字符串
text1 = """! "#$%&'()*+,-./0\9:;<=>?@A-Z[\]^_`a#z{|}~°¢¥→←↑↓■""" # 这个注释应该被删除
text2 = """! '''#$%&'''()*+,-./0\9:;<=>?@A-Z[\]^_`a#z{|}~°¢¥→←↑↓■""" # 这个注释应该被删除
text3 = '''! "#$%&'()*+,-./0\9:;<=>?@A-Z[\]^_`a#z{|}~°¢¥→←↑↓■''' # 这个注释应该被删除
text4 = '''! """#$%&"""()*+,-./0\9:;<=>?@A-Z[\]^_`a#z{|}~°¢¥→←↑↓■''' # 这个注释应该被删除
text5 = "! '''#$%&'''()*+,-./0\9:;<=>?@A-Z[\]^_`a#z{|}~°¢¥→←↑↓■" # 这个注释应该被删除
text6 = "! \"'''#$%&'''()*+,-./0\9:;<=>?@A-Z[\]^_`a#z{|}~°¢¥→←↑↓■" # 这个注释应该被删除
text7 = "! '''#$%&'()*+,-./0\9:;<=>?@A-Z[\]^_`a#z{|}~°¢¥→←↑↓■" # 这个注释应该被删除
text8 = '! """#$%&"""()*+,-./0\9:;<=>?@A-Z[\]^_`a#z{|}~°¢¥→←↑↓■' # 这个注释应该被删除
text9 = '! \'"""#$%&"""()*+,-./0\9:;<=>?@A-Z[\]^_`a#z{|}~°¢¥→←↑↓■' # 这个注释应该被删除
text10 = '! "#$%&"""()*+,-./0\9:;<=>?@A-Z[\]^_`a#z{|}~°¢¥→←↑↓■' # 这个注释应该被删除
text11 = r"""! '#$%&'''()*+,-./0\9:;<=>?@A-Z[\]^_`a#z{|}~°¢¥→←↑↓■""" # 这个注释应该被删除
text12 = r'''! '#$%&"""()*+,-./0\9:;<=>?@A-Z[\]^_`a#z{|}~°¢¥→←↑↓■''' # 这个注释应该被删除
text13 = r"""! "\""#$%&'''()*+,-./0\9:;<=>?@A-Z[\]^_`a#z{|}~°¢¥→←\\/""" # 这个注释应该被删除
text14 = r'''! "\""#$%&"""()*+,-./0\9:;<=>?@A-Z[\]^_`a#z{|}~°¢¥→←.\#/''' # 这个注释应该被删除
text15 = r"! '\''#$%&'''()*+,-./0\9:;<=>?@A-Z[\]^_`a#z{|}~°¢¥→←'''" # 这个注释应该被删除
text16 = r'! "\""#$%&"""()*+,-./0\9:;<=>?@A-Z[\]^_`a#z{|}~°¢¥→←"""' # 这个注释应该被删除


# ========================================
# 第三部分：嵌套引号组合测试
# ========================================

# 单引号内包含双引号
single_contains_double = 'He said "Hello World"'  # 注释：单引号包含双引号
single_contains_double_hash = 'Command: "echo #test"'  # 注释：包含#的命令

# 双引号内包含单引号
double_contains_single = "It's a beautiful day"  # 注释：双引号包含单引号
double_contains_single_hash = "Don't use # in passwords"  # 注释：包含#的警告

# 单引号内包含三引号
single_contains_triple_double = 'Code: """print("hello")"""'  # 注释：单引号包含三双引号
single_contains_triple_single = "Code: '''print('hello')'''"  # 注释：双引号包含三单引号

# ========================================
# 第四部分：极端复杂组合测试
# ========================================

# 三引号内包含各种引号和#
complex_1 = '''
这里包含各种引号组合：
- 单引号: 'test'
- 双引号: "test"
- 井号字符: # 这个#在字符串内
- 命令示例: echo "hello # world"
- 转义示例: 'It\\'s a test'
- 引号样本: "sample" and 'sample'
'''  # 这个注释应该被删除

complex_2 = """
更复杂的组合：
- Python代码: print('Hello # World')
- Shell命令: grep "#pattern" file.txt
- 混合引号: 'He said "Use # carefully"'
- 转义序列: "Line 1\nLine 2\tTab"
- 特殊字符: !@#$%^&*()_+-=[]{}|;:,.<>?
"""  # 这个注释也应该被删除

# ========================================
# 第五部分：边界情况测试
# ========================================

# 连续引号
consecutive_quotes_1 = 'First' + 'Second'  # 连续字符串
consecutive_quotes_2 = "First" "Second"  # Python自动连接
consecutive_quotes_3 = '''First''' + """Second"""  # 混合三引号

# 引号和#的紧密组合
tight_combination_1 = '#not_comment_but_string'  # 这是注释，#在引号外
tight_combination_2 = "'#this_is_in_string'"  # 这是注释，#在字符串内
tight_combination_3 = '"#this_is_also_in_string"'  # 这是注释，#在字符串内

# 特殊转义情况
escaped_quotes_1 = 'It\'s a \'test\' with # character'  # 转义单引号
escaped_quotes_2 = "She said \"Hello # World\""  # 转义双引号
escaped_quotes_3 = 'Backslash: \\ and hash: #'  # 转义反斜杠

# ========================================
# 第六部分：实际代码场景测试
# ========================================

# 正则表达式字符串
regex_pattern_1 = r'[a-zA-Z0-9#]+'  # 原始字符串with #
regex_pattern_2 = r"^#.*$"  # 匹配以#开头的行
regex_pattern_3 = r'''
    ^               # 行开始
    ['"]{1,3}       # 1-3个引号
    .*?             # 任意字符
    #               # #字符
    .*?             # 任意字符
    ['"]{1,3}       # 1-3个引号
    $               # 行结束
'''  # 复杂正则表达式

# SQL查询字符串
sql_query_1 = "SELECT * FROM table WHERE column = '#value'"  # SQL with #
sql_query_2 = '''
    SELECT 
        column1,
        column2  -- 这里的 # 在字符串内，不是Python注释
    FROM table 
    WHERE condition = "#test"
'''  # 多行SQL

# 命令行字符串
command_1 = 'grep "#pattern" /var/log/file.log'  # grep命令
command_2 = "awk '{print $1 \"#\" $2}' file.txt"  # awk命令
command_3 = '''
for file in *.py; do
    echo "Processing: $file  # Current file"
    python "$file"
done
'''  # shell脚本

# ========================================
# 第七部分：Unicode和特殊字符测试
# ========================================

# Unicode字符串
unicode_1 = '中文字符串 # 包含井号'  # 中文注释
unicode_2 = "Emoji: 🔥💯 # Special chars"  # Emoji注释
unicode_3 = '''
多种语言：
- English: "Hello # World"
- 中文: "你好 # 世界"
- Español: "Hola # Mundo"
- 日本語: "こんにちは # 世界"
'''  # 多语言注释

# 特殊Unicode引号
special_quotes_1 = 'Left quote: " Right quote: "'  # Unicode引号
special_quotes_2 = "Single quotes: ' and '"  # Unicode单引号

# ========================================
# 第八部分：格式化字符串测试
# ========================================

# f-string测试
name = "World"
f_string_1 = f'Hello {name} # This # is in f-string'  # f-string注释
f_string_2 = f"Value: {42 + 8} # Math in f-string"  # f-string计算注释
f_string_3 = f'''
Multi-line f-string:
Name: {name}
Hash: # character
'''  # 多行f-string注释

# .format()测试
format_string_1 = 'Hello {} # Format string'.format(name)  # format注释
format_string_2 = "Value: {value} # Named format".format(value=100)  # 命名format注释

# % 格式化测试
percent_string_1 = 'Hello %s # Percent format' % name  # 百分号格式化注释
percent_string_2 = "Value: %d # Integer format" % 42  # 整数格式化注释

# ========================================
# 第九部分：字符串方法和操作测试
# ========================================

# 字符串方法
method_test_1 = 'hello # world'.upper()  # 字符串方法注释
method_test_2 = "HELLO # WORLD".lower()  # 大小写转换注释
method_test_3 = 'a#b#c'.split('#')  # 分割方法注释

# 字符串拼接
concat_test_1 = 'Part1 #' + ' Part2'  # 拼接注释1
concat_test_2 = "Base" + " # " + "Extension"  # 拼接注释2

# 字符串乘法
multiply_test = 'Hash # ' * 3  # 字符串重复注释

# ========================================
# 第十部分：容器中的字符串测试
# ========================================

# 列表中的字符串
string_list = [
    'String 1 # with hash',  # 列表元素注释1
    "String 2 # with hash",  # 列表元素注释2
    '''String 3
    # with hash
    ''',  # 列表元素注释3
]

# 字典中的字符串
string_dict = {
    'key1': 'value # 1',  # 字典值注释1
    "key2": "value # 2",  # 字典值注释2
    '''key3''': """value # 3""",  # 字典值注释3
}

# 元组中的字符串
string_tuple = (
    'Tuple element # 1',  # 元组元素注释1
    "Tuple element # 2",  # 元组元素注释2
)

# ========================================
# 第十一部分：函数和类中的字符串测试
# ========================================

def test_function1():
    '''
    测试函数的文档字符串（单三引号）
    包含 # 字符的说明
    '''
    # 函数内注释
    local_var = 'Function local # string'  # 局部变量注释
    return "Function return # value"  # 返回值注释

def test_function2():  # 这个注释在函数定义上
    """
    测试函数的文档字符串
    包含 # 字符的说明
    """
    # 函数内注释
    local_var = 'Function local # string'  # 局部变量注释
    return "Function return # value"  # 返回值注释

def test_function3(self, param1=None, param2=False,
                param3='default', param4="*.py", # if 这个注释里面有一些字符串 try:
                param5="""
                三重引号字符串参数默认值
                """):
    """
    测试函数的文档字符串
    包含 # 字符的说明
    """
    # 函数内注释
    local_var = 'Function local # string'  # 局部变量注释
    return "Function return # value"  # 返回值注释

async def async_test_function():
    """
    测试异步函数的'''文档字符串'''
    包含 '''# 字符的说明
    """
    # 异步函数内注释
    local_var = 'Async """function""" local # string'  # 局部变量注释
    return "Async '''function''' return # value"  # 返回值注释

class TestClass1:
    '''
    测试类的文档字符串（单三引号）
    包含各种 # 字符说明
    '''
    def __init__(self):
        '''
        单引号方法文档字符串 # 包含井号
        '''
        # 构造函数注释
        self.instance_var = 'Instance # variable'  # 实例变量注释

    def method(self):
        """
        方法文档字符串 # 包含井号
        """
        # 方法内注释
        return 'Method return # value'  # 方法返回注释

class TestClass2: # 这个注释在类定义上 还有一些特殊字符 if True:
    """
    测试类的文档字符串
    包含各种 # 字符说明
    """
    
    class_var = 'Class variable # string'  # 类变量注释
    
    def __init__(self):
        # 构造函数注释
        self.instance_var = "Instance # variable"  # 实例变量注释
    
    def method1(self) :
        """方法文档字符串 # 包含井号 方法冒号还有个空格"""
        # 方法内注释
        return 'Method return # value'  # 方法返回注释

    def method2(self, param1="", param2="'''",
                param5="""参数空一行""",
                                                # 这里有一个参数空行
                param6=r'''\#""""""#"""\\'''):
        """双引号方法文档字符串 # 包含井号"""
        # 方法内注释
        return 'Method return # value'  # 方法返回注释

    def method3(self, param1=None, param2=False,
                param3='default', param4=(2>1),
                param5=                 # 这个有一个等号
                """
                三重引号字符串参数默认值
                """,
                param6=1):
        '''单引号方法文档字符串 # 包含井号'''
        # 方法内注释
        return '''Method return # value'''  # 方法返回注释

# ========================================
# 第十二部分：异常和错误处理中的字符串
# ========================================

try:
    # 尝试执行可能出错的代码
    result = eval('1 + 1 # Simple math')  # eval注释
except Exception as e:
    # 异常处理注释
    error_msg = f'Error occurred: {e} # Exception handling'  # 异常消息注释
    print("Error message # in exception")  # 错误输出注释
finally:
    # 最终执行块注释
    cleanup_msg = 'Cleanup completed # finally block'  # 清理消息注释

# ========================================
# 第十三部分：装饰器和高级特性中的字符串
# ========================================

def decorator_with_string(func):
    """装饰器文档字符串 # 包含井号"""
    # 装饰器注释
    def wrapper():
        # 包装函数注释
        print('Before function # execution')  # 执行前注释
        result = func()
        print("After function # execution")  # 执行后注释
        return result
    return wrapper

@decorator_with_string
def decorated_function():
    """被装饰函数的文档字符串 # 包含井号"""
    # 被装饰函数注释
    return 'Decorated function # result'  # 装饰函数返回注释

# ========================================
# 第十四部分：生成器和推导式中的字符串
# ========================================

# 列表推导式
list_comprehension = ['Item # ' + str(i) for i in range(3)]  # 列表推导式注释

# 字典推导式
dict_comprehension = {f'key#{i}': f'value#{i}' for i in range(3)}  # 字典推导式注释

# 生成器表达式
generator_expr = (f'Generated # {i}' for i in range(3))  # 生成器表达式注释

def string_generator():
    """生成器函数文档字符串 # 包含井号"""
    # 生成器函数注释
    for i in range(3):
        yield f'Yielded # {i}'  # yield注释

# ========================================
# 测试总结
# ========================================

if __name__ == "__main__":
    # 主程序注释
    print("Complex string test file loaded successfully # Test complete")  # 成功消息注释
    
    # 最终测试统计
    total_strings = 50  # 估计的字符串总数
    total_comments = 100  # 估计的注释总数
    
    print(f'''
    测试文件统计：
    - 预计字符串数量: {total_strings} # 包含各种引号组合
    - 预计注释数量: {total_comments} # 应该被完全删除
    - 井号字符: 在字符串内的应该保留 # 在注释中的应该被删除
    ''')  # 统计信息注释
