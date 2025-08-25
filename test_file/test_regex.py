#!/usr/bin/env python3

import re

# 原始的正则表达式
pattern_with_comments = r'''
    ^               # 行开始
    ['"]{1,3}       # 1-3个引号
    .*?             # 任意字符
    .*?             # 任意字符
    ['"]{1,3}       # 1-3个引号
    $               # 行结束
'''

# 如果删除注释后的正则表达式
pattern_without_comments = r'''
    ^
    ['"]{1,3}
    .*?
    .*?
    ['"]{1,3}
    $
'''

print('原始长度:', len(pattern_with_comments))
print('删除注释后长度:', len(pattern_without_comments))
print('两者是否相等:', pattern_with_comments == pattern_without_comments)

# 测试正则表达式是否仍然有效
test_string = "'''hello'''"
try:
    result1 = re.search(pattern_with_comments, test_string, re.VERBOSE)
    result2 = re.search(pattern_without_comments, test_string, re.VERBOSE)
    print('原始模式匹配结果:', result1 is not None)
    print('删除注释后模式匹配结果:', result2 is not None)
except Exception as e:
    print('错误:', e)
