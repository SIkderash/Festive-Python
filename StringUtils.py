import re
import math

class StringUtils:
    @staticmethod
    def link(mark, *args):
        return mark.join(args)

    @staticmethod
    def link_split(key, split, size):
        return split.join([f"{key}=?" for _ in range(size)])

    @staticmethod
    def find_first_char_to_upper_case(text):
        return text[0].upper() if text else "-"

    @staticmethod
    def is_empty(*strings):
        return any(not s for s in strings)

    @staticmethod
    def half_corner(str):
        regs = [
            "！", "，", "。", "；", "~", "《", "》", "（", "）", "？", "”", "｛", "｝", "“", "：", "【", "】", "”", "‘", "’",
            "!", ",", ".", ";", "`", "<", ">", "(", ")", "?", "'", "{", "}", "\"", ":", "{", "}", "\"", "'", "'"
        ]
        for i in range(len(regs) // 2):
            str = re.sub(regs[i], regs[i + len(regs) // 2], str)
        return str

    @staticmethod
    def to_arrays(text):
        return list(text)

    @staticmethod
    def round(value, places):
        if places < 0:
            raise ValueError("Places must be non-negative")
        return round(value, places)
