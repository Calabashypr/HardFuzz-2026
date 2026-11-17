"""Normalize local tensor sizes."""
import re

_BYTE_SIZE = r'(\w+)\.SetSize\((.*?)\s*\*\s*(?:\((?:u?int32_t|int)\)\s*)?sizeof\(([^)]+)\)\);'

def normalize_local_tensor_sizes(code):
    # Update sizes and offsets.
    code = re.sub(
        _BYTE_SIZE + r'(\s*)offset_ptr\s*\+=\s*\1\.GetSize\(\)\s*;',
        r'\1.SetSize(\2);\4offset_ptr += \1.GetSize() * sizeof(\3);',
        code)
    # Normalize remaining sizes.
    return re.sub(_BYTE_SIZE, r'\1.SetSize(\2);', code)
