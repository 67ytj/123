#!/usr/bin/env python3
"""Quick syntax check for shadow_hand_hora.py modifications."""

import sys
import ast

try:
    with open('hora/tasks/shadow_hand_hora.py', 'r', encoding='utf-8') as f:
        code = f.read()

    # Try to parse the file as valid Python
    ast.parse(code)
    print("? [OK] File syntax is valid!")
    sys.exit(0)

except SyntaxError as e:
    print(f"? [SYNTAX ERROR] {e}")
    sys.exit(1)
except Exception as e:
    print(f"? [ERROR] {e}")
    sys.exit(1)
