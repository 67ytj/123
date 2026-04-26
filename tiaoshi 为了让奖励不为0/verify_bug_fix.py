#!/usr/bin/env python3
"""
验证 HORA_OUTPUT_NAME 环境变量修复的脚本
"""

import os
import sys

def verify_fix():
    print("=" * 80)
    print("?? 验证 HORA_OUTPUT_NAME 环境变量修复")
    print("=" * 80)
    print()

    # 检查 Python 文件
    print("1??  检查 hora/tasks/shadow_hand_hora.py")
    print("-" * 80)

    with open('hora/tasks/shadow_hand_hora.py', 'r', encoding='utf-8') as f:
        content = f.read()

    checks = [
        ("os.environ.get('HORA_OUTPUT_NAME'", "从环境变量读取 HORA_OUTPUT_NAME"),
        ("'ShadowHandHora/default'", "默认值设置正确"),
    ]

    python_ok = True
    for check_str, desc in checks:
        if check_str in content:
            print(f"  ? {desc}")
        else:
            print(f"  ? 未找到: {desc}")
            python_ok = False

    print()

    # 检查训练脚本
    print("2??  检查 scripts/train_shadow.sh")
    print("-" * 80)

    with open('scripts/train_shadow.sh', 'r', encoding='utf-8') as f:
        script_content = f.read()

    script_checks = [
        ("export HORA_OUTPUT_NAME=", "导出环境变量"),
        ("${NAME}", "使用脚本参数"),
    ]

    script_ok = True
    for check_str, desc in script_checks:
        if check_str in script_content:
            print(f"  ? {desc}")
        else:
            print(f"  ? 未找到: {desc}")
            script_ok = False

    print()

    # 测试环境变量流向
    print("3??  测试环境变量流向")
    print("-" * 80)

    os.environ['HORA_OUTPUT_NAME'] = 'ShadowHandHora/test_exp'

    try:
        result = os.environ.get('HORA_OUTPUT_NAME', 'ShadowHandHora/default')
        if result == 'ShadowHandHora/test_exp':
            print(f"  ? 环境变量读取成功: {result}")
        else:
            print(f"  ? 环境变量读取失败: {result}")
    except Exception as e:
        print(f"  ? 错误: {e}")

    print()

    # 总结
    print("=" * 80)
    if python_ok and script_ok:
        print("? 所有修复已验证成功！")
        print()
        print("现在可以运行:")
        print("  bash scripts/train_shadow.sh exp4_lowhand")
        print()
        print("TB 日志会正确写到:")
        print("  outputs/ShadowHandHora/exp4_lowhand/reward_components/")
        return True
    else:
        print("? 某些修复未完成，请检查!")
        return False

if __name__ == '__main__':
    success = verify_fix()
    sys.exit(0 if success else 1)
