#!/usr/bin/env python3
"""
Final verification checklist for ShadowHandHora TensorBoard modifications.
Run this to confirm all changes are correctly applied.
"""

import os
import ast
import sys

def check_file_exists(path):
    """Check if file exists."""
    if os.path.exists(path):
        print(f"  ? {path}")
        return True
    else:
        print(f"  ? {path} NOT FOUND")
        return False

def check_syntax(filepath):
    """Check if Python file has valid syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        print(f"  ? Syntax valid")
        return True
    except SyntaxError as e:
        print(f"  ? Syntax error: {e}")
        return False

def check_imports(filepath):
    """Check if required imports exist."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    imports_to_check = [
        ('import os', 'import os'),
        ('from torch.utils.tensorboard import SummaryWriter', 'SummaryWriter'),
    ]

    all_ok = True
    for check_line, display_name in imports_to_check:
        if check_line in content or display_name in content:
            print(f"  ? Import: {display_name}")
        else:
            print(f"  ? Missing import: {display_name}")
            all_ok = False

    return all_ok

def check_init_setup(filepath):
    """Check if TensorBoard writer is initialized in __init__."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    checks = [
        ('self.rew_writer = SummaryWriter', 'SummaryWriter initialization'),
        ('self.rew_log_counter = 0', 'Log counter initialization'),
        ('reward_components', 'Output directory name'),
    ]

    all_ok = True
    for check_str, display_name in checks:
        if check_str in content:
            print(f"  ? {display_name}")
        else:
            print(f"  ? Missing: {display_name}")
            all_ok = False

    return all_ok

def check_logging_logic(filepath):
    """Check if reward logging is implemented in post_physics_step."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    checks = [
        ('self.rew_log_counter += 1', 'Log counter increment'),
        ('if self.rew_log_counter % 50 == 0:', 'Logging condition'),
        ("'rewards/reach'", 'Reach reward logging'),
        ("'rewards/lift_low'", 'Lift low reward logging'),
        ("'rewards/lift_mid'", 'Lift mid reward logging'),
        ("'rewards/lift_high'", 'Lift high reward logging'),
        ("'rewards/penalty'", 'Penalty logging'),
        ("'rewards/total'", 'Total reward logging'),
        ("'diagnostics/tip_contact_force_mean'", 'Contact force diagnostic'),
        ("'diagnostics/ball_height'", 'Ball height diagnostic'),
        ("'diagnostics/mean_tip_dist'", 'Mean tip distance diagnostic'),
        ("'diagnostics/success_rate_4cm'", 'Success rate diagnostic'),
    ]

    all_ok = True
    for check_str, display_name in checks:
        if check_str in content:
            print(f"  ? {display_name}")
        else:
            print(f"  ? Missing: {display_name}")
            all_ok = False

    return all_ok

def main():
    """Run all checks."""
    print("=" * 80)
    print("?? FINAL VERIFICATION CHECKLIST")
    print("=" * 80)
    print()

    all_passed = True

    # Check main file
    print("1??  Main file existence:")
    target_file = 'hora/tasks/shadow_hand_hora.py'
    if not check_file_exists(target_file):
        print("\n? Cannot find main file. Aborting.")
        return False
    print()

    # Check syntax
    print("2??  File syntax:")
    if not check_syntax(target_file):
        print("\n? Syntax error in file. Fix and retry.")
        all_passed = False
    print()

    # Check imports
    print("3??  Required imports:")
    if not check_imports(target_file):
        print("\n? Some imports are missing.")
        all_passed = False
    print()

    # Check __init__ setup
    print("4??  TensorBoard initialization in __init__:")
    if not check_init_setup(target_file):
        print("\n? Some __init__ setup is missing.")
        all_passed = False
    print()

    # Check logging implementation
    print("5??  Reward logging in post_physics_step:")
    if not check_logging_logic(target_file):
        print("\n? Some logging logic is missing.")
        all_passed = False
    print()

    # Summary
    print("=" * 80)
    if all_passed:
        print("? ALL CHECKS PASSED! ?")
        print()
        print("You are ready to:")
        print("  1. Run: bash scripts/train_shadow.sh exp4_lowhand")
        print("  2. Launch: tensorboard --logdir outputs/ShadowHandHora --port 6006")
        print("  3. View: http://localhost:6006 ¡ú SCALARS tab")
        print()
        print("Expected to see 10 new metrics:")
        print("  - rewards/reach, rewards/lift_low, rewards/lift_mid")
        print("  - rewards/lift_high, rewards/penalty, rewards/total")
        print("  - diagnostics/tip_contact_force_mean, diagnostics/ball_height")
        print("  - diagnostics/mean_tip_dist, diagnostics/success_rate_4cm")
        print()
        return True
    else:
        print("? SOME CHECKS FAILED")
        print()
        print("Please review the errors above and fix them.")
        print("Refer to TROUBLESHOOTING.md for more help.")
        print()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
