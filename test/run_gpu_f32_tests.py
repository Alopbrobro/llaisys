#!/usr/bin/env python3
"""Run all GPU unit tests with F32 only (CUDA kernels only support F32 currently)."""
import subprocess, sys

tests = [
    ("add", "test/ops/add.py"),
    ("argmax", "test/ops/argmax.py"),
    ("embedding", "test/ops/embedding.py"),
    ("linear", "test/ops/linear.py"),
    ("rms_norm", "test/ops/rms_norm.py"),
    ("rope", "test/ops/rope.py"),
    ("swiglu", "test/ops/swiglu.py"),
    ("self_attention", "test/ops/self_attention.py"),
]

passed = []
failed = []

for name, path in tests:
    print(f"\n{'='*60}")
    print(f"Testing {name} on nvidia (F32 only)")
    print(f"{'='*60}")
    
    # Read the test file and modify to only test F32
    with open(path) as f:
        code = f.read()
    
    # Replace __file__ reference and force F32 only + nvidia device
    modified = code.replace("__file__", f"'{path}'")
    
    # Force nvidia device and F32
    modified = modified.replace(
        'args = parser.parse_args()',
        'args = parser.parse_args(["--device", "nvidia"])'
    )
    
    # Replace dtype lists to only include f32
    for old in [
        '["f32", "f16", "bf16"]',
        '[\n        # type\n        "f32",\n        "f16",\n        "bf16",\n    ]',
    ]:
        if old in modified:
            modified = modified.replace(old, '["f32"]')
    
    for old in [
        '[\n        # type, atol, rtol\n        ("f32", 1e-5, 1e-5),\n        ("f16", 1e-3, 1e-3),\n        ("bf16", 1e-3, 1e-3),\n    ]',
        '[\n        # type, atol, rtol\n        ("f32", 1e-4, 1e-4),\n        ("f16", 5e-2, 5e-2),\n        ("bf16", 1e-1, 1e-1),\n    ]',
        '[\n        # type, atol, rtol\n        ("f32", 1e-3, 1e-3),\n        ("f16", 5e-2, 5e-2),\n        ("bf16", 1e-1, 1e-1),\n    ]',
    ]:
        if old in modified:
            modified = modified.replace(old, '[("f32", 1e-5, 1e-5)]')
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", modified],
            capture_output=True, text=True, timeout=60, cwd="/home/bbq/llaisys"
        )
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        if result.returncode != 0:
            print(f"STDERR: {result.stderr[-500:]}")
            failed.append(name)
        else:
            passed.append(name)
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT!")
        failed.append(name)

print(f"\n{'='*60}")
print(f"Results: {len(passed)} passed, {len(failed)} failed")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
