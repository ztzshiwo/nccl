#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build/run Inspector CPU lifecycle tests; CUDA APIs are not exercised."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser()
    sanitizer = parser.add_mutually_exclusive_group()
    sanitizer.add_argument('--sanitize', action='store_true')
    sanitizer.add_argument('--thread-sanitize', action='store_true')
    args = parser.parse_args()
    plugin = Path(__file__).resolve().parents[1]
    root = plugin.parents[2]
    os.chdir(root)
    mode = 'asan' if args.sanitize else 'tsan' if args.thread_sanitize else 'normal'
    out = root / 'build/tests/inspector-lifecycle' / mode
    out.mkdir(parents=True, exist_ok=True)
    sources = ['inspector_plugin.cc', 'inspector_handles.cc', 'inspector_event_pool.cc',
               'inspector_ring.cc', 'inspector_json.cc',
               'inspector_prom.cc', 'inspector_cudawrap.cc', 'json.cc']
    flags = ['-std=c++14', '-O1', '-g', '-Wall', '-Wextra', '-Werror', '-pthread',
             '-DINSPECTOR_TESTING', '-I'+str(plugin), '-I'+str(plugin/'nccl'),
             '-I'+str(plugin/'tests/cpu_stubs')]
    if args.sanitize:
        flags += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    if args.thread_sanitize:
        flags += ['-fsanitize=thread', '-fno-omit-frame-pointer']
    cmd = shlex.split(os.environ.get('CXX', 'c++')) + flags
    cmd += [str(plugin/'tests/inspector_timeout_lifecycle_test.cc')] + [str(plugin/s) for s in sources]
    cmd += ['-o', str(out/'lifecycle-test')]
    if os.uname().sysname == 'Linux':
        cmd += ['-ldl']
    subprocess.run(cmd, check=True)
    output = out/'regression.jsonl'
    subprocess.run([str(out/'lifecycle-test'), str(output)], check=True, timeout=120)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    timeouts = [r for r in records if 'coll_timeout' in r]
    assert len(timeouts) == 4
    assert [r['metadata']['event_type'] for r in timeouts] == [
        'collective_timeout_gpu_issue', 'collective_timeout_communication_issue',
        'collective_timeout_gpu_and_communication_issue', 'collective_timeout_gpu_issue']
    assert all(set(r) == {'header', 'metadata', 'coll_timeout'} for r in timeouts)
    assert all('proxy' not in r for r in records)
    assert len([r for r in records if 'coll_perf' in r]) == 3
    assert len([r for r in records if 'p2p_perf' in r]) == 1
    print('PASS historical JSON schema, normal collective/P2P output, non-exempt timeout output')


if __name__ == '__main__':
    main()
