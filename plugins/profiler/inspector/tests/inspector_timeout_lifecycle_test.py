#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build/run Inspector CPU lifecycle tests; CUDA APIs are not exercised."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


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
    assert len(timeouts) == 5
    assert [r['metadata']['event_type'] for r in timeouts] == [
        'collective_timeout_gpu_issue', 'collective_timeout_communication_issue',
        'collective_timeout_gpu_and_communication_issue', 'collective_timeout_gpu_issue',
        'collective_timeout_communication_issue']
    assert all(set(r) == {'header', 'metadata', 'coll_timeout'} for r in timeouts)
    # Channels proven by later progress are folded into the r6 fields; no keys are added.
    assert all(set(r['header']) == {'id', 'comm_name', 'rank', 'n_ranks', 'nnodes'} for r in timeouts)
    assert all(set(r['metadata']) == {'event_type', 'job_id', 'retry_count', 'dump_timestamp_us',
                                      'hostname', 'pid'} for r in timeouts)
    assert all(set(r['coll_timeout']) == {
        'coll', 'coll_seq_number', 'algo', 'proto', 'coll_msg_size_bytes', 'timeout_elapsed_ms',
        'queue_wait_elapsed_ms', 'kernel_execution_elapsed_ms', 'expected_channels',
        'estimated_channel_payload_bytes', 'started_channels', 'completed_channels',
        'started_channel_mask', 'completed_channel_mask', 'min_exec_time_us', 'min_channel',
        'max_exec_time_us', 'max_channel', 'timing_source'} for r in timeouts)
    folded = timeouts[-1]['coll_timeout']
    assert (folded['expected_channels'], folded['started_channels'], folded['completed_channels'],
            folded['started_channel_mask'], folded['completed_channel_mask']) == (4, 4, 3, 15, 7)
    assert all('proxy' not in r for r in records)
    assert len([r for r in records if 'coll_perf' in r]) == 3
    assert len([r for r in records if 'p2p_perf' in r]) == 1
    print('PASS historical JSON schema, normal collective/P2P output, non-exempt and folded timeout output')

    # Keep filesystem fixtures under the build directory, isolated across runs.
    paths = Path(tempfile.mkdtemp(prefix='job-paths-', dir=out))
    timeout_root = paths/'timeouts'
    writer = out/'lifecycle-test'
    for job_id in ['j-alpha', 'j-beta', None, '']:
        for retry_count in ['0', '3', None, '']:
            env = os.environ.copy()
            for name, value in [('JOB_ID', job_id), ('RETRY_COUNT', retry_count)]:
                env.pop(name, None)
                if value is not None:
                    env[name] = value
            result = subprocess.Popen([str(writer), '--timeout-path', str(timeout_root)], env=env)
            assert result.wait(timeout=30) == 0
            directory = timeout_root/(job_id or 'unknown')/(retry_count or 'unknown')
            expected = directory/f'{os.uname().nodename}-pid{result.pid}.log'
            record, = [json.loads(line) for line in expected.read_text().splitlines()]
            assert record['metadata']['job_id'] == (job_id or 'unknown')
            assert record['metadata']['retry_count'] == (retry_count or 'unknown')
    assert not list(timeout_root.glob('*.log'))
    assert {p.name for p in timeout_root.iterdir()} == {'j-alpha', 'j-beta', 'unknown'}
    for job_dir in timeout_root.iterdir():
        assert not list(job_dir.glob('*.log'))
        assert {p.name for p in job_dir.iterdir()} == {'0', '3', 'unknown'}

    # Multiple ranks race to create the same root, job and retry directories.
    shared = paths/'concurrent'
    env = dict(os.environ, JOB_ID='j-shared', RETRY_COUNT='0')
    ranks = [subprocess.Popen([str(writer), '--timeout-path', str(shared)], env=env) for _ in range(16)]
    assert all(rank.wait(timeout=30) == 0 for rank in ranks)
    assert len(list((shared/'j-shared'/'0').glob('*.log'))) == len(ranks)

    for name in ['JOB_ID', 'RETRY_COUNT']:
        for invalid in ['.', '..', '../escape', '/absolute', 'nested/job', 'j'*300]:
            env = dict(os.environ, JOB_ID='j-alpha', RETRY_COUNT='0')
            env[name] = invalid
            result = subprocess.run([str(writer), '--timeout-path', str(timeout_root)], env=env, timeout=30)
            assert result.returncode == 2
    assert not (paths/'escape').exists() and not (timeout_root/'escape').exists()
    for blocked, job_id, retry_count in [(timeout_root/'blocked', 'blocked', '0'),
                                         (timeout_root/'j-alpha'/'blocked', 'j-alpha', 'blocked')]:
        blocked.write_text('existing file')
        result = subprocess.run([str(writer), '--timeout-path', str(timeout_root)],
                                env=dict(os.environ, JOB_ID=job_id, RETRY_COUNT=retry_count), timeout=30)
        assert result.returncode == 2 and blocked.read_text() == 'existing file'
    print('PASS JOB_ID/RETRY_COUNT paths, fallback, lazy files, concurrent ranks, invalid paths and collisions')


if __name__ == '__main__':
    main()
