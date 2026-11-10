import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / 'runs/evidence_compression_probe_v2'
SOURCE = ROOT / 'runs/joint_command_development1_archive/heavy_skid/evidence'
ZSTD = Path('/opt/homebrew/bin/zstd')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def main():
    OUT.mkdir(exist_ok=False)
    files = [SOURCE / f'frame_{i:06d}.npz' for i in range(200)]
    expected = {p.name: {'size': p.stat().st_size, 'sha256': sha(p)} for p in files}
    freeze = {'source': str(SOURCE), 'inputs': expected,
              'sources': {str(p): sha(p) for p in [HERE/'protocol.md',Path(__file__)]},
              'zstd': {'path': str(ZSTD), 'sha256': sha(ZSTD),
                       'version': subprocess.check_output([str(ZSTD),'--version'],text=True).strip()},
              'recipe': {'level': 10, 'threads': 1, 'long': 27}}
    write(OUT/'freeze.json',freeze)
    started = time.monotonic()
    archive = OUT/'archive.tar.zst'
    with subprocess.Popen(['tar','-C',str(SOURCE),'-cf','-',*[p.name for p in files]],stdout=subprocess.PIPE, env={**os.environ, 'COPYFILE_DISABLE': '1'}) as producer:
        with archive.open('xb') as target:
            compressed = subprocess.run([str(ZSTD),'-10','--long=27','-T1','-q','-c'],stdin=producer.stdout,stdout=target,check=True)
        producer.stdout.close()
        if producer.wait() != 0:
            raise RuntimeError('tar producer failed')
    compression_seconds = time.monotonic()-started
    started = time.monotonic()
    actual = {}
    with subprocess.Popen([str(ZSTD),'-d','-q','-c',str(archive)],stdout=subprocess.PIPE) as producer:
        with tarfile.open(fileobj=producer.stdout,mode='r|') as entries:
            for entry in entries:
                if not entry.isfile() or entry.name in actual:
                    raise ValueError('unexpected or duplicate archive member')
                with entries.extractfile(entry) as f:
                    digest = hashlib.file_digest(f,'sha256').hexdigest()
                actual[entry.name] = {'size': entry.size, 'sha256': digest}
        producer.stdout.close()
        if producer.wait() != 0:
            raise RuntimeError('zstd decoder failed')
    assert actual == expected
    assert expected == {p.name:{'size':p.stat().st_size,'sha256':sha(p)} for p in files}
    original_bytes = sum(v['size'] for v in expected.values())
    write(OUT/'receipt.json',{'passed':True,'freeze_sha256':sha(OUT/'freeze.json'),
        'selected_files':len(files),'source_bytes':original_bytes,'archive_bytes':archive.stat().st_size,
        'archive_to_source_ratio':archive.stat().st_size/original_bytes,'archive_sha256':sha(archive),
        'compression_seconds':compression_seconds,'verification_seconds':time.monotonic()-started,
        'all_members_exact':True,'source_unchanged':True,'source_deleted':False,
        'limitation':'First200correlatedskidframes, not a worst-case budget or storage transport proof.'})
    print((OUT/'receipt.json').read_text())


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if OUT.exists() and not (OUT/'receipt.json').exists():
            write(OUT/'failure.json', {'passed': False, 'error': f'{type(exc).__name__}: {exc}', 'source_deleted': False})
        raise
