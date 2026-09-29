#!/usr/bin/env python3
"""Restore optional apple images from a pinned snapshot; verify every SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def matches(path, entry):
    if not path.is_file() or path.stat().st_size != entry['bytes']:
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']


def restore(manifest, destination, subset, archive=None, verify_only=False):
    entries = [e for e in manifest['files']
               if subset == 'all' or e['path'].split('/')[0] == subset]
    if not entries:
        raise ValueError('No files selected.')
    destination = destination.resolve()
    for entry in entries:
        path = PurePosixPath(entry['path'])
        if path.is_absolute() or '..' in path.parts or '\\' in entry['path']:
            raise ValueError('Unsafe manifest path.')
        if not (destination / entry['path']).resolve().is_relative_to(destination):
            raise ValueError('Destination path escapes the data directory.')
    missing = [e for e in entries if not matches(destination / e['path'], e)]
    if verify_only:
        if missing:
            raise ValueError(f'{len(missing)} of {len(entries)} files missing or invalid.')
    elif missing:
        with tempfile.TemporaryDirectory(prefix='pdi-download-') as temp:
            temp = Path(temp)
            if archive is None:
                archive = temp / 'snapshot.zip'
                print('Downloading pinned snapshot (both subsets are in the archive)...')
                with urllib.request.urlopen(manifest['url'], timeout=120) as source:
                    with archive.open('wb') as target:
                        shutil.copyfileobj(source, target)
            # Validate all selected replacements before writing any destination files.
            with zipfile.ZipFile(archive) as source:
                for entry in missing:
                    member = manifest['archive_prefix'] + entry['path']
                    info = source.getinfo(member)
                    if info.file_size != entry['bytes']:
                        raise ValueError(f"Size mismatch: {entry['path']}")
                    target = temp / 'verified' / entry['path']
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(info) as src, target.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                    if not matches(target, entry):
                        raise ValueError(f"SHA-256 mismatch: {entry['path']}")
            for entry in missing:
                target = destination / entry['path']
                target.parent.mkdir(parents=True, exist_ok=True)
                # Atomic replacement also works when temp and destination differ in filesystem.
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
                    staging = Path(out.name)
                    try:
                        with (temp / 'verified' / entry['path']).open('rb') as src:
                            shutil.copyfileobj(src, out)
                    except BaseException:
                        staging.unlink(missing_ok=True)
                        raise
                try:
                    staging.replace(target)
                finally:
                    staging.unlink(missing_ok=True)
    print(f'Verified {len(entries)} images in {destination}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subset', choices=['without', 'with', 'all'], default='without')
    parser.add_argument('--destination', type=Path, default=ROOT / 'data' / 'raw')
    parser.add_argument('--archive', type=Path, help='Use an already downloaded snapshot ZIP.')
    parser.add_argument('--verify-only', action='store_true', help='Check local files without downloading.')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'data' / 'download_manifest.json').read_text())
    try:
        restore(manifest, args.destination, args.subset, args.archive, args.verify_only)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f'Dataset error: {exc}\n')


if __name__ == '__main__':
    main()
