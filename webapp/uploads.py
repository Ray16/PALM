"""Bounded ZIP extraction for uploaded dataset folders."""
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile


def extract_dataset_zip(archive, destination, max_bytes, max_files=5000):
    destination = Path(destination)
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
        if len(entries) > max_files:
            raise ValueError(f'ZIP contains too many entries (maximum {max_files:,})')
        total = 0
        for item in entries:
            name = PurePosixPath(item.filename.replace('\\', '/'))
            mode = item.external_attr >> 16
            if name.is_absolute() or '..' in name.parts or (name.parts and ':' in name.parts[0]):
                raise ValueError('ZIP paths must stay inside the dataset folder')
            if stat.S_ISLNK(mode):
                raise ValueError('ZIP symbolic links are not supported')
            if item.flag_bits & 1:
                raise ValueError('Encrypted ZIP files are not supported')
            total += item.file_size
            if total > max_bytes:
                raise ValueError('The uncompressed ZIP exceeds this server’s extraction limit')
        destination.mkdir(parents=True, exist_ok=True)
        written = 0
        for item in entries:
            target = destination.joinpath(*PurePosixPath(item.filename.replace('\\', '/')).parts)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(item) as src, target.open('wb') as dst:
                while chunk := src.read(1024 * 1024):
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError('The uncompressed ZIP exceeds this server’s extraction limit')
                    dst.write(chunk)
    visible = [p for p in destination.iterdir() if not p.name.startswith(('.', '_'))]
    return visible[0] if len(visible) == 1 else destination
