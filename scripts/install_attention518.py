"""Install the selected complete model archive without overwriting existing files."""
from pathlib import Path
import argparse, hashlib, tempfile, zipfile

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path,nargs='?',default=None,
                   help='Optional unsplit archive. By default, joins models/attention518_parts/*.part*')
    p.add_argument('--output',type=Path,default=Path('runtime/attention518'))
    a=p.parse_args()
    expected='68894a977620a3c5b0ac07f0677e79ebcc3f9941d3b517d7c31f49c88071f43a'
    temporary = None
    if a.archive is None:
        parts=sorted(Path('models/attention518_parts').glob('attention518_primary.zip.part*'))
        if len(parts)!=9: raise FileNotFoundError('Expected 9 model parts. Run git lfs pull, then retry.')
        if any(part.stat().st_size < 1024 for part in parts):
            raise ValueError('Model parts are probably Git LFS pointers. Run git lfs pull first.')
        temporary=tempfile.NamedTemporaryFile(prefix='attention518-',suffix='.zip',delete=False)
        try:
            for part in parts:
                with part.open('rb') as source:
                    for block in iter(lambda:source.read(1024*1024),b''): temporary.write(block)
        finally: temporary.close()
        archive=Path(temporary.name)
    else:
        archive=a.archive
        if not archive.is_file(): raise FileNotFoundError(archive)
    try:
        with archive.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=expected: raise ValueError('Wrong or incomplete Attention518 archive')
        root=a.output.resolve()
        if root.exists(): raise FileExistsError('Destination exists; choose a new empty directory')
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                target=(root/info.filename).resolve()
                if not target.is_relative_to(root): raise ValueError('Unsafe archive path')
            for info in z.infolist():
                target=root/info.filename
                if not info.is_dir():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(z.read(info))
    finally:
        if temporary is not None: archive.unlink(missing_ok=True)
    print('Installed',root)

if __name__=='__main__':main()
