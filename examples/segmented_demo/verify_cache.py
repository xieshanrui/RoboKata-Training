"""Full SHA pass once; every subsequent read checks immutable file metadata."""
from pathlib import Path
import argparse,hashlib,json,time

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def verify(cache):
    cache=Path(cache);manifest_path=cache/'cache_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'));assert manifest['passed']
    started=time.perf_counter();records={}
    for identity,entry in manifest['records'].items():
        assert Path(entry['file']).name==entry['file']
        path=cache/entry['file'];before=path.stat()
        assert before.st_size==entry['bytes'] and sha(path)==entry['sha256'],identity
        after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        records[identity]=dict(bytes=after.st_size,mtime_ns=after.st_mtime_ns)
    result=dict(passed=True,cache_manifest_sha256=sha(manifest_path),fingerprint=manifest['fingerprint'],
        all_file_sha256_verified=True,records=records,seconds=time.perf_counter()-started,
        read_policy='Check size and nanosecond mtime against this full-SHA pass on every file load; reject mutations')
    destination=cache/'verification_receipt.json';temporary=destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');temporary.replace(destination)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);args=p.parse_args()
    result=verify(args.cache)
    print(json.dumps(dict(passed=True,records=len(result['records']),seconds=result['seconds'],
        cache_manifest_sha256=result['cache_manifest_sha256'])),flush=True)
