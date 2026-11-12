"""Length-prefixed NPZ protocol shared by the segmented policy and its caller."""
import io
import json
import struct

import numpy as np

def exact(stream,count):
    result=[]
    while count:
        block=stream.read(count)
        if not block:raise EOFError
        result.append(block);count-=len(block)
    return b''.join(result)

def receive(stream):
    size=struct.unpack('!Q',exact(stream,8))[0]
    if size>256*1024*1024:raise ValueError('Oversized packet')
    with np.load(io.BytesIO(exact(stream,size)),allow_pickle=False) as d:value={k:d[k] for k in d.files}
    return json.loads(value.pop('meta').tobytes().decode('utf-8')),value

def send(stream,meta,**arrays):
    out=io.BytesIO();np.savez_compressed(out,meta=np.frombuffer(json.dumps(meta).encode('utf-8'),dtype=np.uint8),**arrays)
    payload=out.getvalue();stream.write(struct.pack('!Q',len(payload))+payload);stream.flush()
