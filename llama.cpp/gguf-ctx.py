#!/usr/bin/env python3
# Reads <arch>.context_length from a GGUF file's metadata header.
# Pure stdlib, no numpy / no gguf package needed.
# Prints the integer to stdout, or nothing (exit 1) on failure.
import struct, sys

_FMT = {0:('B',1),1:('b',1),2:('H',2),3:('h',2),4:('I',4),5:('i',4),
        6:('f',4),7:('?',1),10:('Q',8),11:('q',8),12:('d',8)}

def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')

def read_value(f, t):
    if t in _FMT:
        fmt, sz = _FMT[t]
        return struct.unpack('<'+fmt, f.read(sz))[0]
    if t == 8:  # STRING
        return read_str(f)
    if t == 9:  # ARRAY
        et = struct.unpack('<I', f.read(4))[0]
        n  = struct.unpack('<Q', f.read(8))[0]
        return [read_value(f, et) for _ in range(n)]
    raise ValueError(f"unknown type {t}")

def main(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            return 1
        ver, tc, kvc = struct.unpack('<IQQ', f.read(20))
        ctx = None
        for _ in range(kvc):
            k = read_str(f)
            t = struct.unpack('<I', f.read(4))[0]
            v = read_value(f, t)
            if k.endswith('.context_length') and isinstance(v, int):
                ctx = v
                # could break early but also benign to continue
                break
        if ctx is None:
            return 1
        print(ctx)
        return 0

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
