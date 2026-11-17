#!/usr/bin/env python3
"""Isolate device symbols before linking."""
import argparse
import json
from pathlib import Path
import struct
import subprocess

def weak_definitions(path):
    data = Path(path).read_bytes()
    if data[:4] != b'\x7fELF' or data[4] != 2 or data[5] not in (1, 2):
        raise ValueError(f'{path}: expected a 64-bit ELF object')
    endian = '<' if data[5] == 1 else '>'
    shoff = struct.unpack_from(endian + 'Q', data, 40)[0]
    shsize, shnum = struct.unpack_from(endian + 'HH', data, 58)
    sections = [struct.unpack_from(endian + 'IIQQQQIIQQ', data, shoff + i * shsize) for i in range(shnum)]
    names = set()
    # Collect defined weak symbols.
    for section in sections:
        if section[1] != 2:
            continue
        strings = sections[section[6]]
        strtab = data[strings[4]:strings[4] + strings[5]]
        if section[9] < 24:
            raise ValueError(f'{path}: invalid symbol table entry size')
        for offset in range(section[4], section[4] + section[5], section[9]):
            name, info, _, index, _, _ = struct.unpack_from(endian + 'IBBHQQ', data, offset)
            if info >> 4 != 2 or index == 0:
                continue
            end = strtab.find(b'\0', name)
            symbol = strtab[name:end].decode('utf-8')
            if symbol:
                names.add(symbol)
    return sorted(names)

def isolate(path, objcopy):
    path = Path(path)
    symbols = weak_definitions(path)
    if not symbols:
        print(f'[HardFuzz] {path}: no weak definitions to isolate')
        return
    listing = path.with_suffix(path.suffix + '.hardfuzz_symbols.txt')
    listing.write_text('\n'.join(symbols) + '\n')
    subprocess.run([str(objcopy), '--localize-symbols=' + str(listing), str(path)], check=True)
    path.with_suffix(path.suffix + '.hardfuzz_symbols.json').write_text(json.dumps({'object': str(path), 'localized_symbols': symbols}, indent=2) + '\n')
    print(f'[HardFuzz] {path}: localized {len(symbols)} weak definitions')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objcopy', required=True, type=Path)
    parser.add_argument('objects', nargs='+', type=Path)
    args = parser.parse_args()
    for path in args.objects:
        isolate(path, args.objcopy)

if __name__ == '__main__':
    main()
