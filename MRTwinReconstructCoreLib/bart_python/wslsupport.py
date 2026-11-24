# Vendored from the BART Python bindings (python/bart.py, cfl.py, wslsupport.py of
# https://github.com/mrirecon/bart). BSD 3-Clause licence; see LICENSE in this
# directory and the copyright header of each file. Local changes: bart.py renamed
# to bart_python.py, package-relative imports, WSL path conversion helpers.

import os
import string

def WindowsToWSLPath(inData):
    """
    Convert Windows-style path to WSL-style path.
    Example: C:\\Users\\me\\file.txt → /mnt/c/Users/me/file.txt
    """
    outData = inData
    for i in string.ascii_lowercase:
        outData = outData.replace(i + ':', '/mnt/' + i)
        outData = outData.replace(i.upper() + ':', '/mnt/' + i)
    outData = outData.replace(os.path.sep, '/')
    return outData


def WSLToWindowsPath(wsl_path):
    """
    Convert WSL-style path to Windows-style path.
    Example: /mnt/c/Users/me/file.txt → C:\\Users\\me\\file.txt
    """
    if wsl_path.startswith("/mnt/") and len(wsl_path) > 6:
        drive_letter = wsl_path[5].upper()
        rest = wsl_path[6:].replace('/', '\\')  # convert slashes first
        windows_path = f"{drive_letter}:{os.path.sep}{rest}"
        return windows_path
    else:
        return wsl_path
