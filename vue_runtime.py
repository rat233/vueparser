"""Launch isolated FBX helpers from Python or a frozen executable."""
from pathlib import Path
import sys


def helper_command(module, *arguments):
    if getattr(sys, 'frozen', False):
        return [sys.executable, '--internal-helper', module, *map(str, arguments)]
    return [sys.executable, str(Path(__file__).with_name(module+'.py')), *map(str, arguments)]
