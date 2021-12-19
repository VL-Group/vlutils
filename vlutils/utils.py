import re
import yaml
from io import StringIO


def _alignYAML(str, pad=0, aligned_colons=False):
    props = re.findall(r'^\s*[\S]+:', str, re.MULTILINE)
    if not props:
        return str
    longest = max([len(i) for i in props]) + pad
    if aligned_colons:
        return ''.join([i+'\n' for i in map(
                    lambda str: re.sub(r'^(\s*.+?[^:#]): \s*(.*)',
                        lambda m: m.group(1) + ''.ljust(longest-len(m.group(1))-1-pad) + ':'.ljust(pad+1) + m.group(2), str, re.MULTILINE),
                    str.split('\n'))])
    else:
        return ''.join([i+'\n' for i in map(
                    lambda str: re.sub(r'^(\s*.+?[^:#]: )\s*(.*)',
                        lambda m: m.group(1) + ''.ljust(longest-len(m.group(1))+1) + m.group(2), str, re.MULTILINE),
                    str.split('\n'))])

def pPrint(d: dict) -> str:
    """Print dict prettier.

    Args:
        d (dict): The input dict.

    Returns:
        str: Resulting string.
    """
    with StringIO() as stream:
        yaml.safe_dump(d, stream, default_flow_style=False)
        return _alignYAML(stream.getvalue(), pad=1, aligned_colons=True)
