from typing import Any
import re
import yaml
from io import StringIO


"""https://github.com/click-contrib/click-default-group/blob/master/click_default_group.py
   click_default_group
   ~~~~~~~~~~~~~~~~~~~
   Define a default subcommand by `default=True`:
   .. sourcecode:: python
      import click
      from click_default_group import DefaultGroup
      @click.group(cls=DefaultGroup, default_if_no_args=True)
      def cli():
          pass
      @cli.command(default=True)
      def foo():
          click.echo('foo')
      @cli.command()
      def bar():
          click.echo('bar')
   Then you can invoke that without explicit subcommand name:
   .. sourcecode:: console
      $ cli.py --help
      Usage: cli.py [OPTIONS] COMMAND [ARGS]...
      Options:
        --help    Show this message and exit.
      Command:
        foo*
        bar
      $ cli.py
      foo
      $ cli.py foo
      foo
      $ cli.py bar
      bar
"""
import warnings

import click


class DefaultGroup(click.Group):
    """Invokes a subcommand marked with `default=True` if any subcommand not
    chosen.
    :param default_if_no_args: resolves to the default command if no arguments
                               passed.
    """

    def __init__(self, *args, **kwargs):
        # To resolve as the default command.
        if not kwargs.get('ignore_unknown_options', True):
            raise ValueError('Default group accepts unknown options')
        self.ignore_unknown_options = True
        self.default_cmd_name = kwargs.pop('default', None)
        self.default_if_no_args = kwargs.pop('default_if_no_args', False)
        super(DefaultGroup, self).__init__(*args, **kwargs)

    def set_default_command(self, command):
        """Sets a command function as the default command."""
        cmd_name = command.name
        self.add_command(command)
        self.default_cmd_name = cmd_name

    def parse_args(self, ctx, args):
        if not args and self.default_if_no_args:
            args.insert(0, self.default_cmd_name)
        return super(DefaultGroup, self).parse_args(ctx, args)

    def get_command(self, ctx, cmd_name):
        if cmd_name not in self.commands:
            # No command name matched.
            ctx.arg0 = cmd_name
            cmd_name = self.default_cmd_name
        return super(DefaultGroup, self).get_command(ctx, cmd_name)

    def resolve_command(self, ctx, args):
        base = super(DefaultGroup, self)
        cmd_name, cmd, args = base.resolve_command(ctx, args)
        if hasattr(ctx, 'arg0'):
            args.insert(0, ctx.arg0)
            cmd_name = cmd.name
        return cmd_name, cmd, args

    def format_commands(self, ctx, formatter):
        formatter = DefaultCommandFormatter(self, formatter, mark='*')
        return super(DefaultGroup, self).format_commands(ctx, formatter)

    def command(self, *args, **kwargs):
        default = kwargs.pop('default', False)
        decorator = super(DefaultGroup, self).command(*args, **kwargs)
        if not default:
            return decorator
        warnings.warn('Use default param of DefaultGroup or '
                      'set_default_command() instead', DeprecationWarning)

        def _decorator(f):
            cmd = decorator(f)
            self.set_default_command(cmd)
            return cmd

        return _decorator


class DefaultCommandFormatter(object):
    """Wraps a formatter to mark a default command."""

    def __init__(self, group, formatter, mark='*'):
        self.group = group
        self.formatter = formatter
        self.mark = mark

    def __getattr__(self, attr):
        return getattr(self.formatter, attr)

    def write_dl(self, rows, *args, **kwargs):
        rows_ = []
        for cmd_name, help in rows:
            if cmd_name == self.group.default_cmd_name:
                rows_.insert(0, (cmd_name + self.mark, help))
            else:
                rows_.append((cmd_name, help))
        return self.formatter.write_dl(rows_, *args, **kwargs)


def strNoneorEmpty(string: Any):
    return string is None or not (string or string.strip())

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
