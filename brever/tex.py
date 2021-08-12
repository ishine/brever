import sys

import pandas as pd


class TexWriter:
    def __init__(self, filename=None, stdout=True, end='\n', indent=4):
        if filename is None:
            self.file = None
        else:
            self.file = open(filename, 'w')
        self.stdout = stdout
        self.end_ = end
        self.indent_size = indent
        self._indent = 0
        self._env_struct = []
        self._newline = True

    def indent(self):
        self._indent += self.indent_size

    def unindent(self):
        self._indent -= self.indent_size
        if self._indent < 0:
            raise ValueError('unindenting to negative indent')

    def write(self, data, end=None):
        if data is None:
            data = ''
        if not isinstance(data, str):
            data = str(data)
        if end is None:
            end = self.end_
        to_write = data + end
        if self._newline:
            to_write = ' '*self._indent + to_write
            self._newline = False
        if to_write.endswith('\n'):
            self._newline = True
        if self.file is not None:
            self.file.write(to_write)
            self.file.flush()
        if self.stdout:
            sys.stdout.write(to_write)
            sys.stdout.flush()

    def command(self, command, *args, options=[]):
        self.write(f'\\{command}', end='')
        for option in options:
            self.write(f'[{option}]', end='')
        for arg in args:
            self.write(f'{{{arg}}}', end='')
        self.write('')

    def begin(self, env, *args, indent=True, options=[]):
        self.write(f'\\begin{{{env}}}', end='')
        for option in options:
            self.write(f'[{option}]', end='')
        for arg in args:
            self.write(f'{{{arg}}}', end='')
        self.write('')
        self._env_struct.append(env)
        if indent:
            self.indent()

    def end(self, env, *args, unindent=True):
        if unindent:
            self.unindent()
        self.write(f'\\end{{{env}}}')
        current_env = self._env_struct.pop()
        if current_env != env:
            raise ValueError('closing an environment that is not the last '
                             'opened environment')


def df_to_tex(df, writer=None, filename=None, precision=2, loc=None,
              label='my_label', caption='Caption', star=False, bold=[]):
    if writer is None:
        writer = TexWriter(filename)
    if star:
        table = 'table*'
    else:
        table = 'table'
    writer.begin(table, options=['h'])
    writer.command('centering')
    table_spec = '|'.join(['c' for i in range(df.shape[1] + 1)])
    writer.begin('tabular', table_spec)
    writer.write(df.index.name, end=' & ')
    writer.write(' & '.join([str(c) for c in df.columns]), end=' \\\\\n')
    writer.command('hline')
    for i, (index, data) in enumerate(df.iterrows()):
        writer.write(index, end=' & ')
        data = [f'{x:.{precision}f}' for x in data]
        for j in range(len(data)):
            if (i, j) in bold:
                data[j] = f'\\textbf{{{data[j]}}}'
        writer.write(' & '.join(data), end=' \\\\\n')
    writer.end('tabular')
    writer.command('caption', caption)
    writer.command('label', label)
    writer.end(table)


def np_to_tex(array, writer=None, filename=None, precision=2, loc=None,
              label='my_label', caption='Caption', columns=None, index=None,
              index_name=None, star=False, bold=[]):
    if index is None:
        index = pd.RangeIndex(0, len(array), name=index_name)
    else:
        index = pd.Index(index, name=index_name)
    df = pd.DataFrame(array, index, columns)
    df_to_tex(df, writer=writer, filename=filename, precision=precision,
              loc=loc, label=label, caption=caption, star=star, bold=bold)
