import sys


def str_join(str_, iterable, precision=None):
    if precision is None:
        list_ = [str(item) for item in iterable] 
    else:
        list_ = [f'{item:.2f}' for item in iterable] 
    return str_.join(list_)


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
            raise ValueError('unindenting to negative ident')

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
        if self.file is not None:
            self.file.write(to_write)
            self.file.flush()
        if self.stdout:
            sys.stdout.write(to_write)
            sys.stdout.flush()
        if to_write.endswith('\n'):
            self._newline = True

    def command(self, command, *args, options_first=True, **kwargs):
        self.write(f'\\{command}', end='')
        if not options_first:
            for arg in args:
                self.write(f'{{{arg}}}', end='')
        if kwargs:
            self.write('[', end='')
            to_write = []
            for key, item in kwargs.items():
                if item is None:
                    to_write.append(key)
                else:
                    to_write.append(f'{key}={item}')
            self.write(', '.join(to_write), end='')
            self.write(']', end='')
        if options_first:
            for arg in args:
                self.write(f'{{{arg}}}', end='')
        self.write('')

    def begin(self, env, *args, indent=True, **kwargs):
        self.command('begin', env, *args, options_first=False, **kwargs)
        self._env_struct.append(env)
        if indent:
            self.indent()

    def end(self, env, *args, unindent=True, **kwargs):
        if unindent:
            self.unindent()
        self.command('end', env, *args, options_first=False, **kwargs)
        current_env = self._env_struct.pop()
        if current_env != env:
            raise ValueError('ending an environment that has not begun')


def df_to_tex(df, writer=None, precision=2, loc=None, label='my_label',
              caption='Caption'):
    if writer is None:
        writer = TexWriter()
    writer.begin('table', h=None)
    writer.command('centering')
    table_spec = '|'.join(['c' for i in range(df.shape[1] + 1)])
    writer.begin('tabular', table_spec)
    writer.write(df.index.name, end=' & ')
    writer.write(str_join(' & ', df.columns), end=' \\\\\n')
    writer.command('hline')
    for index, data in df.iterrows():
        writer.write(index, end=' & ')
        writer.write(str_join(' & ', data, precision), end=' \\\\\n')
    writer.end('tabular')
    writer.command('caption', caption)
    writer.command('label', f'tab:{label}')
    writer.end('table')
