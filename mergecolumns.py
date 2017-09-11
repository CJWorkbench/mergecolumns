class Importable:
    @staticmethod
    def __init__(self):
        pass

    @staticmethod
    def event():
        pass

    @staticmethod
    def render(wf_module, table):
        firstcol = wf_module.get_param_column('firstcolumn')
        secondcol = wf_module.get_param_column('secondcolumn')
        delimiter = wf_module.get_param_string('delimiter')
        newcol = wf_module.get_param_string('newcolumn')

        if firstcol == '' or secondcol == '':
            return table

        wf_module.set_ready(notify=False)

        table[newcol] = table[[firstcol, secondcol]].apply(lambda x: delimiter.join([str(i) for i in x if str(i) != 'nan']), axis=1)
        return table