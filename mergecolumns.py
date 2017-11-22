def render(table, params):
    if table is None:
        return None
    
    firstcol = params['firstcolumn']
    secondcol = params['secondcolumn']
    delimiter = params['delimiter']
    newcol = params['newcolumn']

    if firstcol == '' or secondcol == '' or newcol == '':
        return table

    table[newcol] = table[[firstcol, secondcol]].apply(lambda x: delimiter.join([str(i) for i in x if str(i) != 'nan']), axis=1)
    return table
