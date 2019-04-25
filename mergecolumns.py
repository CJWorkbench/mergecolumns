import numpy as np


def render(table, params):
    if table is None:
        return None
    
    firstcol = params['firstcolumn']
    secondcol = params['secondcolumn']
    delimiter = params['delimiter']
    newcol = params['newcolumn']

    if not firstcol or not secondcol or not newcol:
        return table

    col1 = table[firstcol]
    col2 = table[secondcol]
    na1 = col1.isnull()
    na2 = col2.isnull()

    if hasattr(col1, 'cat'):
        col1 = col1.astype(str)
        col1[na1] = np.nan
    if hasattr(col2, 'cat'):
        col2 = col2.astype(str)
        col2[na2] = np.nan

    result = col1 + delimiter  # np.nan in col1 will stay na
    result[na1 & ~na2] = '' # so we can add col2 when it is not nan
    result[~na2] += col2

    table[newcol] = result
    return table
