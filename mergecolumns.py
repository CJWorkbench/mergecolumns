from functools import partial, reduce
import numpy as np
import pandas as pd


def merge2(delimiter: str, s1: pd.Series, s2: pd.Series) -> pd.Series:
    """Elementwise return `{s1}{delimiter}{s2}`.

    Omit nulls.
    """
    na1 = s1.isnull()
    na2 = s2.isnull()

    # Convert to str, always
    if hasattr(s1, 'cat'):
        s1 = s1.astype(str)
        s1[na1] = np.nan
    if hasattr(s2, 'cat'):
        s2 = s2.astype(str)
        s2[na2] = np.nan

    result = s1 + delimiter + s2  # invalid when na1 | na2
    result[na1] = s2
    result[na2] = s1
    # Now results[na1 | na2] are valid
    return result


def render(table, params):
    if table is None:
        return None
    
    colnames = params['columns']
    delimiter = params['delimiter']
    newcolname = params['newcolumn']

    if not colnames or not newcolname:
        return table

    columns = [table[c] for c in colnames]
    result = reduce(partial(merge2, delimiter), columns)

    table[newcolname] = result
    return table


def _migrate_params_v0_to_v1(params):
    return dict(
        columns=[n for n in (params['firstcolumn'], params['secondcolumn']) if n],
        delimiter=params['delimiter'],
        newcolumn=params['newcolumn'],
    )

def migrate_params(params):
    if 'firstcolumn' in params:
        params = _migrate_params_v0_to_v1(params)
    return params
