import unittest
import pandas as pd
import numpy as np
from mergecolumns import render


class TestMergeColumns(unittest.TestCase):

    def setUp(self):
        # Test data includes:
        #  - rows of numeric, string, and categorical types
        #  - zero entries (which should not be removed)
        #  - if either column categorical type, retain caegorical
        self.table = pd.DataFrame([
            ['a',    1,  2,   'b',    '', 1],
            ['a',   1,  None,   'b',  'b',    None],
            ['a',    1,    None,   '', 'c',    2],
            ['a',    1, 2, '', 'd', None],
            ['a',  1,  2,  'b',    'e',    3]],
            columns=['stringcol1','intcol','floatcol','stringcol2','catcol','floatcatcol'])

        # Pandas should infer these types anyway, but leave nothing to chance
        self.table['stringcol1'] = self.table['stringcol1'].astype(str)
        self.table['intcol'] = self.table['intcol'].astype(np.int64)
        self.table['floatcol'] = self.table['floatcol'].astype(np.float64)
        self.table['stringcol2'] = self.table['stringcol2'].astype(str)
        self.table['catcol'] = self.table['catcol'].astype('category')
        self.table['floatcatcol'] = self.table['floatcatcol'].astype('category')

    def test_NOP(self):
        params = { 'firstcolumn': '', 'secondcolumn': 'intcol', 'newcolumn': 'intcol2', 'delimiter': '-'}
        out = render(self.table, params)
        self.assertTrue(out.equals(self.table)) # should NOP when first applied

        params = { 'firstcolumn': 'intcol', 'secondcolumn': '', 'newcolumn': 'intcol2', 'delimiter': '-'}
        out = render(self.table, params)
        self.assertTrue(out.equals(self.table))  # should NOP when first applied

        params = { 'firstcolumn': 'catcol', 'secondcolumn': 'intcol', 'newcolumn': '', 'delimiter': '-'}
        out = render(self.table, params)
        self.assertTrue(out.equals(self.table))  # should NOP when first applied

    def test_string(self):
        params = {'firstcolumn': 'stringcol1', 'secondcolumn': 'stringcol2', 'newcolumn': 'newcol', 'delimiter': '-'}
        out = render(self.table, params)
        ref = self.table
        ref['newcol'] = pd.Series(['a-b', 'a-b', 'a-', 'a-', 'a-b'])
        self.assertTrue(out.equals(ref))

    def test_cat(self):
        params = {'firstcolumn': 'stringcol1', 'secondcolumn': 'catcol', 'newcolumn': 'newcol', 'delimiter': '-'}
        out = render(self.table, params)
        ref = self.table
        ref['newcol'] = pd.Series(['a-', 'a-b', 'a-c', 'a-d', 'a-e'])
        self.assertTrue(out.equals(ref))

    def test_num(self):
        params = {'firstcolumn': 'intcol', 'secondcolumn': 'floatcol', 'newcolumn': 'newcol', 'delimiter': '-'}
        out = render(self.table, params)
        ref = self.table
        ref['newcol'] = pd.Series(['1-2', '1-', '1-', '1-2', '1-2'])
        self.assertTrue(out.equals(ref))

    def test_col_overwrite(self):
        params = {'firstcolumn': 'stringcol1', 'secondcolumn': 'stringcol2', 'newcolumn': 'stringcol2', 'delimiter': '-'}
        out = render(self.table, params)
        ref = self.table
        ref['stringcol2'] = pd.Series(['a-b', 'a-b', 'a-', 'a-', 'a-b'])
        self.assertTrue(out.equals(ref))

if __name__ == '__main__':
    unittest.main()


