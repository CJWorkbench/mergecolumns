import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from mergecolumns import migrate_params, render


def P(columns=[], delimiter='-', newcolumn='New column'):
    return {
        'columns': columns,
        'delimiter': delimiter,
        'newcolumn': newcolumn,
    }


class MigrateParamsTest(unittest.TestCase):
    def test_v0_to_v1(self):
        self.assertEqual(
            migrate_params({
                'firstcolumn': 'A',
                'secondcolumn': 'B',
                'delimiter': '-',
                'newcolumn': 'C',
            }),
            {
                'columns': ['A', 'B'],
                'delimiter': '-',
                'newcolumn': 'C',
            },
        )

    def test_v0_to_v1_ignore_empty_colname(self):
        self.assertEqual(
            migrate_params({
                'firstcolumn': 'A',
                'secondcolumn': '',
                'delimiter': '-',
                'newcolumn': 'C',
            }),
            {
                'columns': ['A'],
                'delimiter': '-',
                'newcolumn': 'C',
            },
        )

    def test_v1_to_v1(self):
        self.assertEqual(
            migrate_params({
                'columns': ['A', 'B'],
                'delimiter': '-',
                'newcolumn': 'C',
            }),
            {
                'columns': ['A', 'B'],
                'delimiter': '-',
                'newcolumn': 'C',
            },
        )


class TestMergeColumns(unittest.TestCase):
    def test_defaults_no_op(self):
        result = render(
            pd.DataFrame({'A': ['a', 'b'], 'B': ['b', 'c']}),
            P()
        )
        expected = pd.DataFrame({'A': ['a', 'b'], 'B': ['b', 'c']})
        assert_frame_equal(result, expected)

    def test_one_column_build_empty_column(self):
        result = render(
            pd.DataFrame({'A': ['a', 'b'], 'B': ['b', 'c']}),
            P(columns=['A'], newcolumn='C'),
        )
        expected = pd.DataFrame({'A': ['a', 'b'], 'B': ['b', 'c'], 'C': ['a', 'b']})
        assert_frame_equal(result, expected)

    def test_no_newcolumn_no_op(self):
        result = render(
            pd.DataFrame({'A': ['a', 'b'], 'B': ['b', 'c']}),
            P(columns=['A', 'B'], newcolumn=''),
        )
        expected = pd.DataFrame({'A': ['a', 'b'], 'B': ['b', 'c']})
        assert_frame_equal(result, expected)

    def test_str(self):
        result = render(
            pd.DataFrame({'A': ['a', 'b'], 'B': ['c', 'd']}),
            P(['A', 'B'], '-', 'C')
        )
        expected = pd.DataFrame({
            'A': ['a', 'b'],
            'B': ['c', 'd'],
            'C': ['a-c', 'b-d'],
        })
        assert_frame_equal(result, expected)

    def test_overwrite_newcolumn(self):
        result = render(
            pd.DataFrame({'A': ['a', 'b'], 'B': ['c', 'd']}),
            P(['A', 'B'], '-', 'A')
        )
        expected = pd.DataFrame({
            'A': ['a-c', 'b-d'],
            'B': ['c', 'd'],
        })
        assert_frame_equal(result, expected)

    def test_str_ignore_null(self):
        result = render(
            # Test for silly tricks involving str(np.nan)....
            pd.DataFrame({
                'A': [np.nan, np.nan, '3', '4', 'nan', '6'],
                'B': ['1', np.nan, '3', 'nan', '5', np.nan],
            }),
            P(['A', 'B'], '-', 'C')
        )
        expected = pd.DataFrame({
            'A': [np.nan, np.nan, '3', '4', 'nan', '6'],
            'B': ['1', np.nan, '3', 'nan', '5', np.nan],
            'C': ['1', np.nan, '3-3', '4-nan', 'nan-5', '6'],
        })
        assert_frame_equal(result, expected)

    def test_categorical(self):
        result = render(
            pd.DataFrame({'A': ['a', 'b'], 'B': ['c', 'd']}, dtype='category'),
            P(['A', 'B'], '-', 'A')
        )
        # Let's just convert to str, for now
        expected = pd.DataFrame({
            'A': ['a-c', 'b-d'],
            'B': pd.Series(['c', 'd'], dtype='category'),
        })
        assert_frame_equal(result, expected)

    def test_categorical_ignore_nan(self):
        result = render(
            # Test for silly tricks involving str(np.nan)....
            pd.DataFrame({
                'A': [np.nan, '2'],
                'B': ['1', '3'],
            }, dtype='category'),
            P(['A', 'B'], '-', 'C')
        )
        expected = pd.DataFrame({
            'A': pd.Series([np.nan, '2'], dtype='category'),
            'B': pd.Series(['1', '3'], dtype='category'),
            'C': ['1', '2-3'],
        })
        assert_frame_equal(result, expected)

    def test_mix_str_and_categorical(self):
        result = render(
            pd.DataFrame({
                'A': ['a', 'b'],
                'B': pd.Series(['c', 'd'], dtype='category'),
            }),
            P(['A', 'B'], '-', 'A')
        )
        # Let's just convert to str, for now
        expected = pd.DataFrame({
            'A': ['a-c', 'b-d'],
            'B': pd.Series(['c', 'd'], dtype='category'),
        })
        assert_frame_equal(result, expected)

    def test_many_columns(self):
        result = render(
            pd.DataFrame({
                'A': ['a', 'b'],
                'B': ['c', 'd'],
                'C': ['e', 'f'],
                'D': [np.nan, 'h'],
            }),
            P(['A', 'B', 'C', 'D'], '+', 'E'),
        )
        expected = pd.DataFrame({
            'A': ['a', 'b'],
            'B': ['c', 'd'],
            'C': ['e', 'f'],
            'D': [np.nan, 'h'],
            'E': ['a+c+e', 'b+d+f+h'],
        })
        assert_frame_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
