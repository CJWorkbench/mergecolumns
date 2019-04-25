import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from mergecolumns import render


def P(firstcolumn=None, secondcolumn=None, delimiter='-', newcolumn='New column'):
    return {
        'firstcolumn': firstcolumn,
        'secondcolumn': secondcolumn,
        'delimiter': delimiter,
        'newcolumn': newcolumn,
    }


class TestMergeColumns(unittest.TestCase):
    def test_defaults_no_op(self):
        result = render(
            pd.DataFrame({'A': ['a', 'b'], 'B': ['b', 'c']}),
            P()
        )
        expected = pd.DataFrame({'A': ['a', 'b'], 'B': ['b', 'c']})
        assert_frame_equal(result, expected)

    def test_no_firstcolumn_no_op(self):
        result = render(
            pd.DataFrame({'A': ['a', 'b'], 'B': ['b', 'c']}),
            P(secondcolumn='B', newcolumn='C')
        )
        expected = pd.DataFrame({'A': ['a', 'b'], 'B': ['b', 'c']})
        assert_frame_equal(result, expected)

    def test_no_secondcolumn_no_op(self):
        result = render(
            pd.DataFrame({'A': ['a', 'b'], 'B': ['b', 'c']}),
            P(firstcolumn='A', newcolumn='C')
        )
        expected = pd.DataFrame({'A': ['a', 'b'], 'B': ['b', 'c']})
        assert_frame_equal(result, expected)

    def test_no_newcolumn_no_op(self):
        result = render(
            pd.DataFrame({'A': ['a', 'b'], 'B': ['b', 'c']}),
            P(firstcolumn='A', secondcolumn='B', newcolumn=None)
        )
        expected = pd.DataFrame({'A': ['a', 'b'], 'B': ['b', 'c']})
        assert_frame_equal(result, expected)

    def test_str(self):
        result = render(
            pd.DataFrame({'A': ['a', 'b'], 'B': ['c', 'd']}),
            P('A', 'B', '-', 'C')
        )
        expected = pd.DataFrame({
            'A': ['a', 'b'],
            'B': ['c', 'd'],
            'C': ['a-c', 'b-d'],
        })
        assert_frame_equal(result, expected)

    def test_ignore_table_ordering(self):
        result = render(
            pd.DataFrame({'A': ['a', 'b'], 'B': ['c', 'd']}),
            P('B', 'A', '-', 'C')
        )
        expected = pd.DataFrame({
            'A': ['a', 'b'],
            'B': ['c', 'd'],
            'C': ['c-a', 'd-b'],
        })
        assert_frame_equal(result, expected)

    def test_overwrite_newcolumn(self):
        result = render(
            pd.DataFrame({'A': ['a', 'b'], 'B': ['c', 'd']}),
            P('A', 'B', '-', 'A')
        )
        expected = pd.DataFrame({
            'A': ['a-c', 'b-d'],
            'B': ['c', 'd'],
        })
        assert_frame_equal(result, expected)

    def test_str_ignore_nan(self):
        result = render(
            # Test for silly tricks involving str(np.nan)....
            pd.DataFrame({
                'A': [np.nan, np.nan, '3', '4', 'nan'],
                'B': ['1', np.nan, '3', 'nan', '5'],
            }),
            P('A', 'B', '-', 'C')
        )
        expected = pd.DataFrame({
            'A': [np.nan, np.nan, '3', '4', 'nan'],
            'B': ['1', np.nan, '3', 'nan', '5'],
            'C': ['1', np.nan, '3-3', '4-nan', 'nan-5'],
        })
        assert_frame_equal(result, expected)

    def test_categorical(self):
        result = render(
            pd.DataFrame({'A': ['a', 'b'], 'B': ['c', 'd']}, dtype='category'),
            P('A', 'B', '-', 'A')
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
            P('A', 'B', '-', 'C')
        )
        expected = pd.DataFrame({
            'A': pd.Series([np.nan, '2'], dtype='category'),
            'B': pd.Series(['1', '3'], dtype='category'),
            'C': ['1', '2-3'],
        })
        assert_frame_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
