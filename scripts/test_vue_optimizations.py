"""Functional checks for record-index reuse, MDB2 streaming and progress."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from vue_business import load_business
from vue_mdb_export import read_business_database, read_database


class GeometryOptimizationTests(unittest.TestCase):
    def test_record_index_reuse(self):
        from vue_record_parser import build_moniker_indexes, read_records
        data = Path('063E.vue').read_bytes()
        indexes = build_moniker_indexes(data)
        expected = read_records(data)
        with patch('vue_record_parser.build_moniker_indexes', side_effect=AssertionError('rebuilt')):
            self.assertEqual(read_records(data, indexes=indexes), expected)


def business_tables():
    return dict(
        linkage=[dict(linkage_index=1, moniker='A', file_name='Piping'),
                 dict(linkage_index=2, moniker='A', file_name='Piping')],
        label_names=[dict(label_name_index=i, label_name=name) for i, name in enumerate(
            ['Name', 'Quality Index', 'Enabled', 'Empty', 'Numeric', 'Name'])],
        label_values=[dict(label_value_index=i, label_value=value, label_value_numeric=number)
                      for i, (value, number) in enumerate([('One',0),('Two',0),(0,0),(False,0),('',0),('',12)])],
        labels=[dict(linkage_index=link, label_line_number=line, label_name_index=name,
                     label_value_index=value, extended_label=extended)
                for line, (link,name,value,extended) in enumerate([
                    (1,0,0,False),(1,1,2,False),(1,2,3,False),(1,3,4,False),
                    (1,4,5,'detail'),(1,0,1,False),(1,5,0,False),(2,0,1,False)])])


class BusinessStreamingTests(unittest.TestCase):
    def test_streamed_properties_match_csv_and_preserve_special_values(self):
        tables = business_tables()
        with patch('vue_mdb_export.read_database_tables', return_value=(tables, {})):
            _, wide, long, columns, report = read_database(Path('unused'))
            stream, actual_columns, actual_report = read_business_database(Path('unused'))
            streamed = list(stream)
            self.assertEqual([row for row, _ in streamed], wide)
            self.assertEqual(actual_columns, columns)
            self.assertEqual(actual_report, report)
            self.assertEqual(sum(len(details) for _, details in streamed), 1)
            self.assertEqual(len(long), 8)
            data = load_business(Path('unused'), [], {}, 'test')
        props = data['objects']['A']['properties']
        self.assertEqual(props['Name'], ['One', 'Two'])
        self.assertEqual(props['Name__2'], 'Two')
        self.assertEqual(props['Name [5]'], 'One')
        self.assertEqual(props['Quality Index'], 0)
        self.assertIs(props['Enabled'], False)
        self.assertNotIn('Empty', props)
        self.assertEqual(data['objects']['A']['label_details'], [{'numeric':12, 'extended':'detail', 'label':'Numeric'}])

    def test_broken_references_and_duplicate_primary_keys_still_fail(self):
        for table, key in [('linkage','linkage_index'), ('label_names','label_name_index'),
                           ('label_values','label_value_index')]:
            tables = copy.deepcopy(business_tables())
            tables[table].append(dict(tables[table][0]))
            with self.subTest(table=table), patch('vue_mdb_export.read_database_tables', return_value=(tables, {})):
                with self.assertRaises(ValueError):
                    load_business(Path('unused'), [], {}, 'test')
        for key in ('linkage_index', 'label_name_index', 'label_value_index'):
            tables = copy.deepcopy(business_tables())
            tables['labels'][0][key] = 999
            with self.subTest(key=key), patch('vue_mdb_export.read_database_tables', return_value=(tables, {})):
                with self.assertRaises(ValueError):
                    load_business(Path('unused'), [], {}, 'test')


class ProgressTests(unittest.TestCase):
    def test_progress_is_throttled_and_always_reports_completion(self):
        from vue_progress import ExportProgress
        messages = []
        with patch('vue_progress.time.perf_counter', return_value=10.) as clock:
            progress = ExportProgress(messages.append, 'sample')
            progress.count('Geometry', 0, 10)
            progress.count('Geometry', 1, 10)
            self.assertEqual(len(messages), 1)
            clock.return_value = 12.1
            progress.count('Geometry', 2, 10)
            progress.count('Geometry', 10, 10)
        self.assertEqual(len(messages), 3)
        self.assertIn('sample +2.1s', messages[-1])
        self.assertIn('10/10 (100%)', messages[-1])

    def test_console_updates_are_flushed(self):
        from vue_progress import console_progress
        with patch('builtins.print') as output:
            console_progress('test')
        output.assert_called_once_with('test', flush=True)


if __name__ == '__main__':
    unittest.main()
