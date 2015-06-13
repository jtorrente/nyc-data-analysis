__author__ = 'jtorrente'

import unittest
import nycsubway.module1.simple_log as sl

class TestSimpleLog(unittest.TestCase):
    """
    Simple test for a simple log
    """
    def test_simple_log(self):
        simple_log = sl.SimpleLog(2)
        self.assertEqual(simple_log.current_step, 0)
        self.assertEqual(simple_log.total_steps, 2)
        msg = "TESTING\nSecond line \nA very looooooooooooooooooooooooooooooooooooooooooooong line\n"
        simple_log.log(msg)
        self.assertEqual(simple_log.current_step, 1)
        self.assertEqual(simple_log.total_steps, 2)
        max_chars, all_lines = simple_log.split_in_lines(msg)
        self.assertEqual(max_chars, 60)
        self.assertEqual(len(all_lines), 3)


if __name__ == '__main__':
    unittest.main()
