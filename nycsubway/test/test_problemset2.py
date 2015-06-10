__author__ = 'jtorrente'

import unittest

import nycsubway.module1.problemsets.problemset2 as ps2


class MyTestCase(unittest.TestCase):
    def test_count_rainy_days(self):
        count_rainy_days = ps2.count_rainy_days(r"../../data/weather_underground.csv")
        self.assertEqual(count_rainy_days, 10, "count_rainy_days is not returning expected value")



if __name__ == '__main__':
    unittest.main()
