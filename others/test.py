import unittest

class BIT:
    def __init__(self, N):
        self.N = N
        self.bit = [0] * (N + 1)

    def add(self, i, x):
        idx = i + 1
        while idx <= self.N:
            self.bit[idx] += x
            idx += idx & -idx

    def sum(self, i):
        idx = i + 1
        result = 0
        while idx > 0:
            result += self.bit[idx]
            idx -= idx & -idx
        return result

class TestBIT(unittest.TestCase):
    def test_single_add(self):
        bit = BIT(10)
        bit.add(0, 5)
        self.assertEqual(bit.sum(0), 5)

    def test_multiple_adds(self):
        bit = BIT(10)
        bit.add(0, 5)
        bit.add(1, 3)
        bit.add(2, 7)
        self.assertEqual(bit.sum(0), 5)
        self.assertEqual(bit.sum(1), 8)
        self.assertEqual(bit.sum(2), 15)

    def test_boundary(self):
        bit = BIT(10)
        bit.add(9, 5)  # 最後の要素に加算
        self.assertEqual(bit.sum(9), 5)
        self.assertEqual(bit.sum(8), 0)

    def test_large_updates(self):
        bit = BIT(100)
        bit.add(99, 1)
        self.assertEqual(bit.sum(99), 1)
        bit.add(0, 10)
        self.assertEqual(bit.sum(99), 11)  # 全体の和

if __name__ == '__main__':
    unittest.main()
