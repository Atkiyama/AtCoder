# from functools import lru_cache
# import sys
# sys.setrecursionlimit(10**9)
# input = sys.stdin.readline
# from decimal import Decimal
# from functools import cmp_to_key
# from collections import Counter
# from itertools import permutations
# from itertools import combinations
# from itertools import combinations_with_replacement
# from itertools import product
# from itertools import accumulate
# from itertools import groupby
# from itertools import pairwise
# from copy import deepcopy
# import networkx as nx
# import networkx.algorithms as nxa
# import numpy as np
# import math
# import heapq
# from collections import OrderedDict
# import bisect
# from collections import deque
from collections import defaultdict

INF = 10**18
MIN = -1 * INF
dx = [1, 0, -1, 0]
dy = [0, 1, 0, -1]
dxy = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def main():
    N, T = map(int, input().split())
    A = list(map(int, input().split()))

    mass = [[INF] * N for _ in range(N)]

    for k, a in enumerate(A):
        i = a // N
        j = a % N - 1
        if j == -1:
            i -= 1
            j = N - 1
        mass[i][j] = k + 1

    done_min = INF

    # よこのビンゴ確認
    for i in range(N):
        done_min = min(done_min, max(mass[i]))

    # 縦
    for i in range(N):
        done_min = min(done_min, max([mass[j][i] for j in range(N)]))

    # 斜め
    done_min = min(done_min, max([mass[i][i] for i in range(N)]))
    done_min = min(done_min, max([mass[i][N - 1 - i] for i in range(N)]))

    print(done_min if done_min != INF else -1)


def swap(A, i, j):
    tmp = A[i]
    A[i] = A[j]
    A[j] = tmp


# ポリオミノを右に90度回転


def rotate(matrix):
    N = len(matrix)
    M = len(matrix[0])
    rotated = [[0] * N for _ in range(M)]
    for i in range(N):
        for j in range(M):
            rotated[j][N - 1 - i] = matrix[i][j]
    return rotated


# # と .で表現された二次元配列を01で表現する
def change(P):
    p = []
    for i in P:
        line = []
        for j in i:
            if j == "#":
                line.append(1)
            else:
                line.append(0)
        p.append(line)
    return p


def basezero(num):
    return num - 1


def checkIndex(list, i):
    length = len(list)

    if 0 <= i < length:
        return True
    else:
        return False


def checkIndex2(list, i, j):
    H = len(list)
    W = len(list[0])
    if 0 <= i < H and 0 <= j < W:
        return True
    else:
        return False


# index_listをリストを区間Kで探索し、区間内の(最大値-最小値)の最小値を求める
class SegTree:
    """
    init(init_val, ide_ele): 配列init_valで初期化 O(N)
    update(k, x): k番目の値をxに更新 O(logN)
    query(l, r): 区間[l, r)をsegfuncしたものを返す O(logN)
    """

    def __init__(self, init_val, segfunc, ide_ele):
        """
        init_val: 配列の初期値
        segfunc: 区間にしたい操作
        ide_ele: 単位元
        n: 要素数
        num: n以上の最小の2のべき乗
        tree: セグメント木(1-index)
        """
        n = len(init_val)
        self.segfunc = segfunc
        self.ide_ele = ide_ele
        self.num = 1 << (n - 1).bit_length()
        self.tree = [ide_ele] * 2 * self.num
        # 配列の値を葉にセット
        for i in range(n):
            self.tree[self.num + i] = init_val[i]
        # 構築していく
        for i in range(self.num - 1, 0, -1):
            self.tree[i] = self.segfunc(self.tree[2 * i], self.tree[2 * i + 1])

    def update(self, k, x):
        """
        k番目の値をxに更新
        k: index(0-index)
        x: update value
        """
        k += self.num
        self.tree[k] = x
        while k > 1:
            self.tree[k >> 1] = self.segfunc(self.tree[k], self.tree[k ^ 1])
            k >>= 1

    def query(self, l, r):
        """
        [l, r)のsegfuncしたものを得る
        l: index(0-index)
        r: index(0-index)
        """
        res = self.ide_ele
        l += self.num
        r += self.num
        while l < r:
            if l & 1:
                res = self.segfunc(res, self.tree[l])
                l += 1
            if r & 1:
                res = self.segfunc(res, self.tree[r - 1])
            l >>= 1
            r >>= 1
        return res


"""
Binary Indexed Tree (BIT)を実装したクラス
区間の和をlogNで求めることができる
詳しくは以下の記事を参照
https://algo-logic.info/binary-indexed-tree/
転倒数を調べるのに使える
https://scrapbox.io/pocala-kyopro/%E8%BB%A2%E5%80%92%E6%95%B0
"""


class BIT:
    """
    N: 要素数
    Nをもとにbitを初期化する
    """

    def __init__(self, N):
        self.N = N
        self.bit = [0] * (N + 1)

    """
    i番目の要素にxを加算する
    i: 1-indexed
    x: 加算する値
    """

    def add(self, i, x):
        idx = i
        while idx <= self.N:
            self.bit[idx] += x
            idx += idx & (-idx)

    """
    i番目までの要素の和を求める
    i: 1-indexed
    戻り値: 0~i番目までの和
    """

    def sum(self, i):
        idx = i
        ans = 0
        while idx > 0:
            ans += self.bit[idx]
            idx -= (-idx) & idx
        return ans


class UnionFind:
    def __init__(self, n):
        self.n = n
        self.parents = [-1] * n

    def find(self, x):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.parents[x] > self.parents[y]:
            x, y = y, x

        self.parents[x] += self.parents[y]
        self.parents[y] = x

    def size(self, x):
        return -self.parents[self.find(x)]

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def members(self, x):
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]

    def roots(self):
        return [i for i, x in enumerate(self.parents) if x < 0]

    def group_count(self):
        return len(self.roots())

    def all_group_members(self):
        group_members = defaultdict(list)
        for member in range(self.n):
            group_members[self.find(member)].append(member)
        return group_members

    def __str__(self):
        return "\n".join(f"{r}: {m}" for r, m in self.all_group_members().items())


class WeightedDSU:
    """重み付きUnionFind

    wuf=WeightedDSU(N): 初期化
    wuf.leader(x): xの根を返します。
    wuf.merge(x,y,w): weight(x)-weight(y)でx,yを結合
    wuf.same(x,y): xとyが同じグループに所属するかどうかを返す
    wuf.diff(x,y): weight(x)-weight(y)を返す
    """

    def __init__(self, n: int):
        self.par = [i for i in range(n + 1)]
        self.rank = [0] * (n + 1)
        self.weight = [0] * (n + 1)

    def leader(self, x: int) -> int:
        if self.par[x] == x:
            return x
        else:
            y = self.leader(self.par[x])
            self.weight[x] += self.weight[self.par[x]]
            self.par[x] = y
            return y

    def merge(self, x: int, y: int, w: int):
        rx = self.leader(x)
        ry = self.leader(y)
        if self.rank[rx] < self.rank[ry]:
            self.par[rx] = ry
            self.weight[rx] = w - self.weight[x] + self.weight[y]
        else:
            self.par[ry] = rx
            self.weight[ry] = -w - self.weight[y] + self.weight[x]
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1

    def same(self, x: int, y: int) -> bool:
        return self.leader(x) == self.leader(y)

    def diff(self, x: int, y: int) -> bool:
        return self.weight[x] - self.weight[y]


# 削除可能ヒープ
# self.heap: ヒープ
# self.delete_heap: 削除予約待ちのヒープ
# self.item_cnt: defaultdict(key:heapの要素からdelete_heapを除いた要素，value:その要素が何個あるか)
# len(): ヒープから削除予約待ちのヒープを除いた要素数
# __repr__(): ヒープから削除予約待ちのヒープを除いた要素をリストにして返す
# __iter__(): ヒープから削除予約待ちのヒープを除いた要素をイテレータとして返す
# pop(): ヒープの最小値を取り出す
# push(item): ヒープにitemを追加する
# remove(item): ヒープからitemを削除する(内部的には削除予約しておいて，その要素がヒープの先頭にきたら削除する)
# get_item_cnt(item): ヒープ内のitemの個数を返す
from heapq import *


class DeletableHeap:
    def __init__(self, heap):
        self.heap = heap
        self.delete_heap = []
        self.item_cnt = defaultdict(int)
        for item in heap:
            self.item_cnt[item] += 1

    def __len__(self):
        return len(self.heap) - len(self.delete_heap)

    def __repr__(self):
        return str(list(self))

    def __iter__(self):
        j = 0
        for item in self.heap:
            if j < len(self.delete_heap):
                if item == self.delete_heap[j]:
                    j += 1
                    continue
            yield item

    def pop(self):
        while self.heap and self.delete_heap:
            if self.heap[0] == self.delete_heap[0]:
                heappop(self.heap)
                heappop(self.delete_heap)
            else:
                break
        return_item = heappop(self.heap)
        self.item_cnt[return_item] -= 1
        return return_item

    def push(self, item):
        heappush(self.heap, item)
        self.item_cnt[item] += 1

    def remove(self, item):
        heappush(self.delete_heap, item)
        self.item_cnt[item] -= 1
        while self.heap and self.delete_heap:
            if self.heap[0] == self.delete_heap[0]:
                heappop(self.heap)
                heappop(self.delete_heap)
            else:
                break

    def get_item_cnt(self, item):
        return self.item_cnt[item]


def yes():
    print("Yes")


def no():
    print("No")


def minusOne():
    print(-1)


if __name__ == "__main__":
    main()
