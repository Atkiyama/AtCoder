# from functools import lru_cache
# import sys
# sys.setrecursionlimit(10**9)
# # input = sys.stdin.readline
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
#import heapq
# from collections import OrderedDict
# import bisect
# from collections import deque
from collections import defaultdict
INF = 10 ** 18
dx = [1, 0, -1, 0]
dy = [0, 1, 0, -1]
dxy = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def main():
    N=int(input())
    A=list(map(int,input().split()))
    
    maxAcc=[0]*(N)
    maxAccB=[0]*(N)
    
    maxAcc[0]=A[0]
    for i in range(1,N):
        maxAcc[i]=max(maxAcc[i-1],A[i])
        
    maxAccB[-1]=A[-1]
    for i in reversed(range(0,N-1)):
        maxAccB[i]=max(maxAccB[i+1],A[i])
        
    # print(maxAcc)
    # print(maxAccB)
    D=int(input())
    for _ in range(D):
        L,R=map(int,input().split())
        L-=1
        R-=1
        ans=max(maxAcc[L-1],maxAccB[R+1])
        print(ans)


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
            rotated[j][N-1-i] = matrix[i][j]
    return rotated


# # と .で表現された二次元配列を01で表現する
def change(P):
    p = []
    for i in P:
        line = []
        for j in i:
            if j == '#':
                line.append(1)
            else:
                line.append(0)
        p.append(line)
    return p


def basezero(num):
    return num-1


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


class UnionFind():
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
        return '\n'.join(f'{r}: {m}' for r, m in self.all_group_members().items())
    
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
    def __init__(self,heap):
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
    def push(self,item):
        heappush(self.heap,item)
        self.item_cnt[item] += 1
    def remove(self,item):
        heappush(self.delete_heap,item)
        self.item_cnt[item] -= 1
        while self.heap and self.delete_heap:
            if self.heap[0] == self.delete_heap[0]:
                heappop(self.heap)
                heappop(self.delete_heap)
            else:
                break
    def get_item_cnt(self,item):
        return self.item_cnt[item]

def yes():
    print("Yes")


def no():
    print("No")


def minusOne():
    print(-1)


if __name__ == "__main__":
    main()
