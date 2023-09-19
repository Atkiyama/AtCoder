from functools import lru_cache
import sys
sys.setrecursionlimit(10**9)
# input = sys.stdin.readline
from decimal import Decimal
from functools import cmp_to_key
from collections import Counter
from itertools import permutations
from itertools import combinations
from itertools import combinations_with_replacement
from itertools import product
from itertools import accumulate
from itertools import groupby
from itertools import pairwise
from copy import deepcopy
import networkx as nx
import networkx.algorithms as nxa
import numpy as np
import math
import heapq
from collections import OrderedDict
import bisect
from collections import deque
from collections import defaultdict

'''
https://twitter.com/e869120/status/1379565222541680644
そのままだとTLEなのでソートをかけて二分探索をする
二分探索でインデックスの入る範囲を探すのはbisect.bisect_left(配列, 探す値)でできる
返り値として探す値<配列[i]となる最小のiを返す
'''
INF = 2000000000
def main():
    
    N = int(input())
    A = list(map(int, input().split()))
    Q = int(input())
    B = []
    for i in range(Q):
        B.append(int(input()))
    A.sort()
    for b in range(Q):
        place=bisect.bisect_left(A, B[b])
        diff1=INF
        diff2=INF
        if place >0:
            diff1= abs(A[place-1]-B[b])
        if place <N:
            diff2= abs(A[place]-B[b])
        print(min(diff1,diff2))
       
if __name__ == "__main__":
    main()