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
https://twitter.com/e869120/status/1380652465834532865
累積和を使う問題
そのままやるとTLEになるが累積和にしてA[R]-A[L-1]にすると各クエリに対してO(1)で計算できる
'''
def main():
    N=int(input())
    A=[]
    B=[]
    A.append(0)
    B.append(0)
    apre=0
    bpre=0
    for i in range(N):
        C,P=map(int,input().split())
        if C==1:
            apre+=P
        else:
            bpre+=P
        A.append(apre)
        B.append(bpre)
    # print(A)
    # print(B)
    Q=int(input())
    for i in range(Q):
        L,R=map(int,input().split())
        # L-=1
        # R-=1
        print(A[R]-A[L-1])
        print(B[R]-B[L-1])
        
        

if __name__ == "__main__":
    main()