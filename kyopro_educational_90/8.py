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
https://twitter.com/e869120/status/1379927227739987972/photo/1
DPを使う問題
今回だとdp[文字数][現在の状態(何文字目までを選択するか)]の形は典型
'''

mod = 1000000007
def main():
    N=int(input())
    S=input()
    dp=[[0 for _ in range(8)] for _ in range(N+9)]
    atcoder="atcoder"
    dp[0][0]=1
    for i in range(len(S)):
        for j in range(8):
            dp[i+1][j]+=dp[i][j]
            if j<7 and S[i]==atcoder[j]:
                dp[i+1][j+1]+=dp[i][j]
            dp[i + 1][j] %= mod
    print(dp[len(S)][7]%mod)

if __name__ == "__main__":
    main()