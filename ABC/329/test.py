N = int(input())
S = input()

ans_set = set()
tmp = []
now = ""

for i in range(N):

    if (now != S[i]):
        tmp = [S[i]]
        now = S[i]

    elif now == S[i]:
        tmp.append(S[i])
        
    ans_set.add("".join(tmp))
    print(ans_set)
print(len(ans_set))