'''
【第 1 条】
  query-id:  q7395
  corpus-id: d7395
  score:     1
------------------------------------------------------------
  QUERY 内容:
    A tree is an undirected connected graph without cycles. The distance between two vertices is the number of edges in a simple path between them.

Limak is a little polar bear. He lives in a tree that consists of n vertices, numbered 1 through n.

Limak recently learned how to jump. He can jump from a vertex to any vertex within distance at most k.

For a pair of vertices (s, t) we define f(s, t) as the minimum number of jumps Limak needs to get from s to t. Your task is to find the sum of f(s, t) over all pairs of vertices (s, t) such that s < t.


-----Input-----

The first line of the input contains two integers n and k (2 ≤ n ≤ 200 000, 1 ≤ k ≤ 5) — the number of vertices in the tree and the maximum allowed jump distance respectively.

The next n - 1 lines describe edges in the tree. The i-th of those lines contains two integers a_{i} and b_{i} (1 ≤ a_{i}, b_{i} ≤ n) — the indices on vertices connected with i-th edge.

It's guaranteed that the given edges form a tree.


-----Output-----

Print one integer, denoting the sum of f(s, t) over all pairs of vertices (s, t) such that s < t.


-----Examples-----
Input
6 2
1 2
1 3
2 4
2 5
4 6

Output
20

Input
13 3
1 2
3 2
4 2
5 2
3 6
10 6
6 7
6 13
5 8
5 9
9 11
11 12

Output
114

Input
3 5
2 1
3 1

Output
3



-----Note-----

In the first sample, the given tree has 6 vertices and it's displayed on the drawing below. Limak can jump to any vertex within distance at most 2. For example, from the vertex 5 he can jump to any of vertices: 1, 2 and 4 (well, he can also jump to the vertex 5 itself). [Image] 

There are $\frac{n \cdot(n - 1)}{2} = 15$ pairs of vertices (s, t) such that s < t. For 5 of those pairs Limak would need two jumps: (1, 6), (3, 4), (3, 5), (3, 6), (5, 6). For other 10 pairs one jump is enough. So, the answer is 5·2 + 10·1 = 20.

In the third sample, Limak can jump between every two vertices directly. There are 3 pairs of vertices (s < t), so the answer is 3·1 = 3.
------------------------------------------------------------
  CORPUS 内容:
    title: 
    text:  """
#If FastIO not needed, used this and don't forget to strip
#import sys, math
#input = sys.stdin.readline
"""

import os
import sys
from io import BytesIO, IOBase
import heapq as h 
from bisect import bisect_left, bisect_right

from types import GeneratorType
BUFSIZE = 8192
class FastIO(IOBase):
    newlines = 0
 
    def __init__(self, file):
        import os
        self.os = os
        self._fd = file.fileno()
        self.buffer = BytesIO()
        self.writable = "x" in file.mode or "r" not in file.mode
        self.write = self.buffer.write if self.writable else None
 
    def read(self):
        while True:
            b = self.os.read(self._fd, max(self.os.fstat(self._fd).st_size, BUFSIZE))
            if not b:
                break
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines = 0
        return self.buffer.read()
 
    def readline(self):
        while self.newlines == 0:
            b = self.os.read(self._fd, max(self.os.fstat(self._fd).st_size, BUFSIZE))
            self.newlines = b.count(b"\n") + (not b)
            ptr = self.buffer.tell()
            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
        self.newlines -= 1
        return self.buffer.readline()
 
    def flush(self):
        if self.writable:
            self.os.write(self._fd, self.buffer.getvalue())
            self.buffer.truncate(0), self.buffer.seek(0)
 
 
class IOWrapper(IOBase):
    def __init__(self, file):
        self.buffer = FastIO(file)
        self.flush = self.buffer.flush
        self.writable = self.buffer.writable
        self.write = lambda s: self.buffer.write(s.encode("ascii"))
        self.read = lambda: self.buffer.read().decode("ascii")
        self.readline = lambda: self.buffer.readline().decode("ascii")
 
 
sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip("\r\n")

from collections import defaultdict as dd, deque as dq, Counter as dc
import math, string


def getInts():
    return [int(s) for s in input().split()]

def getInt():
    return int(input())

def getStrs():
    return [s for s in input().split()]

def getStr():
    return input()

def listStr():
    return list(input())

def getMat(n):
    return [getInts() for _ in range(n)]

MOD = 10**9+7


"""
Each edge goes from parent U to child V
Edge appears on S_V * (N - S_V) paths

For each path of length L, (L + (-L)%K)/K


L%K 0, 1, 2, 3, 4
(K - L%K)%K K K-1 K-2 ...
0 K-1 K-2 ...

"""
def bootstrap(f, stack=[]):
    def wrappedfunc(*args, **kwargs):
        if stack:
            return f(*args, **kwargs)
        else:
            to = f(*args, **kwargs)
            while True:
                if type(to) is GeneratorType:
                    stack.append(to)
                    to = next(to)
                else:
                    stack.pop()
                    if not stack:
                        break
                    to = stack[-1].send(to)
            return to
    return wrappedfunc

def solve():
    N, K = getInts()
    graph = dd(set)
    for i in range(N-1):
        A, B = getInts()
        graph[A].add(B)
        graph[B].add(A)
    dp_count = [[0 for j in range(5)] for i in range(N+1)]
    dp_total = [0 for j in range(N+1)]
    nonlocal ans
    ans = 0
    @bootstrap
    def dfs(node,parent,depth):
        nonlocal ans
        dp_count[node][depth % K] = 1
        dp_total[node] = 1
        for neigh in graph[node]:
            if neigh != parent:
                yield dfs(neigh,node,depth+1)
                for i in range(K):
                    for j in range(K):
                        diff = (i+j-2*depth)%K
                        req = (-diff)%K
                        ans += req * dp_count[node][i] * dp_count[neigh][j]
                for i in range(K):
                    dp_count[node][i] += dp_count[neigh][i]
                dp_total[node] += dp_total[neigh]
        ans += dp_total[node] * (N - dp_total[node])
        yield
    dfs(1,-1,0)
    return ans//K
    
    
print(solve())

================================================================================

【第 2 条】
  query-id:  q5561
  corpus-id: d5561
  score:     1
------------------------------------------------------------
  QUERY 内容:
    You are given a rectangular cake, represented as an r × c grid. Each cell either has an evil strawberry, or is empty. For example, a 3 × 4 cake may look as follows: [Image] 

The cakeminator is going to eat the cake! Each time he eats, he chooses a row or a column that does not contain any evil strawberries and contains at least one cake cell that has not been eaten before, and eats all the cake cells there. He may decide to eat any number of times.

Please output the maximum number of cake cells that the cakeminator can eat.


-----Input-----

The first line contains two integers r and c (2 ≤ r, c ≤ 10), denoting the number of rows and the number of columns of the cake. The next r lines each contains c characters — the j-th character of the i-th line denotes the content of the cell at row i and column j, and is either one of these:   '.' character denotes a cake cell with no evil strawberry;  'S' character denotes a cake cell with an evil strawberry. 


-----Output-----

Output the maximum number of cake cells that the cakeminator can eat.


-----Examples-----
Input
3 4
S...
....
..S.

Output
8



-----Note-----

For the first example, one possible way to eat the maximum number of cake cells is as follows (perform 3 eats). [Image]  [Image]  [Image]
------------------------------------------------------------
  CORPUS 内容:
    title: 
    text:  r, c = list(map(int, input().split()))
cake = [input().strip() for _ in range(r)]
ans = 0
for i in range(r):
    for j in range(c):
        if cake[i][j] == '.' and ('S' not in cake[i] or 'S' not in list(zip(*cake))[j]):
            ans += 1
print(ans)

================================================================================

【第 3 条】
  query-id:  q5354
  corpus-id: d5354
  score:     1
------------------------------------------------------------
  QUERY 内容:
    Every summer Vitya comes to visit his grandmother in the countryside. This summer, he got a huge wart. Every grandma knows that one should treat warts when the moon goes down. Thus, Vitya has to catch the moment when the moon is down.

Moon cycle lasts 30 days. The size of the visible part of the moon (in Vitya's units) for each day is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, and then cycle repeats, thus after the second 1 again goes 0.

As there is no internet in the countryside, Vitya has been watching the moon for n consecutive days and for each of these days he wrote down the size of the visible part of the moon. Help him find out whether the moon will be up or down next day, or this cannot be determined by the data he has.


-----Input-----

The first line of the input contains a single integer n (1 ≤ n ≤ 92) — the number of consecutive days Vitya was watching the size of the visible part of the moon. 

The second line contains n integers a_{i} (0 ≤ a_{i} ≤ 15) — Vitya's records.

It's guaranteed that the input data is consistent.


-----Output-----

If Vitya can be sure that the size of visible part of the moon on day n + 1 will be less than the size of the visible part on day n, then print "DOWN" at the only line of the output. If he might be sure that the size of the visible part will increase, then print "UP". If it's impossible to determine what exactly will happen with the moon, print -1.


-----Examples-----
Input
5
3 4 5 6 7

Output
UP

Input
7
12 13 14 15 14 13 12

Output
DOWN

Input
1
8

Output
-1



-----Note-----

In the first sample, the size of the moon on the next day will be equal to 8, thus the answer is "UP".

In the second sample, the size of the moon on the next day will be 11, thus the answer is "DOWN".

In the third sample, there is no way to determine whether the size of the moon on the next day will be 7 or 9, thus the answer is -1.
------------------------------------------------------------
  CORPUS 内容:
    title: 
    text:  # You lost the game.

n = int(input())
L = list(map(int, input().split()))

if n == 1:
    if L[0] == 0:
        print("UP")
    elif L[0] == 15:
        print("DOWN")
    else:
        print("-1")
else:
    d = L[n-2] - L[n-1]
    if d < 0:
        if L[n-1] == 15:
            print("DOWN")
        else:
            print("UP")
    else:
        if L[n-1] == 0:
            print("UP")
        else:
            print("DOWN")

================================================================================

【第 4 条】
  query-id:  q5074
  corpus-id: d5074
  score:     1
------------------------------------------------------------
  QUERY 内容:
    Mister B once received a gift: it was a book about aliens, which he started read immediately. This book had c pages.

At first day Mister B read v_0 pages, but after that he started to speed up. Every day, starting from the second, he read a pages more than on the previous day (at first day he read v_0 pages, at second — v_0 + a pages, at third — v_0 + 2a pages, and so on). But Mister B is just a human, so he physically wasn't able to read more than v_1 pages per day.

Also, to refresh his memory, every day, starting from the second, Mister B had to reread last l pages he read on the previous day. Mister B finished the book when he read the last page for the first time.

Help Mister B to calculate how many days he needed to finish the book.


-----Input-----

First and only line contains five space-separated integers: c, v_0, v_1, a and l (1 ≤ c ≤ 1000, 0 ≤ l < v_0 ≤ v_1 ≤ 1000, 0 ≤ a ≤ 1000) — the length of the book in pages, the initial reading speed, the maximum reading speed, the acceleration in reading speed and the number of pages for rereading.


-----Output-----

Print one integer — the number of days Mister B needed to finish the book.


-----Examples-----
Input
5 5 10 5 4

Output
1

Input
12 4 12 4 1

Output
3

Input
15 1 100 0 0

Output
15



-----Note-----

In the first sample test the book contains 5 pages, so Mister B read it right at the first day.

In the second sample test at first day Mister B read pages number 1 - 4, at second day — 4 - 11, at third day — 11 - 12 and finished the book.

In third sample test every day Mister B read 1 page of the book, so he finished in 15 days.
------------------------------------------------------------
  CORPUS 内容:
    title: 
    text:  read = lambda: map(int, input().split())
c, v0, v1, a, l = read()
cur = 0
cnt = 0
while cur < c:
    cur = max(0, cur - l)
    cur += min(v1, v0 + a * cnt)
    cnt += 1
print(cnt)
================================================================================

【第 5 条】
  query-id:  q8473
  corpus-id: d8473
  score:     1
------------------------------------------------------------
  QUERY 内容:
    You are given an array $a$ consisting of $n$ integers. In one move, you can jump from the position $i$ to the position $i - a_i$ (if $1 \le i - a_i$) or to the position $i + a_i$ (if $i + a_i \le n$).

For each position $i$ from $1$ to $n$ you want to know the minimum the number of moves required to reach any position $j$ such that $a_j$ has the opposite parity from $a_i$ (i.e. if $a_i$ is odd then $a_j$ has to be even and vice versa).


-----Input-----

The first line of the input contains one integer $n$ ($1 \le n \le 2 \cdot 10^5$) — the number of elements in $a$.

The second line of the input contains $n$ integers $a_1, a_2, \dots, a_n$ ($1 \le a_i \le n$), where $a_i$ is the $i$-th element of $a$.


-----Output-----

Print $n$ integers $d_1, d_2, \dots, d_n$, where $d_i$ is the minimum the number of moves required to reach any position $j$ such that $a_j$ has the opposite parity from $a_i$ (i.e. if $a_i$ is odd then $a_j$ has to be even and vice versa) or -1 if it is impossible to reach such a position.


-----Example-----
Input
10
4 5 7 6 7 5 4 4 6 4

Output
1 1 1 2 -1 1 1 3 1 1
------------------------------------------------------------
  CORPUS 内容:
    title: 
    text:  from heapq import heappush, heappop

n = int(input())
a = list(map(int, input().split()))
edge = [[] for _ in range(n)]
rev = [[] for _ in range(n)]
inf = 10**9
cost = [inf]*n
hq = []

for i, x in enumerate(a):
    if i+x < n:
        edge[i].append(i+x)
        rev[i+x].append(i)
        if (a[i] ^ a[i+x]) & 1:
            cost[i] = 1
    if i-x >= 0:
        edge[i].append(i-x)
        rev[i-x].append(i)
        if (a[i] ^ a[i-x]) & 1:
            cost[i] = 1

    if cost[i] == 1:
        hq.append((1, i))

while hq:
    c, v = heappop(hq)
    if cost[v] < c:
        continue
    c += 1
    for dest in rev[v]:
        if cost[dest] > c:
            cost[dest] = c
            heappush(hq, (c, dest))

for i in range(n):
    if cost[i] == inf:
        cost[i] = -1

print(*cost)

================================================================================

'''