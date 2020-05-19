---
toc: true
layout: post
description: Notes for graph search algorithm problems
categories: [note, algo]
title: "Graph Search: Subsets, Permutations"
---


### Traits for using BFS:

1. Shortest path in a simple graph: given a initial state, a final state, a transition rule between states, ask how many (the least #) transitions from init to final.
2. Graph traversal. Only visit each node once.

### Traits for using DFS:

(DFS method: build search tree, check conditions for recursing down a branch)

1. Enumerate subsets, permutations.
2. Find all possible solutions.

Generally, when we do a search to enumerate cases using DFS recursion, there are 3 steps we should keep in mind,

1. Define the recursion.
2. Think about what to do in the base case. When should we return directly.
3. In general cases (other than base case), how to make the problem smaller and recurse down.


### Subsets I & II (DFS template)

The thinking is to categorize cases by different head items.
Enumerate the head item of a case (path) in a for loop in this way:

1. Append the item into the path.
2. DFS recurse down onto the next case (generally a smaller case, with advanced index and/or updated reference parameter).
3. Pop the item from the path, to iterate to a different head item on the next iteration.

For this specific Subsets problem, the base case is just adding the current path. For Subsets II, the only difference is that the input can have duplicates and we don't want the result subsets to be duplicate sets. Since the input is sorted (or we sort it by ourselves before DFS), when we encounter a number which is equal to the previous number in the for loop, we continue. Because the same number is taking the same place as the previous one, the resulting subsets with either of them are the same sets.


### Permutations I & II (DFS template)

It's quite similar to the Subsets problems. The thinking is also to categorize cases by different head items, and enumerate the head item of a case (path) in a for loop. The difference is that now we don't want to keep track of the index as a parameter passed into DFS. Our base case is that when the path has the same length as the original input sequence, the current path is added.

The for loop is now as such:

1. Append the item into the path.
2. DFS recurse down after appending the new head item. Avoid the same number by checking if it's already in path, if yes, continue.
3. Pop the item from the path, iterate to a different head item on the next iteration.

For permutations II where we allow duplicates in the input list, we must sort it first and then do DFS. In the results, duplicate permutations must be avoided, but how? We introduce a new list, visited. We only add continuous same numbers to path, meaning if the previous same number is not visited, we continue. Check the code for details.


### Summary for Subsets and Permutations

This kind of problems is not easy to understand. Recursion tree diagrams can help to clarify, but also keep in mind the code templates: inside the for loop, check condition to recurse down, then append in path, DFS down with path (with appropriate update in some parameter), pop from path.

<br>

#### Subsets

<img src="{{ site.baseurl }}/images/algo/subsets.png" alt="Drawing" style="width: 1000px;"/>

```py
class Solution:
    """
    @param S: The set of numbers.
    @return: A list of lists. See example.
    """
    def subsets(self, S):
        if S is None or len(S) == 0:
            return []
        S.sort()
        self.results = []
        self.DFS([], 0, S)
        return self.results

    def DFS(self, path, ind, S):
        # base case, add each path of each recursion (sorted as required)
        # must make new list, list(path). If not,
        # res (path) points to the obj passed in, which is empty at the beginning
        res = list(path)
        self.results.append(res)
        # i is the first item's index in a path
        for i in xrange(ind, len(S)):
            path.append(S[i])
            self.DFS(path, i+1, S)
            path.pop()
```

<br>

#### Permutations:

<img src="{{ site.baseurl }}/images/algo/permutations.png" alt="Drawing" style="width: 1000px;"/>

```py
class Solution:
    """
    @param nums: A list of Integers.
    @return: A list of permutations.
    """
    def permute(self, nums):
        if nums is None or len(nums) == 0:
            return []

        self.results = []
        self.DFS([], nums)
        return self.results

    def DFS(self, path, nums):
        # base case
        if len(path) == len(nums):
            # must make new list, list(path). If not,
            # it points to the obj passed in, which is empty at
            # the beginning
            self.results.append(list(path))
            return

        for i in xrange(len(nums)):
            # check if the ith number is already in path
            if nums[i] in path:
                continue
            path.append(nums[i])
            self.DFS(path, nums)
            path.pop()
```


Combination Sum is the Sum version of Subsets, with duplicates allowed.

<br>

#### Palindrome Partitioning

We deem the cuts to be the member of a subset, and this problem becomes finding all subsets of valid cuts. If there are N cuts, we can choose whether to include each cut, so there are 2^N ways to cut our string. For O(2^N) problems, it's usually a Subsets problem.

The thinking is that, we have a substring from start to i, s[start:i], called prefix. This is the next head item of the new node (path) in the DFS tree, we later append it to path, DFS down, and pop. But before that we should check if it is a valid palindrome.

For a fixed start, we loop through all substrings starting there, check if it satisfies the condition (palindrome in this case), if yes DFS down starting at i (the next char after s[old_start:i]). Again the template of DFS:

{% highlight python %}
for i in range(old_start_ind, data.length):
    get next_head item
    check if next_head satisfies our condition
        if not, continue
    path.append(next_head)
    DFS(path, i or i+1, data)    # i or i+1 greater than old_start_ind
    path.pop()
{% endhighlight %}

For `"aab"`, we have start = 0: `|a|ab, |aa|b, |aab|`; start = 1: `a|a|b, a|ab|`; start = 2: `aa|b|`; start = 3 == len, `aab|`, add one full path and return. start progresses by new recursion, i scans inside each recursion from `start+1` to `len+1`.

This is the method to enumerate all substrings that satisfies some condition.

<br>

#### Factor Combinations

For example, 8 -> `[[2, 2, 2], [2, 4]]`. Do not include 1 or n as a factor, and each sublist should be ascending.

This is a typical DFS problem: list all solutions (combinations) or a certain decomposition problem, in this case,
factorization.

```python
class Solution:

    def factors(self, n):
        self.result = []
        if n <= 1:
            return self.result
        path = []
        self.dfs(n, path)
        return self.result

    def dfs(self, n, path):
        # base case
        # len(path) == 1 is the case [n], which is not included
        # in this question
        if n == 1 and len(path) > 1:
            self.result.append(list(path))
            return
        # recursion
        # factor must include n itself
        # or when it's down to the last factor, it's not added
        for factor in xrange(2, n+1):
            # check if it's a factor of not
            if n % factor != 0:
                continue
            # ensure ascending order
            if path == [] or path[-1] <= factor:
                path.append(factor)
                self.dfs(n/factor, path)
                path.pop()
        return
```

3 points that need special attention in this particular problem,

- For n <= 3, `result = []`
- For the loop from 2 to n, must include n: `for factor in xrange(2, n+1): ...`. Because if we don't include n,
the factors are not added to `path`. For example,

```
If we use xrange(2, n)...

Input: 8

Ex1:

1st level DFS: n = 8, factor = 2, dfs(n/2 = 4, [2])
2nd level DFS: n = 4, factor = 2, dfs(n/2 = 2, [2, 2])
3rd level DFS: n = 2, factor is in xrange(2, 2) which is nothing, abort
Failed to add path [2, 2, 2]

Ex2:

1st level DFS: n = 8, factor = 2, dfs(n/2 = 4, [2])
2nd level DFS: n = 4, factor is in xrange(2, 4), factor cannot reach 4
Failed to add path [2, 4]
```

- To ensure ascending order in `path`s, check for `path == [] or path[-1] <= factor` before `append(factor)`.

(Note that Python has short circuit evaluation in the conditionals. For `or` it means that if `path == []`, the
latter part(s) won't be checked. So there won't be a case where `path[-1]` does not exist in this expression.)
