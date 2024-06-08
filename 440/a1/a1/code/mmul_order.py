import numpy as np


def _fmt(a, b):
    return str(a) if a == b else f"{a}:{b}"


def mmul_order(dims):
    # the first matrix is dims[0] by dims[1]; the last is dims[k-1] by dims[k]
    # so, matrix i (in *one*-indexing, as in the math) is dims[i-1] by dims[i]

    # you should specify your order as string specifying the order of multiplications:
    # for instance, right-to-left for a list of four matrices would be something like
    #    3 x 4; 2 x 3:4; 1 x 2:4
    # the _fmt() function above might help

    # feel free to either construct an explicit table or use functools.cache, which we imported for you
    

    num = len(dims) - 1
    table = np.full((num, num), float('inf'))
    sol = np.zeros((num, num), dtype=int)

    for i in range (num):
        table[i][i] = 0


    for l in range(2, num):
        
        for i in range(num - l + 1):
            j = i + l - 1

            for k in range(i, j):
                q = table[i][k] + table[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]

                if   table[i][j]>q:
                    table[i][j] = q
                    sol[i][j] = k

    optimal_cost = table[0][num - 1]
    optim = construct(sol, 0, num - 1, _fmt)



    return optimal_cost, optim


def construct(split, i, j, _fmt):
    if i == j:
        return _fmt(i + 1, i + 1)  
    

    k = split[i][j] 

    left_order = construct(split, i, k, _fmt)
    right_order = construct(split, k + 1,  j, _fmt)


    return f"{left_order} x {right_order}"