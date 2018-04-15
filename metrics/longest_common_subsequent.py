def lcs(a, b):
    if not a or not b:
        return 0
    if a[-1] == b[-1]:
        return 1+lcs(a[:-1], b[:-1])
    else:
        return max(lcs(a[:-1], b), lcs(a, b[:-1]))
