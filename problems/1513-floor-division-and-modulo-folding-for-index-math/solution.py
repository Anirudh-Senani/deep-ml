def term_bounds(terms, ranges):
    lo = hi = terms.get("", 0)
    for v, k in terms.items():
        if v == "" or k == 0: continue
        a, b = ranges[v][0], ranges[v][1] - 1
        lo += min(k * a, k * b)
        hi += max(k * a, k * b)
    return (lo, hi)


def _clean(d):
    out = {v: k for v, k in d.items() if v == "" or k != 0}
    out.setdefault("", 0)
    return out


def fold_div(terms, c, ranges):
    q, rest = {"": 0}, {"": 0}
    for v, k in terms.items():
        if k % c == 0: q[v] = k // c
        else: rest[v] = k
    lo, hi = term_bounds(rest, ranges)
    if lo >= 0 and hi < c: return _clean(q), None
    return _clean(q), _clean(rest)


def fold_mod(terms, c, ranges):
    rest = {v: k for v, k in terms.items() if k % c != 0}
    rest.setdefault("", 0)
    lo, hi = term_bounds(rest, ranges)
    return _clean(rest), (lo >= 0 and hi < c)


def unflatten(shape, flat_terms, ranges):
    strides, acc = [], 1
    for s in reversed(shape):
        strides.append(acc); acc *= s
    strides = strides[::-1]
    out = []
    for st, s in zip(strides, shape):
        q, rem = fold_div(flat_terms, st, ranges)
        if rem is not None:
            out.append("div/mod"); continue
        m, exact = fold_mod(q, s, ranges)
        out.append(m if exact else "div/mod")
    return out
