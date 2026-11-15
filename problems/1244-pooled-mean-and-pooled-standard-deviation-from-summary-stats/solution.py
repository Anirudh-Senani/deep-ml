def pooled_stats(m1, s1, n1, m2, s2, n2):
    # m1, s1, n1: mean, sample sd, size of group 1
    # m2, s2, n2: mean, sample sd, size of group 2
    # return (combined_mean, combined_sd)
    cm = (m1 * n1 + m2 * n2)/(n1 + n2)
    cs = (s1**2 * (n1 - 1) + s2**2 *
          (n2 - 1)) + (n1 * (m1 - cm)**2 + n2 * (m2 - cm)**2)
    cs = cs/(n1 + n2 - 1)
    cs = cs**0.5

    return [cm, cs]
