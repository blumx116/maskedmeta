def exponential_average(vals, gamma=0.99):
    result = [vals[0]]
    for v in vals[1:]:
        result.append((gamma * result[-1]) + ((1-gamma) * v))
    return result