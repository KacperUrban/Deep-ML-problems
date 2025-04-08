def min_max(x: list[int]) -> list[float]:
    x_min = min(x)
    span = max(x) - min(x)

    if span == 0:
        return [0 for _ in range(len(x))]
    return [(elem - x_min) / span for elem in x]