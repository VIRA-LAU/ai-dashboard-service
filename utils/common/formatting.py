def stringToListOfFloat(input: str) -> list[float]:
    return list(map(float, input[1:-2].split(',')))