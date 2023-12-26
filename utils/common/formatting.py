def stringToListOfFloat(input: str) -> 'list[float]':
    # print(input[2:-2].split(','))
    # return list(map(float, input[1:-1].split(',')))
    return list(map(float, input[2:-2].split(',')))

def stringToListOfFloat_Shot(input: str) -> 'list[float]':
    # print(input[2:-2].split(','))
    # return list(map(float, input[1:-1].split(',')))
    return list(map(float, input[1:-1].split(',')))