def arith_geo(list_n):
    if len(list_n) == 0:
        return 0
    diff = list_n[1] - list_n[0]
    div = list_n[1] / list_n[0]
    for num in range(1, len(list_n)):
        if list_n[num] - list_n[num - 1] == diff:
            counter = 'Arithmetic'
        elif list_n[num] / list_n[num-1] == div:
            counter = 'Geometric'
        else:
            counter = -1
            break
    return counter

print(arith_geo([180000,200000,220000,240000]))