f1 = open('cpu_output.txt', 'r')
f2 = open('gpu_output.txt', 'r')

count = 0
for i in f1:
    tmp1 = i.strip()
    tmp2 = f2.readline().strip()

    if tmp1 != tmp2:
        count += 1

print("There are {} errors".format(count))
