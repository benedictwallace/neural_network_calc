import random
import pandas as pd
import numpy as np
import math

output_no = 200

def dtb(n):
    a = str(bin(n).replace("0b", ""))
    while len(a) < 8:
        a = "0" + a
    return a

x = []

# Format [(x)(y)(sign)(z)]
# sign: 0 = -, 1 = +, 2 = *, 3 = /
for i in range(0, 40000):
    sign = random.randint(0,3)
    if sign < 2:
        c = random.randint(1, output_no)
        b = random.randint(0,c)
        a = c - b
        if sign == 1:
            s = dtb(a) + dtb(b)
            s = [*s]
            for i in range(0, 4):
                if i == sign:
                    s.append(1)
                else:
                    s.append(0)
            s.append(c) 
        else:
            s = dtb(c) + dtb(a)
            s = [*s]
            for i in range(0, 4):
                if i == sign:
                    s.append(1)
                else:
                    s.append(0)
            s.append(b)
    else: 
        a = random.randint(1, math.floor(np.sqrt(output_no)))
        b = random.randint(1, math.floor(output_no/a))
        c = a * b
        if sign == 2:
            s = dtb(a) + dtb(b)
            s = [*s]
            for i in range(0, 4):
                if i == sign:
                    s.append(1)
                else:
                    s.append(0)
            s.append(c)
        else:
            s = dtb(c) + dtb(a)
            s = [*s]
            for i in range(0, 4):
                if i == sign:
                    s.append(1)
                else:
                    s.append(0)
            s.append(b)        
    x.append(s)

df = pd.DataFrame(x)

df.to_csv("C:/Users/b.wallace/Desktop/python/input_data/training_data.csv", header=False, index=False)

print("done")
