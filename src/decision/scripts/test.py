import os
a = os.path.dirname(__file__)
a += '/aaa.txt'
print(a)
with open(a, 'a') as f:
    f.write('hhhh'+'\n')

