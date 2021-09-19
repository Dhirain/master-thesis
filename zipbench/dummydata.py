import os
import random
import string
GB1 = 1024*1024*10 # 1GB
i = 0
print('Creating hundred dummy data file of 10 mb with mb.txt')
#chars = ''.join([random.choice(string.ascii_letters) for i in range(GB1)])
for i in range(0,100):
    with open('input/mb%s.txt' %i, 'wb') as fout:
        fout.write(str('0' * GB1).encode())

