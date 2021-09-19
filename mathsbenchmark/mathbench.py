import time
import platform
import math

PI = math.pi

print("Welcome to the CPU mathematical opeartion speed test")
print('This will execute complex mathematical operation 10 times')

if "Linux" == platform.system():
  print('Processor:')
  with open('/proc/cpuinfo') as cpuinoffile:
    for line in cpuinoffile:
        if line.strip():
            if line.rstrip('\n').startswith('model name'):
                model_name = line.rstrip('\n').split(':')[1]
                print(model_name)
                break
else:
  print('Your CPU is only shown automatic on Linux system.')

def factorial(n):
    fact = 1
    for i in range(1, n + 1):
        fact = fact * i
    return fact



noperation = 1000
ntime = 10
times = 0

for a in range(0,ntime):

  start = time.time()

  for i in range(0,noperation):
    for x in range(1,1000):
      PI * 2**x
    for x in range(1,100):
      factorial(x)
    for x in range(1,10000):
      math.log(x)

  end = time.time()
  duration = (end - start)
  duration = round(duration, 1)
  times += duration
  print('Time taken for complex mathematical operation: ' + str(duration) + 's')

average_time = round(times / ntime, 3)
print('Average time taken for complex mathematical operation : ' + str(average_time) + 's')