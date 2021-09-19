import zipfile
import time

print("Welcome to zipping benchmark test")
print("In this test different zipping operation are preformed")

print("")
print("Test 1: Archiving hundred files of 10mb")
#files = ["input/newfile.txt"]
files = ["input/mb"+str(i)+".txt" for i in range(0,100)]
archive = "output/archive.zip"

total_start = time.time()
#Zipping all the files and storing in archive.zip
with zipfile.ZipFile(archive, "w") as zf:
    for f in files:
        zf.write(f)

test1time = time.time() - total_start
print("Time: "+ str(round(test1time,2))+"sec")

print("")
print("Test 2: Checking for bad zipping file")
# testing archived file
with zipfile.ZipFile(archive, "r") as zf:
    crc = zf.testzip()
    if crc is not None:
        print(f"Bad file header: {crc}")
    info = zf.infolist()  # also zf.namelist()
    #print(info)

    # read the file
    file = info[0]
    with zf.open(file) as f:
        f.read().decode()

    test2time = time.time() - (total_start + test1time)
    print("Time: " + str(round(test2time, 2))+"sec")

    print("")
    print("Test 3: Extracting zip file")
    #extrac the file
    zf.extract(file, "/tmp")  # also zf.extractall()

    test3time = time.time() - (total_start + test1time + test2time)
    print("Time: " + str(round(test3time, 2))+"sec")
end = time.time()
duration = (end - total_start)
duration = round(duration, 2)
print("")
print('Total time taken for zipping test: ' + str(duration) +"sec")