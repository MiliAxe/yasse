import sys
import time

from modules.bcolors import bcolors
from modules.dataprocessor import DataProcessor

# testing the speed
file_count = 200
time_limit = 0.25
calculate_occurrences = 0
try:
    file_count = int(sys.argv[1])
    calculate_occurrences = int(sys.argv[2])
    time_limit = float(sys.argv[3])
except IndexError:
    pass
start_time = time.time()
dp = DataProcessor()
for i in range(file_count):
    dp.add_file(f"data/document_{i}.txt")
dp.generate()

if calculate_occurrences:
    for word in dp.occur_dict:
        dp.occurences(word)

end_time = time.time()
exec_time = end_time - start_time
print(f"Adding {bcolors.BLUE}{file_count}{bcolors.ENDC} files took {bcolors.BLUE}{exec_time:.3f}{bcolors.ENDC} seconds",
      end=" ")
if calculate_occurrences:
    print("(Occurrences were calculated too)")
else:
    print()
try:
    assert (exec_time < time_limit)
    print(bcolors.GREEN + bcolors.BOLD + "Speed seems good!" + bcolors.ENDC)
except AssertionError:
    print(bcolors.RED + "Speed test failed! 💀" + bcolors.ENDC)
    exit(1)