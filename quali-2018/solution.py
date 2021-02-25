# #1 is around 27.294.000

import timeit
import operator
import random
import numpy as np

from multiprocessing import Pool

from simanneal import Annealer

rel_path = 'C:\GitHub\HashCodePractice2021\quali-2020\\'

path_a = 'a_example.in'
path_b = 'b_should_be_easy.in'
path_c = 'c_no_hurry.in'
path_d = 'd_metropolis.in'
path_e = 'e_high_bonus.in'

def SolveA(path):
    return 0

def SolveGeneral(path, steps):
    abs_path = rel_path + 'in\\' + path

    totalScore = 0

    #
    # ------------------- LOAD DATA FROM FILE ---------------------
    #

    with open(abs_path, 'r') as f:
        [rows, cols, nCars, nRides, startOnTimeBonus, time] = [int(el) for el in f.readline().split()]

        rides = np.empty((nRides, 6), dtype=np.int64) # [starting row, starting col, ending row, ending col, earliest start, latest finish]

        line_split = f.readline().split()
        for i in range(nRides):
            for ii in range(6):
                rides[i, ii] = line_split[ii]

    #
    # ------------------- SORTING THE NUMPY ARRAY -------------------
    #

    #pred = sortingWeightFun(arr=libraries, totalDays=nDays, reversed=True) # Reverse probably doesn't work with strings
    #libsByOrder = np.argsort(pred) # array of indexes in order

    #
    # ---------------------- GETTING SOLUTION? ----------------------
    #

    print('Calc sol')
    usedCars = 0
    ridesDone = 0

    while usedCars < nCars and ridesDone < nRides:
        break
    
    return totalScore


def sortingWeightFun(arr, totalDays, reversed):
    #nb = min(((totalDays / 1.5) - arr[:, 2]) * arr[:, 3], arr[:, 1])
    #fitness = sum(arr[:, 4:(4+nb)])

    fitness = arr[:, 0]
    
    if(reversed):
        return -fitness
    else:
        return fitness

def scoreOrder(order, libraries, maxDays):
    fitness = 0
    day = 0
    
    for lib in order:
        if day >= maxDays:
            break

        fitness += scoreLocalLib(libraries[lib], maxDays - day)
        day += libraries[lib, 2]
    
    return fitness


def scoreLocalLib(lib, daysToDeadline):
    daysToScan = daysToDeadline - lib[2]
    booksToScan = daysToScan * lib[3]
    return sum(lib[4:booksToScan+4])

def isOrderValid(libDict, libraries, maxDays):
    daysSum = 0
    curPos = 0

    sortedDict = sortDictByValue(libDict)

    for libKey in sortedDict:
        if sortedDict[libKey] == -1:
            continue

        daysSum += libraries[libKey][2]
        curPos += 1

        if curPos != (sortedDict[libKey] - 1) or daysSum > maxDays:
            return False
    
    return True

def sol(args):
    if(args['type'] == 'A'):
        ret = SolveGeneral(args['path'], 400)
    
    elif(args['type'] == 'B'):
        ret = SolveGeneral(args['path'], 2000)
    
    elif(args['type'] == 'C'):
        ret = SolveGeneral(args['path'], 20)
    
    elif(args['type'] == 'D'):
        ret = SolveGeneral(args['path'], 5)
    
    elif(args['type'] == 'E'):
        ret = SolveGeneral(args['path'], 2500)
    
    print('' + args['type'] + ' -> ' + str(ret))
    return ret


def main():
    tic = timeit.default_timer()

    with Pool(8) as pool:
        scores = pool.map(sol, [
            {'path': path_a, 'type': 'A'}, 
            {'path': path_b, 'type': 'B'},
            {'path': path_c, 'type': 'C'},
            {'path': path_d, 'type': 'D'},
            {'path': path_e, 'type': 'E'}
            ])
        print('\n\nFinal score -> %d' % sum(scores))
        print(scores)

    toc = timeit.default_timer()
    elapsed = toc - tic
    print('\ntook ' + str(int(elapsed / 60)) + ' minutes and ' + str(elapsed % 60) + ' seconds')


#
#   ------------------------- HELPER FUNCTIONS -------------------------
#

# --------------- INDEX OF LIST MAX -----------------
def indexOfMax(list):
    """"
    Get index of Max element of list/numpy array
    """
    max = -np.Infinity
    index = 0
    i = 0
    for value in list:
        if value > max:
            max = value
            index = i
        i += 1
    return index
# ---------------------------------------------------

# ------------------ FAST RANDINT -------------------
def randInt(max):
    """
    Returns random integer in range [0, max[
    """
    return int(max * random.random())
# ---------------------------------------------------

# ---------------- SORTING DICTIONARY ---------------
def sortDictByValue(dict, reversed):
    """
    Ascending order
    """
    if not reversed:
        reversed = False
    
    return {
            k: v for k, v in sorted(
                dict.items(),
                reverse=reversed,
                #key=lambda item: item[1]
                key=operator.itemgetter(1) # same as last but more efficient
        )}
# ---------------------------------------------------

# ------------------ TIMSORT FOR [] -----------------
MIN_MERGE = 32

def calcMinRun(n): 
	r = 0
	while n >= MIN_MERGE: 
		r |= n & 1
		n >>= 1
	return n + r 

def insertionSort(arr, left, right): 
	for i in range(left + 1, right + 1): 
		j = i 
		while j > left and arr[j] < arr[j - 1]: 
			arr[j], arr[j - 1] = arr[j - 1], arr[j] 
			j -= 1

def merge(arr, l, m, r): 
	len1, len2 = m - l + 1, r - m 
	left, right = [], [] 
	for i in range(0, len1): 
		left.append(arr[l + i]) 
	for i in range(0, len2): 
		right.append(arr[m + 1 + i]) 

	i, j, k = 0, 0, l 

	while i < len1 and j < len2: 
		if left[i] <= right[j]: 
			arr[k] = left[i] 
			i += 1
		else: 
			arr[k] = right[j] 
			j += 1

		k += 1

	while i < len1: 
		arr[k] = left[i] 
		k += 1
		i += 1

	while j < len2: 
		arr[k] = right[j] 
		k += 1
		j += 1

def timSort(arr): 
	n = len(arr) 
	minRun = calcMinRun(n) 
	
	for start in range(0, n, minRun): 
		end = min(start + minRun - 1, n - 1) 
		insertionSort(arr, start, end) 

	size = minRun 
	while size < n: 
		for left in range(0, n, 2 * size):
			mid = min(n - 1, left + size - 1) 
			right = min((left + 2 * size - 1), (n - 1)) 
			merge(arr, left, mid, right) 

		size = 2 * size 
# -------------------------------------------------


if __name__ == "__main__":
    main()
