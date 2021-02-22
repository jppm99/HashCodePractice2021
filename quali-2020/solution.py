# #1 is around 27.294.000

import timeit
import operator
import numpy as np
from multiprocessing import Pool
from ortools.sat.python import cp_model

rel_path = 'C:\GitHub\HashCodePractice2021\quali-2020\\'

path_a = 'a_example.txt'
path_b = 'b_read_on.txt'
path_c = 'c_incunabula.txt'
path_d = 'd_tough_choices.txt'
path_e = 'e_so_many_books.txt'
path_f = 'f_libraries_of_the_world.txt'

def sol(path):
    abs_path = rel_path + 'in\\' + path

    totalScore = 0

    #
    # ------------------- LOAD DATA FROM FILE ---------------------
    #

    with open(abs_path, 'r') as f:
        [nBooks, nLibs, nDays] = [int(el) for el in f.readline().split()]

        books = np.empty((nBooks, 2), dtype=np.int16) # [i, points]
        libraries = np.empty((nLibs, nBooks + 4), dtype=np.int16) # [i, n books, ndays, books/day, book1, book2, ...]

        books_split = f.readline().split()
        for i in range(len(books_split)):
            books[i] = np.array([i] + [int(books_split[i])])

        for i in range(nLibs):
            [nB, nD, sR] = [int(el) for el in f.readline().split()]
            libraries[i, 0] = i
            libraries[i, 1] = nB
            libraries[i, 2] = nD
            libraries[i, 3] = sR

            b_s = f.readline().split()
            for ii in range(0, len(b_s)):
                libraries[i, ii + 3] = int(b_s[ii])

    #
    # ------------------- SORTING THE NUMPY ARRAY -------------------
    #

    pred = sortingWeightFun(arr=libraries, reversed=True) # Reverse probably doesn't work with strings
    order = np.argsort(pred) # array of indexes in order

    #
    # ------------------------ OR-TOOLS STUFF ------------------------
    #

    model = cp_model.CpModel()

    libIsInPos = {}
    for i in range(nLibs):
        libIsInPos[i] = model.NewIntVar(-1, nLibs, 'lib_%i' % i)

    # Constraint function
    model.Add(isOrderValid(libIsInPos, libraries, nDays))
    
    model.Maximize(scoreOrder(libIsInPos, libraries))

    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = 60

    status = solver.Solve(model)
    print(solver.StatusName(status))
    

    #
    # ------------------------ GETTING SOLUTION? ------------------------
    #

    libsPos = {}
    for key in libIsInPos:
        libsPos[key] = solver.Value(libIsInPos[key])
    
    libsPos = sortDictByValue(libsPos)
    print(libsPos)
    keysToDelete = []
    for key in libsPos:
        if libsPos[key] == -1:
            keysToDelete.append(key)
    for key in keysToDelete:
        libsPos.pop(key)
    
    #libsInOrder = []
    print(libsPos)




def sortingWeightFun(arr, reversed):
    if(reversed):
        return -arr[:, 0]
    else:
        return arr[:, 0]

def scoreOrder(libDict, libraries):
    fitness = 0
    day = 0
    sortedDict = sortDictByValue(libDict)

    for key in sortedDict:
        if sortedDict[key] == -1:
            continue
    
        fitness += scoreLocalLib(libraries[key], day)
        day += libraries[key, 2]
    
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

#
#   ------------------------- HELPER FUNCTIONS -------------------------
#

# ---------------- SORTING DICTIONARY ---------------
def sortDictByValue(dict):
    return {
        k: v for k, v in sorted(
            dict.items(),
            #key=lambda item: item[1]
            key=operator.itemgetter(1) # same as last but more efficient
    )}
# ---------------------------------------------------

# ---------------- TIMSORT FOR [] ---------------
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
# --------------------------------------------


def main():
    tic = timeit.default_timer()

    # sol(path_a)
    # sol(path_b)
    # sol(path_c)
    # sol(path_d)
    # sol(path_e)
    sol(path_f)

    # with Pool(6) as pool:
    #     pool.map(sol, [path_a, path_b, path_c, path_d, path_e, path_f])

    toc = timeit.default_timer()
    print('took ' + str(toc - tic) + ' seconds')


if __name__ == "__main__":
    main()
