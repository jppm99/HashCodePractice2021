# #1 is around 27.294.000

import timeit
import operator
import random
import numpy as np

from multiprocessing import Pool

from simanneal import Annealer

rel_path = 'C:\GitHub\HashCodePractice2021\quali-2020\\'

path_a = 'a_example.txt'
path_b = 'b_read_on.txt'
path_c = 'c_incunabula.txt'
path_d = 'd_tough_choices.txt'
path_e = 'e_so_many_books.txt'
path_f = 'f_libraries_of_the_world.txt'

class SimAnnealSolver(Annealer):
    def __init__(self, initial_state, libraries, maxDays):
        self.libraries = libraries
        self.maxDays = maxDays
        self.orderCache = []
        self.energyCache = 0

        super(SimAnnealSolver, self).__init__(initial_state)
    
    def move(self):
        initial_energy = self.energy()

        # at least one of them must be in the 'useful' library range
        usefulLimit = 0
        days = 0
        for i in range(len(self.state)):
            days += self.libraries[self.state[i], 2]
            if days >= self.maxDays:
                usefulLimit = i
                break

        a = randInt(usefulLimit)
        b = randInt(len(self.state))
        self.state[a], self.state[b] = self.state[b], self.state[a]

        return self.energy() - initial_energy
    
    def energy(self):
        if self.state == self.orderCache:
            return self.energyCache
        else:
            new_energy = -scoreOrder(self.state, self.libraries, self.maxDays)
            self.orderCache = self.state[:]
            self.energyCache = new_energy
            return new_energy

def SolveA(path):
    return 0

def SolveGeneral(path, steps):
    abs_path = rel_path + 'in\\' + path

    totalScore = 0

    #
    # ------------------- LOAD DATA FROM FILE ---------------------
    #

    with open(abs_path, 'r') as f:
        [nBooks, nLibs, nDays] = [int(el) for el in f.readline().split()]

        books = np.empty((nBooks, 2), dtype=np.int64) # [i, points]
        libraries = np.empty((nLibs, nBooks + 4), dtype=np.int64) # [i, n books, ndays, books/day, book1, book2, ...]

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
            for ii in range(len(b_s)):
                libraries[i, ii + 4] = int(b_s[ii])

    #
    # ------------------- SORTING THE NUMPY ARRAY -------------------
    #

    pred = sortingWeightFun(arr=libraries, reversed=True) # Reverse probably doesn't work with strings
    order = np.argsort(pred) # array of indexes in order

    #
    # ------------- CHOOSING THE RIGHT ORDER OF LIBS ----------------
    #

    initial_guess = [i for i in range(len(libraries))]
    #initial_guess = list(order)
    #initial_guess = [47, 76, 79, 3, 22, 5, 8, 87, 78, 9, 54, 7, 12, 57, 83, 74, 55, 51, 20, 80, 48, 62, 18, 85, 27, 35, 11, 25, 28, 71, 30, 75, 65, 33, 63, 73, 84, 81, 38, 23, 41, 26, 69, 24, 44, 93, 16, 50, 2, 14, 29, 10, 52, 53, 1, 4, 6, 13, 0, 72, 60, 61, 19, 94, 89, 77, 15, 99, 68, 40, 70, 21, 39, 32, 42, 31, 95, 66, 92, 43, 67, 37, 59, 45, 46, 82, 86, 17, 88, 64, 90, 91, 58, 49, 34, 36, 96, 97, 98, 56]
    
    solver = SimAnnealSolver(initial_guess, libraries, nDays)
    solver.copy_strategy = 'slice'

    solver.Tmax = 75000
    solver.Tmin = 500

    #print('getting scheduler')
    #solver.set_schedule(solver.auto(minutes=0.1))

    solver.steps = steps
    solver.updates = int(steps/2)
    
    print('started annealing')
    (libsByOrder, fitness) = solver.anneal()
    # print('done -> ')
    # print(libsByOrder)

    #
    # ---------------------- GETTING SOLUTION? ----------------------
    #

    print('Choosing books')
    booksDict = {}
    for i in range(nBooks):
        booksDict[i] = books[i, 1]
    #booksDict = sortDictByValue(booksDict)

    solution = {}
    currDay = 0
    for lib in libsByOrder:
        currDay += libraries[lib, 2]
        booksToScan = (nDays - currDay) * libraries[lib, 3]

        booksValues = {}

        count = 0
        for b in libraries[lib, 4:]:
            count += 1
            booksValues[b] = booksDict[b]
            if count >= libraries[lib, 1]:
                break
        booksValues = sortDictByValue(booksValues, reversed=True)
        
        solution[lib] = []
        count = 0
        for key in list(booksValues.keys())[:booksToScan]:
            count += 1
            totalScore += booksDict[key]
            booksDict[key] = 0
            solution[lib].append(key)
            if count >= libraries[lib, 1]:
                break
    
    return totalScore


def sortingWeightFun(arr, reversed):
    if(reversed):
        return -arr[:, 0]
    else:
        return arr[:, 0]

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
        ret = SolveGeneral(args['path'], 20000)
    
    elif(args['type'] == 'C'):
        ret = SolveGeneral(args['path'], 200)
    
    elif(args['type'] == 'D'):
        ret = SolveGeneral(args['path'], 30)
    
    elif(args['type'] == 'E'):
        ret = SolveGeneral(args['path'], 25000)
    
    elif(args['type'] == 'F'):
        ret = SolveGeneral(args['path'], 35000)
    
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
            {'path': path_e, 'type': 'E'},
            {'path': path_f, 'type': 'F'}
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
