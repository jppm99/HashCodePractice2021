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

def SolveA(path):
    return 0

def SolveGeneral(zip_arg):
    path, vars_opt = zip_arg #use with pool
    #path, vars_opt = list(zip_arg)[0] #use when without pool

    abs_path = rel_path + 'in\\' + path

    totalScore = 0

    #
    # ------------------- LOAD DATA FROM FILE ---------------------
    #

    with open(abs_path, 'r') as f:
        [nBooks, nLibs, nDays] = [int(el) for el in f.readline().split()]

        books = np.empty((nBooks, 2), dtype=np.int32) # [i, points]
        libraries = np.empty((nLibs, nBooks + 4), dtype=np.int32) # [i, n books, ndays, books/day, book1, book2, ...]

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

    pred_res = np.array([sortingWeightFun(elem=e, totalDays=nDays, reversed=True, vars_opt=vars_opt) for e in libraries]) # Reverse probably doesn't work with strings
    libsByOrder = np.argsort(pred_res, kind='stable') # array of indexes in order

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
        booksToScan = (nDays - currDay) * np.int64(libraries[lib, 3])
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


def sortingWeightFun(elem, totalDays, reversed, vars_opt):
    books = min(elem[1], elem[3] * (totalDays / 2 - elem[2]))
    sum_books = sum(elem[4:(4+int(books))])
    avg_books = sum_books / books

    fitness = elem[3] * vars_opt[0] + sum_books * vars_opt[1] + avg_books * vars_opt[2]
    
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
        func = SolveGeneral
        path = args['path']
        initial_guess = [1, 1, 1]
        
    elif(args['type'] == 'B'):
        func = SolveGeneral
        path = args['path']
        initial_guess = [1, 1, 1]
    
    elif(args['type'] == 'C'):
        func = SolveGeneral
        path = args['path']
        initial_guess = [1, 1, 1]
    
    elif(args['type'] == 'D'):
        func = SolveGeneral
        path = args['path']
        initial_guess = [1, 1, 1]
    
    elif(args['type'] == 'E'):
        func = SolveGeneral
        path = args['path']
        initial_guess = [1, 1, 1]
    
    elif(args['type'] == 'F'):
        func = SolveGeneral
        path = args['path']
        initial_guess = [1, 1, 1]

    # pool
    rand_range = 4
    nThreads = 5
    while rand_range > 0.2:
        with Pool(nThreads) as pool:
            attemps = [initial_guess]
            for i in range(1, nThreads):
                attemps.append([0 for _ in initial_guess])
                for ii in range(len(initial_guess)):
                    attemps[i][ii] = initial_guess[ii] * (random.random() * rand_range - rand_range / 2 + 1)
            path_list = [path for _ in attemps]
            new_args = zip(path_list, attemps)
            #print(list(new_args))
            scores = pool.map(SolveGeneral, new_args)

            max_index = indexOfMax(scores)
            score = scores[max_index]
            initial_guess = attemps[max_index]
            print('' + args['type'] + ' curr max -> ' + str(score) + ' with ', initial_guess)
        rand_range /= 1.3

    # simple exec, no pool
    # new_args = zip([path], [initial_guess])
    # score = func(new_args)
    
    print('' + args['type'] + ' -> ' + str(score))
    return score


def main():
    tic = timeit.default_timer()

    scores = []

    scores.append(sol({'path': path_a, 'type': 'A'}))
    scores.append(sol({'path': path_b, 'type': 'B'}))
    scores.append(sol({'path': path_c, 'type': 'C'}))
    scores.append(sol({'path': path_d, 'type': 'D'}))
    scores.append(sol({'path': path_e, 'type': 'E'}))
    scores.append(sol({'path': path_f, 'type': 'F'}))

    print('\n\nFinal score -> %d' % sum(scores))

    # with Pool(8) as pool:
    #     scores = pool.map(sol, [
    #         {'path': path_a, 'type': 'A'}, 
    #         {'path': path_b, 'type': 'B'},
    #         {'path': path_c, 'type': 'C'},
    #         {'path': path_d, 'type': 'D'},
    #         {'path': path_e, 'type': 'E'},
    #         {'path': path_f, 'type': 'F'}
    #         ])
    #     print('\n\nFinal score -> %d' % sum(scores))
    #     print(scores)

    toc = timeit.default_timer()
    elapsed = toc - tic
    print('\ntook ' + str(int(elapsed / 60)) + ' minutes and ' + str(elapsed % 60) + ' seconds')

class SimAnnealSolver(Annealer):
    def __init__(self, initial_state, path):
        self.path = path

        self.stateCache = []
        self.energyCache = 0

        super(SimAnnealSolver, self).__init__(initial_state)
    
    def move(self):
        initial_energy = self.energy()

        i = randInt(len(self.state))
        mult = random.random() * 2 #[0, 2[
        
        self.state[i] = mult * self.state[i]

        return self.energy() - initial_energy
    
    def energy(self):
        if self.state == self.stateCache:
            return self.energyCache
        else:
            new_energy = -1 * SolveGeneral(args['path'], initial_guess[:])
            
            self.stateCache = self.state
            self.energyCache = new_energy
            return new_energy
    
    # def update(self, *args, **kwargs):
    #     return

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
