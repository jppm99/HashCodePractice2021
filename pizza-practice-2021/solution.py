import timeit
import operator

rel_path = 'C:\GitHub\HashCodePractice2021\pizza-practice-2021\\'

path_a = 'a_example.in'
path_b = 'b_little_bit_of_everything.in'
path_c = 'c_many_ingredients.in'
path_d = 'd_many_pizzas.in'
path_e = 'e_many_teams.in'


def sol(path):
    abs_path = rel_path + 'in\\' + path

    totalScore = 0
    pizzas = [] # [i, nIngredients, ingredients..]
    ordered = []

    with open(abs_path, 'r') as f:
        [nPizzas, teams2, teams3, teams4] = [int(el) for el in f.readline().split()]

        for i in range(nPizzas):
            pizzas.append(
                [i] + [el for el in f.readline().split()]
            )
    
    pizzas.sort(key=operator.itemgetter(1), reverse=True)

    ordered.append(pizzas.pop(0))
    index = 0
    max_ingredients = ordered[0][1]

    while len(pizzas) > 0:
        maximum = ordered[index][1] + max_ingredients
        currMax = 0
        currIndex = 0

        step = 1
        if len(pizzas) > 10:
                step = 1

        for i in range(0, len(pizzas), step):

            if int(pizzas[i][1]) > currMax:
                dif = difIngredients(pizzas[i], ordered[index])
                
                if dif == maximum:
                    currMax = dif
                    currIndex = i
                    break

                if dif > currMax:
                    currMax = dif
                    currIndex = i
        
        ordered.append(pizzas.pop(currIndex))
    
    n_ordered = len(ordered)
    deliveries = 0
    
    n4 = min(n_ordered / 4, teams4)
    deliveries += n4
    n_ordered -= n4 * 4
    
    n3 = min(n_ordered / 3, teams3)
    deliveries += n3
    n_ordered -= n3 * 3
    
    n2 = min(n_ordered / 2, teams2)
    deliveries += n2

    with open(rel_path + 'out\\' + path, 'w') as f:
        f.write(str(int(deliveries)) + '\n')

        while teams4 > 0 and len(ordered) > 4:
            teams4 -= 1
            
            p1 = ordered.pop(0)
            p2 = ordered.pop(0)
            p3 = ordered.pop(0)
            p4 = ordered.pop(0)

            difs = len(set(p1[2:] + p2[2:] + p3[2:] + p4[2:]))
            totalScore += difs * difs

            f.write('4 ' + str(p1[0]) + ' ' + str(p2[0]) + ' ' + str(p3[0]) + ' ' + str(p4[0]) + '\n')
        
        while teams3 > 0 and len(ordered) > 3:
            teams3 -= 1
            
            p1 = ordered.pop(0)
            p2 = ordered.pop(0)
            p3 = ordered.pop(0)

            difs = len(set(p1[2:] + p2[2:] + p3[2:]))
            totalScore += difs * difs

            f.write('3 ' + str(p1[0]) + ' ' + str(p2[0]) + ' ' + str(p3[0]) + '\n')

        while teams2 > 0 and len(ordered) > 2:
            teams2 -= 1
            
            p1 = ordered.pop(0)
            p2 = ordered.pop(0)

            difs = len(set(p1[2:] + p2[2:]))
            totalScore += difs * difs

            f.write('2 ' + str(p1[0]) + ' ' + str(p2[0]) + '\n')

    print(path + ' -> score: ' + str(totalScore))


        
def difIngredients(p1, p2):
    return len(set(p1[2:] + p2[2:]))


def main():
    tic = timeit.default_timer()
    
    sol(path_a)
    sol(path_b)
    sol(path_c)
    sol(path_d)
    sol(path_e)

    toc = timeit.default_timer()
    print('took ' + str(toc - tic) + ' seconds')


if __name__ == "__main__":
    main()
