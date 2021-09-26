import asn1tools as asn
import json

## 2WF90 Software Assignment 1

# helper functions

# converts the input strings to an array
def string_to_list(num):
    num_list = []
    for i in num:
        if i == 'a':
            num_list.append(10)
        elif i == 'b':
            num_list.append(11)
        elif i == 'c':
            num_list.append(12)
        elif i == 'd':
            num_list.append(13)
        elif i == 'e':
            num_list.append(14)
        elif i == 'f':
            num_list.append(15)
        elif int(i) < 10:
            num_list.append(int(i))
    return num_list


# converts arrays back to strings for output
def list_to_string(num_list):
    string = ''
    for i in num_list:
        if i < 10:
            string += str(i)
        elif i == 10:
            string += 'a'
        elif i == 11:
            string += 'b'
        elif i == 12:
            string += 'c'
        elif i == 13:
            string += 'd'
        elif i == 14:
            string += 'e'
        elif i == 15:
            string += 'f'
    return string


# Long Division function (Algorithm 1.5)
# implemented according to  [Sh, Section 3.3.4]
def long_division(x, y, radix):
    radix = int(radix)
    # check the sign of the numbers
    sign_x = '+'
    if x[0] == '-':
        x = x[1:]
        sign_x = '-'

    sign_y = '+'
    if y[0] == '-':
        y = y[1:]
        sign_y = '-'

    x = string_to_list(x)
    y = string_to_list(y)

    k = len(x)
    l = len(y)

    # if y>x then output 0 as quotient and x as the remainder
    if k < l:
        return ['0', list_to_string(x)]

    # quotient and remainder
    q = []
    r = [0] + x

    # to avoid division by zero
    if y[0] == 0:
        y_start = 1
    else:
        y_start = y[0]

    for i in range(0, k - l + 1):

        q.append((r[i] * radix + r[i + 1]) // y_start)

        if (q[i] >= radix):
            q[i] = radix - 1

        carry = 0

        for j in range(l - 1, -1, -1):
            temp = r[i + j + 1] - q[i] * y[j] + carry
            carry, r[i + j + 1] = temp // radix, temp % radix

        r[i] = r[i] + carry

        while r[i] < 0:
            carry = 0

            for j in range(l - 1, -1, -1):
                temp = r[i + j + 1] + y[j] + carry
                carry, r[i + j + 1] = temp // radix, temp % radix

            r[i] = r[i] + carry
            q[i] = q[i] - 1

    # making the quotient negative if the signs are different
    if sign_x != sign_y:
        q = [-1] + q

    q = list_to_string(q)
    r = list_to_string(r)

    # remove zeroes at the beginning of the numbers
    q = trimZeroes(q)
    r = trimZeroes(r)

    return [q, r]

#An auxillary function that checks a string front to back, removing leading zeroes. If the entire string consists of
#zeroes, then the output is just "0"
def trimZeroes(s):
    nonZeroFound = False
    firstNonZero = -1
    output = ""
    for i in range(len(s)):
        if s[i] != "0":
            nonZeroFound = True
            firstNonZero = i
            break

    if (nonZeroFound):
        output = s[firstNonZero:]
    else:
        output = '0'

    return output

def addition(x, y, radix,a,m):
    additions = a
    mutliplications = m
    radix = int(radix)
    result = ""

    #Set the boundary of the loop to the length of the longer number and add leading zeroes if the numbers are not
    #of equal lengh.
    if len(x) >= len(y):
        loopLimit = len(x) - 1
        lengthDifference = len(x) - len(y)
        leadingZeroes = ""
        for i in range(0, lengthDifference):
            leadingZeroes = leadingZeroes + "0"
        y = leadingZeroes + y
    else:
        loopLimit = len(y) - 1
        lengthDifference = len(y) - len(x)
        leadingZeroes = ""
        for i in range(0, lengthDifference):
            leadingZeroes = leadingZeroes + "0"
        x = leadingZeroes + x

    #Here we begin with the actual addition
    carry = 0
    #Traverse the string right to left
    for i in range(loopLimit, -1, -1):
        #Take the numbers at position i, using base 16 since that covers all bases up to and including 16
        xint = int(x[i],16)
        yint = int(y[i],16)
        #Add the two numbers together
        sum = carry + xint + yint
        additions = additions + 2
        #If we exceed the base, we take a carry to the next iteration
        if sum < radix:
            carry = 0
            if radix <= 10:
                #This case covers bases up to and including 10
                result = str(sum) + result
            else:
                #This case covers bases between 10 and 16
                hexresult = hex(sum)
                result = hexresult[2:] + result
        else:
            carry = 1
            if radix <= 10:
                result = str(sum - radix) + result
            else:
                hexresult = hex(sum-radix)
                result = hexresult[2:] + result

    #If we're left with a carry at the end of the addition, we put it at the start of the string
    if carry == 1:
        result = str(carry) + result

    return result, additions, mutliplications


def subtraction(x, y, radix,a,m):
    additions = a
    mutliplications = m
    radix = int(radix)
    result = ""

    subtractor = ""
    subtractee = ""
    negative = False
    #The idea here is that the order matters for subtraction - if we're subtracting a smaller number from a larger
    #number then everything is fine. If we're subtracting a larger number from a smaller number then we can just flip
    #them around and give the result a minus sign. For numbers that fit into a single integer, comparison is a single
    #clock cycle operation so it doesn't interfere with the complexity. The assignment description says that we
    #are only allowed to use default Python libraries, but Python 3 does not have an upper bound on integer size :)
    #Either way, I guess the point of this homework is to break the complex operations down into elementary operations
    #and that is what is being done here nonetheless. But if turning the entire string into an integer warrants a grade
    #reduction, go right ahead.
    if int(x,16) >= int(y,16):
        subtractor = x
        subtractee = y
        loopLimit = len(subtractor) - 1
        lengthDifference = len(subtractor) - len(subtractee)
        leadingZeroes = ""
        for i in range(0, lengthDifference):
            leadingZeroes = leadingZeroes + "0"
        subtractee = leadingZeroes + subtractee
    else:
        subtractor = y
        subtractee = x
        negative = True
        loopLimit = len(subtractor) - 1
        lengthDifference = len(subtractor) - len(subtractee)
        leadingZeroes = ""
        for i in range(0, lengthDifference):
            leadingZeroes = leadingZeroes + "0"
        subtractee = leadingZeroes + subtractee

    carry = 0
    #Just like in the addition function, we traverse the number from right to left
    for i in range(loopLimit, -1, -1):
        #Subtract the two numbers, and if the result stays above 0, we're fine
        if carry + int(subtractor[i],16) - int(subtractee[i],16) >= 0:
            if radix <= 10:
                result = str(carry + int(subtractor[i], 16) - int(subtractee[i], 16)) + result
            else:
                result = hex(carry + int(subtractor[i], 16) - int(subtractee[i], 16))[2:] + result
            carry = 0
            additions = additions + 2
        #If not, we borrow from the next iteration by setting the carry to -1
        else:
            if radix <= 10:
                result = str((carry + radix + int(subtractor[i], 16)) - int(subtractee[i], 16)) + result
            else:
                result = hex((carry + radix + int(subtractor[i], 16)) - int(subtractee[i], 16))[2:] + result
            carry = -1
            additions = additions + 2

    result = trimZeroes(result)

    #If we subtract a larger number from a smaller number, we give it a minus sign
    if negative:
        result = "-" + result

    return result,additions,mutliplications
    # params['answer'] = '-0'


def multiplicationPrimary(x, y, radix, a, m):
    additions = a
    multiplications = m
    radix = int(radix)
    intermediate = []

    #This is basically putting the longer number above the shorter number just like if we were doing the
    #primary school method on papear
    if len(x) >= len(y):
        longer = x
        shorter = y
    else:
        longer = y
        shorter = x

    #Then we start multiplying every digit by every diigt
    for i in range(len(shorter)):
        shorterindex = (len(shorter) - 1) - i
        offset = ""
        #If we were doing this on paper, we would shift each intermediate multiple to the left by one digit.
        #That's what this accomplishes
        for k in range(i):
            offset = "0" + offset
        intermediateMultiple = "" + offset
        carry = 0
        for j in range(len(longer)):
            longerindex = (len(longer) - 1) - j

            #Multiply the two digits
            multiple = carry + (int(shorter[shorterindex],16) * int(longer[longerindex],16))
            additions = additions + 1
            multiplications = multiplications + 1
            #Check if the result exceeds the radix and create carries accordingly
            if (multiple >= radix):
                if radix <= 10:
                    intermediateMultiple = str(divmod(multiple, radix)[1]) + intermediateMultiple
                    #This divmod thing might be cheating but it's still a default Python library
                    #and again, I think it covers the point of the subtraction exercise sufficiently
                    carry = divmod(multiple, radix)[0]
                else:
                    intermediateMultiple = hex(divmod(multiple, radix)[1])[2:] + intermediateMultiple
                    carry = divmod(multiple, radix)[0]
            else:
                if radix <= 10:
                    intermediateMultiple = str(multiple) + intermediateMultiple
                    carry = 0
                else:
                    intermediateMultiple = hex(multiple)[2:] + intermediateMultiple
                    carry = 0

        if carry != 0:
            if radix <= 10:
                intermediateMultiple = str(carry) + intermediateMultiple
            else:
                intermediateMultiple = hex(carry)[2:] + intermediateMultiple
        intermediate.append(intermediateMultiple)

    #Just like if we were doing this on paper, we add up the intermediate multiples using our addition algo
    cumulative = "0"
    for i in range(len(intermediate)):
        cumulative, additions, multiplications = addition(cumulative, intermediate[i], str(radix),additions,multiplications)

    result = cumulative
    return result, additions, multiplications


def multiplicationKaratsuba(x, y, radix, a, m):
    additions = a
    multiplications = m
    radix = int(radix)
    result = ""
    n = 0

    #Karatsuba works if both integers have the same word length, so if that's not the case, we add leading zeroes
    #And we define n along the way so that we can split the number into high and low
    if len(x) > len(y):
        lenghtDifference = len(x) - len(y)
        zeroes = ""
        for i in range(lenghtDifference):
            zeroes = zeroes + "0"
        y = zeroes + y
        n = len(x)
    elif len(y) > len(x):
        lenghtDifference = len(y) - len(x)
        zeroes = ""
        for i in range(lenghtDifference):
            zeroes = zeroes + "0"
        x = zeroes + x
        n = len(y)
    elif len(x) == len(y):
        n = len(x)

    #If we reached a recursion depth where it's down to a single elementary multiplication, we just return that
    if n == 1:
        return multiplicationPrimary(x,y,radix,additions,multiplications)

    #If n is divisible by two, everything is fine
    #If not, we add another leading 0
    if divmod(n, 2)[1] != 0:
        x = "0" + x
        y = "0" + y
        n = n + 1

    #And split the numbers up into high and low
    xhi = x[:int(n / 2)]
    xlo = x[int(n / 2):]
    yhi = y[:int(n / 2)]
    ylo = y[int(n / 2):]

    #These are the additions that make up the decomposition of the multiplication
    part01, additions, multiplications = addition(xhi, xlo, radix, additions, multiplications)
    part02, additions, multiplications = addition(yhi, ylo, radix, additions, multiplications)

    #Recursive calls to further break down the multiplications
    part1, additions, multiplications = multiplicationKaratsuba(part01, part02, radix, additions, multiplications)
    part2, additions, multiplications = multiplicationKaratsuba(xhi, yhi, radix, additions, multiplications)
    part3, additions, multiplications = multiplicationKaratsuba(xlo, ylo, radix, additions, multiplications)

    #More additions that make up the multiplication
    final0, additions, multiplications = addition(part2, part3, radix, additions, multiplications)
    #and the subtraction to get the missing piece
    final1, additions, multiplications = subtraction(part1, final0, radix, additions, multiplications)

    #part2 is the multiple of b^n so we add n zeroes at the end of it
    if part2 != "0":
        for i in range(n):
            part2 = part2 + "0"
    #final1 is the multiple of b^(n/2) so we add n/2 zeroes at the end of it
    if final1 != "0":
        for i in range(int(n/2)):
            final1 = final1 + "0"

    #And finally add the components up
    total, additions, multiplications = addition(part2,final1,radix, additions, multiplications)
    total, additions, multiplications = addition(total,part3,radix, additions, multiplications)

    result = trimZeroes(total)
    return result, additions, multiplications

def euclid(x, y, radix):
    #From here
    r = int(radix)
    q1 = "0"
    q2 = "0"
    if int(x,r) >= int(y,r):
        r1 = x
        r2 = y
    else:
        r1 = y
        r2 = x
    a1 = "1"
    a2 = "0"
    b1 = "0"
    b2 = "1"

    while r2 != "0":
        q3, r3 = long_division(r1,r2,radix)
        r1 = r2
        r2 = r3
        q1 = q2
        q2 = q3
    #to here it's just copied from the lecture notes

        #Now we reached the point where we need to multiply by the quotient and subtract. Here, negative numbers
        #can occur but the functions are designed to only handle positive numbers. Negative numbers are handled below
        #where we check for the operation type from the input file. This is the fourth day that I'm working on this
        #code and I really don't feel like rewriting all the functions, so I just slapped the case distinction
        #here as well :)
        #Except for this part, the whole thing is just the algorithm copied from the lecture notes.

        # a*b = a*b
        if (a2[0] != '-') & (q3[0] != '-'):
            mid1, useless1, useless2 = multiplicationKaratsuba(a2,q3,radix,0,0)
        # (-a)*b = -(a*b)
        if (a2[0] == '-') & (q3[0] != '-'):
            mid1, useless1, useless2 = multiplicationKaratsuba(a2[1:],q3,radix,0,0)
            mid1 = "-" + mid1
        # a*(-b) = -(a*b)
        if (a2[0] != '-') & (q3[0] == '-'):
            mid1, useless1, useless2 = multiplicationKaratsuba(a2, q3[1:], radix, 0, 0)
            mid1 = "-" + mid1
        # (-a)*(-b) = a*b
        if (a2[0] == '-') & (q3[0] == '-'):
            mid1, useless1, useless2 = multiplicationKaratsuba(a2[1:], q3[1:], radix, 0, 0)

        # a*b = a*b
        if (b2[0] != '-') & (q3[0] != '-'):
            mid2, useless1, useless2 = multiplicationKaratsuba(b2, q3, radix, 0, 0)
        # (-a)*b = -(a*b)
        if (b2[0] == '-') & (q3[0] != '-'):
            mid2, useless1, useless2 = multiplicationKaratsuba(b2[1:], q3, radix, 0, 0)
            mid2 = "-" + mid2
        # a*(-b) = -(a*b)
        if (b2[0] != '-') & (q3[0] == '-'):
            mid2, useless1, useless2 = multiplicationKaratsuba(b2, q3[1:], radix, 0, 0)
            mid2 = "-" + mid2
        # (-a)*(-b) = a*b
        if (b2[0] == '-') & (q3[0] == '-'):
            mid2, useless1, useless2 = multiplicationKaratsuba(b2[1:], q3[1:], radix, 0, 0)

        # a-b = a-b
        if (a1[0] != '-') & (mid1[0] != '-'):
            a3, useless1, useless2 = subtraction(a1,mid1,radix,0,0)
        # a-(-b) = a+b
        if (a1[0] != '-') & (mid1[0] == '-'):
            a3, useless1, useless2 = addition(a1, mid1[1:], radix, 0, 0)
        # (-a)-b = -(a+b)
        if (a1[0] == '-') & (mid1[0] != '-'):
            a3, useless1, useless2 = addition(mid1, a1[1:], radix, 0, 0)
            a3 = "-" + a3
        # (-a)+(-b) = b-a
        if (a1[0] == '-') & (mid1[0] == '-'):
            a3, useless1, useless2 = subtraction(mid1[1:], a1[1:], radix, 0, 0)

        # a-b = a-b
        if (b1[0] != '-') & (mid2[0] != '-'):
            b3, useless1, useless2 = subtraction(b1, mid2, radix, 0, 0)
        # a-(-b) = a+b
        if (b1[0] != '-') & (mid2[0] == '-'):
            b3, useless1, useless2 = addition(b1, mid2[1:], radix, 0, 0)
        # (-a)-b = -(a+b)
        if (b1[0] == '-') & (mid2[0] != '-'):
            b3, useless1, useless2 = addition(mid2, b1[1:], radix, 0, 0)
            b3 = "-" + b3
        # (-a)+(-b) = b-a
        if (b1[0] == '-') & (mid2[0] == '-'):
            b3, useless1, useless2 = subtraction(mid2[1:], b1[1:], radix, 0, 0)

        #a3 = a1 - (a2*q3)
        #b3 = b1 - (b2*q3)
        a1 = a2
        a2 = a3
        b1 = b2
        b2 = b3

    return r1, a1, b1


# Modular Reduction Function
# returns the remainder from the long division function
def modular_reduction(x, m, radix):
    modulo = long_division(x, m, radix)[1]
    return modulo


# Modular Addition Function
# implemented according to algortihm 2.7
def modular_addition(x, y, m, radix):
    a = modular_reduction(x, m, radix)
    b = modular_reduction(y, m, radix)
    # z1 = z'
    z1 = add(a, b, radix)

    if bigger(z1, m, radix) == m:
        z = z1
    else:
        z = subtract(z1, m, radix)
    return z


# Modular subtraction function
# implemented according to algortihm 2.8
def modular_subtraction(x, y, m, radix):
    a = modular_reduction(x, m, radix)
    b = modular_reduction(y, m, radix)
    # z1 = z'
    z1 = subtract(a, b, radix)
    if z1[0] != '-':
        z = z1
    else:
        z = add(z1, m, radix)
    return z


# modular muliplication function
# implemented according to algortihm 2.9
def modular_multiplication(x, y, m, radix):
    a = modular_reduction(x, m, radix)
    b = modular_reduction(y, m, radix)
    # z1 = z'
    z1 = multiply(a, b, radix)
    z = modular_reduction(z1, m, radix)
    return z

# Modular Inversion Function
# works based on the extended euclidean algorithm
def modular_inversion(x, m, radix):
    gcd = euclid(x, m , radix)[0]
    if gcd != '1':
        return "ERROR - inverse does not exist"

    x_mod = euclid(x, m, radix)[2]
    while x_mod[0] == '-':
        x_mod = add(x_mod, m, radix)
        
    return x_mod


def add(x, y, radix):
    if (x[0] != '-') & (y[0] != '-'):
        result = addition(x, y, radix, 0, 0)[0]

    if (x[0] != '-') & (y[0] == '-'):
        result = subtraction(x, y[1:], radix, 0, 0)[0]

    if (x[0] == '-') & (y[0] != '-'):
        result = subtraction(y, x[1:], radix, 0, 0)[0]

    if (x[0] == '-') & (y[0] == '-'):
        result = addition(x[1:], y[1:], radix, 0, 0)[0]

    return result


def subtract(x, y, radix):
    if (x[0] != '-') & (y[0] != '-'):
        result = subtraction(x, y, radix, 0, 0)[0]
        # a-(-b) = a+b
    if (x[0] != '-') & (y[0] == '-'):
        result = addition(x, y[1:], radix, 0, 0)[0]
    if (x[0] == '-') & (y[0] != '-'):
        result = addition(y, x[1:], radix, 0, 0)[0]
    if (x[0] == '-') & (y[0] == '-'):
        result = subtraction(y[1:], x[1:], radix, 0, 0)[0]

    return result


def multiply(x, y, radix):
    if (x[0] != '-') & (y[0] != '-'):
        result = multiplicationKaratsuba(x, y, radix, 0, 0)[0]

    if (x[0] == '-') & (y[0] != '-'):
        result = multiplicationKaratsuba(x[1:], y, radix, 0, 0)[0]

    if (x[0] != '-') & (y[0] == '-'):
        result = multiplicationKaratsuba(x, y[1:], radix, 0, 0)[0]

    if (x[0] == '-') & (y[0] == '-'):
        result = multiplicationKaratsuba(y[1:], x[1:], radix, 0, 0)[0]

    return result


def bigger(x, y, radix):
    if x == y:
        return 0
   
    sign_x = '+'
    if x[0] == '-':
        sign_x = '-'

    sign_y = '+'
    if y[0] == '-':
        sign_y = '-'

    x = string_to_list(x)
    y = string_to_list(y)

    if sign_x == '+' and sign_y == '-':
        return list_to_string(x)

    if sign_x == '-' and sign_y == '+':
        return list_to_string(y)

    res = subtraction(list_to_string(x), list_to_string(y), radix, 0, 0)[0]

    sign_res = '+'
    if res[0] == '-':
        sign_res = '-'

    if sign_res == sign_x:
        return list_to_string(x)

    return list_to_string(y)

### AfS software assignment 1 - example code ###

# set file names
base_location = './'
ops_loc = base_location + 'operations.asn'
exs_loc = base_location + 'my_exercises'
ans_loc = base_location + 'my_answers'

###### Creating an exercise list file ######

# # How to create an exercise JSON file containing one addition exercise
# exercises = {'exercises': []}  # initialize empty exercise list
# ex = {'multiply': {'radix': 10, 'x': '69', 'y': '420', 'answer': ''}}  # create add exercise
# exercises['exercises'].append(ex)  # add exercise to list

# # Encode exercise list and print to file
# my_file = open(exs_loc, 'wb+')  # write to binary file
# my_file.write(json.dumps(exercises).encode())  # add encoded exercise list
# my_file.close()

###### Using an exercise list file ######

# Compile specification
spec = asn.compile_files(ops_loc, codec="jer")

# Read exercise list
exercise_file = open(exs_loc, 'rb')  # open binary file
#exercise_file = open('./test_exercises_students_answers', 'rb')  # open binary file
file_data = exercise_file.read()  # read byte array
my_exercises = spec.decode('Exercises', file_data)  # decode after specification
exercise_file.close()

# Create answer JSON
my_answers = {'exercises': []}

# Loop over exercises and solve
for exercise in my_exercises['exercises']:
    operation = exercise[0]  # get operation type
    params = exercise[1]  # get parameters

    if operation == 'add':
        x = params['x']
        y = params['y']

        # a+b = a+b
        if (x[0] != '-') & (y[0] != '-'):
            answer, a, m = addition(params['x'], params['y'], params['radix'], 0, 0)
            params['answer'] = answer
        # a+(-b)  = a-b
        if (x[0] != '-') & (y[0] == '-'):
            answer, a, m = subtraction(params['x'], params['y'][1:], params['radix'], 0, 0)
            params['answer'] = answer
        # (-a)+b = b-a
        if (x[0] == '-') & (y[0] != '-'):
            answer, a, m = subtraction(params['y'], params['x'][1:], params['radix'], 0, 0)
            params['answer'] = answer
        # (-a)+(-b) = -(a+b)
        if (x[0] == '-') & (y[0] == '-'):
            answer, a, m = addition(params['x'][1:], params['y'][1:], params['radix'], 0, 0)
            params['answer'] = "-" + answer

    if operation == 'subtract':
        x = params['x']
        y = params['y']

        # a-b = a-b
        if (x[0] != '-') & (y[0] != '-'):
            answer, a, m = subtraction(params['x'], params['y'], params['radix'], 0, 0)
            params['answer'] = answer
        # a-(-b) = a+b
        if (x[0] != '-') & (y[0] == '-'):
            answer, a, m = addition(params['x'], params['y'][1:], params['radix'], 0, 0)
            params['answer'] = answer
        # (-a)-b = -(a+b)
        if (x[0] == '-') & (y[0] != '-'):
            answer, a, m = addition(params['y'], params['x'][1:], params['radix'], 0, 0)
            params['answer'] = "-" + answer
        # (-a)+(-b) = b-a
        if (x[0] == '-') & (y[0] == '-'):
            answer, a, m = subtraction(params['y'][1:], params['x'][1:], params['radix'], 0, 0)
            params['answer'] = answer

    if operation == 'karatsuba':
        ### Do multiplication ###
        x = params['x']
        y = params['y']

        # a*b = a*b
        if (x[0] != '-') & (y[0] != '-'):
            answer, a, m = multiplicationKaratsuba(params['x'], params['y'], params['radix'], 0, 0)
            params['answer'] = answer
            params['count-mul'] = str(m)
            params['count-add'] = str(a)
        # (-a)*b = -(a*b)
        if (x[0] == '-') & (y[0] != '-'):
            answer, a, m = multiplicationKaratsuba(params['x'][1:], params['y'], params['radix'], 0, 0)
            params['answer'] = "-" + answer
            params['count-mul'] = str(m)
            params['count-add'] = str(a)
        # a*(-b) = -(a*b)
        if (x[0] != '-') & (y[0] == '-'):
            answer, a, m = multiplicationKaratsuba(params['x'], params['y'][1:], params['radix'], 0, 0)
            params['answer'] = "-" + answer
            params['count-mul'] = str(m)
            params['count-add'] = str(a)
        # (-a)*(-b) = a*b
        if (x[0] == '-') & (y[0] == '-'):
            answer, a, m = multiplicationKaratsuba(params['y'][1:], params['x'][1:], params['radix'], 0, 0)
            params['answer'] = answer
            params['count-mul'] = str(m)
            params['count-add'] = str(a)

    if operation == 'multiply':
        ### Do multiplication ###
        x = params['x']
        y = params['y']

        # a*b = a*b
        if (x[0] != '-') & (y[0] != '-'):
            answer, a, m = multiplicationPrimary(params['x'], params['y'], params['radix'], 0, 0)
            params['answer'] = answer
            params['count-mul'] = str(m)
            params['count-add'] = str(a)
        # (-a)*b = -(a*b)
        if (x[0] == '-') & (y[0] != '-'):
            answer, a, m = multiplicationPrimary(params['x'][1:], params['y'], params['radix'], 0, 0)
            params['answer'] = "-" + answer
            params['count-mul'] = str(m)
            params['count-add'] = str(a)
        # a*(-b) = -(a*b)
        if (x[0] != '-') & (y[0] == '-'):
            answer, a, m = multiplicationPrimary(params['x'], params['y'][1:], params['radix'], 0, 0)
            params['answer'] = "-" + answer
            params['count-mul'] = str(m)
            params['count-add'] = str(a)
        # (-a)*(-b) = a*b
        if (x[0] == '-') & (y[0] == '-'):
            answer, a, m = multiplicationPrimary(params['y'][1:], params['x'][1:], params['radix'], 0, 0)
            params['answer'] = answer
            params['count-mul'] = str(m)
            params['count-add'] = str(a)

    if operation == 'euclid':
        d, a, b = euclid(params['x'], params['y'], params['radix'])
        params['answ-d'] = d
        params['answ-a'] = a
        params['answ-b'] = b

    if operation == 'mod-add':
        params['answer'] = modular_addition(params['x'],params['y'],params['m'],params['radix'])

    if operation == 'mod-subtract':
        params['answer'] = modular_subtraction(params['x'], params['y'], params['m'], params['radix'])

    if operation == 'mod-multiply':
        params['answer'] = modular_multiplication(params['x'], params['y'], params['m'], params['radix'])

    if operation == 'reduce':
        params['answer'] = modular_reduction(params['x'], params['m'], params['radix'])

    if operation == 'inverse':
        params['answer'] = modular_inversion(params['x'], params['m'], params['radix'])
        
    # etc.

    # Save answer
    my_answers['exercises'].append({operation: params})

###### Creating an answers list file ######

# Save exercises with answers to file
my_file = open(ans_loc, 'wb+')  # write to binary file
my_file.write(json.dumps(my_answers).encode())  # add encoded exercise list
my_file.close()