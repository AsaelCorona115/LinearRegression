import random

# The following is a program to calculate the Gradient descent using given values
# of x, y, w0, w1 and alpha.
# The given values are the following

xValues = [2, 4, 6, 7, 8, 10]
yValues = [5, 7, 14, 14, 17, 19]
w0 = 0.25  # Initial w0
w1 = 0.25  # Initial w1
alpha = 0.001  # Learning Rate

# This function can be used to calculate the Hw value for w0. It is called under
# the calculateNewValue function


def valuesForw0w1(w0, w1, x, y):

    hwValuew0 = y - (w0 + (w1 * x))
    hwValuew1 = (y - (w0 + (w1 * x))) * x
    return hwValuew0, hwValuew1


def SGDWOW1(w0, w1, x, y, learningRate):
    neww0 = (learningRate * (y - ((w1 * x) + w0))) + w0
    neww1 = ((learningRate * (y - ((w1 * x) + w0))) * x) + w1
    return neww0, neww1

# The following function performs the BatchGradientDescent Calculation


def BatchGradientDescent(originalW0, originalW1, valuesOfX, valuesOfY):
    # Initialization of original arrays
    w0sum = 0
    w1sum = 0

    # The following section calculates the sum of (Yj - (Hw(Xj))) and
    # (Yj - (Hw(Xj))) * Xj for all x and y values given
    for i in range(len(valuesOfX)):
        w0Value, w1Value = valuesForw0w1(
            originalW0, originalW1, valuesOfX[i], valuesOfY[i])
        w0sum += w0Value
        w1sum += w1Value

    # This code below finished the w0 and w1 formula by multiplying the sum times
    # the learning rate (alpha) and adding the original value of w
    BGDW0 = (w0sum * alpha) + originalW0
    BGDW1 = (w1sum * alpha) + originalW1

    return(BGDW0, BGDW1)


# Initial Call. BGD stands for Batch Gradient Descent. SGD stands for Stochastic
# Gradient descent.
BGDW0, BGDW1 = BatchGradientDescent(w0, w1, xValues, yValues)

# Using the following counter and loop we are able to run the BatchGradientDescent
# Algorithm until convergence or until the iteration has happened 100000 times
counter = 0
while((abs(BGDW0 - w0) > 10 ** -10) and counter < 100000):
    counter += 1
    w0 = BGDW0
    w1 = BGDW1
    BGDW0, BGDW1 = BatchGradientDescent(BGDW0, BGDW1, xValues, yValues)


# Output. Added some text for easier understanding
print('Using Batch Gradient Descent: ')
print('W0 = ' + str(BGDW0))
print('W1 = ' + str(BGDW1))
print('It took ' + str(counter) + ' iterations to reach convergence')
print('Equation of the line: y = ' +
      str(round(BGDW1, 2)) + 'x + ' + str(round(BGDW0, 2)))


print('')
print('')

# SGD Initial Call
# Reinitializing Values
w0 = 0.25  # Initial w0
w1 = 0.25  # Initial w1
alpha = 0.001  # Learning Rate
SGDW0, SGDW1 = SGDWOW1(w0, w1, xValues[0], yValues[0], alpha)


counter = 0
while((abs(SGDW0 - w0) > 10 ** -10) and counter < 100000):
    counter += 1
    w0 = SGDW0
    w1 = SGDW1
    random_index = random.randint(0, len(xValues) - 1)
    SGDW0, SGDW1 = SGDWOW1(
        w0, w1, xValues[random_index], yValues[random_index], alpha)

print('Using Stochastic Gradient Descent: ')
print('w0 = ' + str(SGDW0))
print('w1 = ' + str(SGDW1))
print('It took ' + str(counter) + ' iterations to reach convergence')
print('Equation of the line: y = ' +
      str(round(SGDW1, 2)) + 'x + ' + str(round(SGDW0, 2)))
