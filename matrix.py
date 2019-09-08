import random

class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix
        self.height = len(self.matrix)
        self.width = len(self.matrix[0])

    def __repr__(self):
        s = ''
        for i in range(self.height):
            for j in range(self.width):
                s += str(self.matrix[i][j]) + ' '
            s += '\n'

        return s

    def getRank(self):
        return (self.height, self.width)

    def __add__(self, other):
        if self.getRank() != other.getRank():
            return Exception("Matrices are of different rank")
        
        newMatrix = [[0 for x in range(self.width)] for y in range(self.height)]
        for i in range(self.height):
            for j in range(self.width):
                newMatrix[i][j] = self.matrix[i][j] + other.matrix[i][j]
        return Matrix(newMatrix)

    def __sub__(self, other):
        if self.getRank() != other.getRank():
            return Exception("Matrices are of different rank")

        newMatrix = [[0 for x in range(self.width)] for y in range(self.height)]
        for i in range(self.height):
            for j in range(self.width):
                newMatrix[i][j] = self.matrix[i][j] - other.matrix[i][j]
        return Matrix(newMatrix)

    def __mul__(self, other):
        if self.width != other.height:
            Exception("Can't multiply!")
        
        newMatrix = [[0 for x in range(self.width)] for y in range(other.height)]
        for i in range(self.height):
            for j in range(other.width):
                for k in range(other.height):
                    newMatrix[i][j] += self.matrix[i][k] * other.matrix[k][j]

        return Matrix(newMatrix)

    def transpose(self): 
        newMatrix = [[0 for x in range(self.height)] for y in range(self.width)]

        for i in range(self.width): 
            for j in range(self.height): 
                newMatrix[i][j] = self.matrix[j][i] 

        return Matrix(newMatrix)

m1 = Matrix([[random.randint(0, 10) for x in range(3)] for y in range(3)])
m2 = Matrix([[random.randint(0, 10) for x in range(3)] for y in range(3)])
print(m1)
print(m2)
print(m1.transpose())