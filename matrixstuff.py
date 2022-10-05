#!/usr/bin/env python3

class Matrix:

    def __init__(self, matrix):
        self.matrix = matrix
        self.col = len(matrix[0])
        self.row = len(matrix)

    def __add__(self, other):
        assert self.row == other.row and self.col == other.col
        return Matrix([
            [
                self.matrix[r][c] + other.matrix[r][c]
                for c in range(self.col)
            ]
            for r in range(self.row)
        ])

    __radd__ = __add__

    def __str__(self):
        return '\n'.join(['\t'.join([f'{cell:.10f}' for cell in row]) for row in self.matrix]) + '\n'

    def __matmul__(self, other):
        '''
        IN PYTHON MATRIX MULTIPLICATION IS LEFT ASSOCIATIVE
        Here's an example: vector @ transformation1 @ transformation2
        '''
        assert self.row == other.col
        newMat = [[0 for c in range(self.col)] for r in range(other.row)]
        for r in range(other.row):
            for c in range(self.col):
                sum = 0
                for i in range(self.row):
                    sum += other.matrix[r][i] * self.matrix[i][c]
                newMat[r][c] = sum

        return Matrix(newMat)

    def __imatmul__(self, other):
        '''inplace matrix multiplication'''
        self = self.__matmul__(other)
        return self

    def __mul__(self, other):
        '''matrix multiplied by constant'''
        if isinstance(other, float) or isinstance(other, int):
            return Matrix([[self.matrix[row][col]*other for col in range(len(self.matrix[0]))] for row in range(len(self.matrix))])
        else:
            return None

    __rmul__ = __mul__

    def __imul__(self, other):
        '''inplace constant multiplication'''
        self = self.__mul__(other)
        return self

    def __truediv__(self, other):
        '''matrix divided by constant'''
        if isinstance(other, float) or isinstance(other, int):
            return Matrix([[self.matrix[row][col]/other for col in range(len(self.matrix[0]))] for row in range(len(self.matrix))])
        else:
            return None

    def __itruediv__(self, other):
        '''inplace constant division'''
        self = self.__truediv__(other)
        return self

    @staticmethod
    def det(matrix):
        '''finds determinant of 2d list'''
        sum = 0
        if len(matrix[0]) == len(matrix) == 1: return matrix[0][0]

        for i in range(len(matrix[0])):
            sum += (-1)**(i) * matrix[0][i] * Matrix.det([[matrix[row][col] for col in range(len(matrix[0])) if col!=i] for row in range(len(matrix)) if row != 0])
        return sum

    def T(self):
        '''returns new Matrix that is transposed version of self'''
        return Matrix([[self.matrix[row][col] for row in range(len(self.matrix))] for col in range(len(self.matrix[0]))])

    def transpose(self):
        '''transposes self inplace'''
        self = self.T()
        return self

    def inverse(self):
        deter = Matrix.det(self.matrix)
        if deter == 0: return None
        cofactors = Matrix([
            [
                Matrix.det(
                        [
                            [
                                self.matrix[row][col]
                                for col in range(len(self.matrix[0])) if col != c
                            ]
                            for row in range(len(self.matrix)) if row != r
                        ]
                    ) * (-1)**(r+c)
                for c in range(len(self.matrix[0]))
            ]
            for r in range(len(self.matrix))
        ])

        adjugate = cofactors.T()

        return adjugate / deter
    
    def invert(self):
        '''inverts self inplace'''
        self = self.inverse()
        return self

#Example
m1 = Matrix([
    [2,3],
    [1,0]
])

print(m1)
print(m1.inverse())
print(m1.T())
m2 = (m1 @ m1.inverse() @ m1 @ m1 @ m1.inverse()).T() # m2 is just m1 transposed. again, remember to read left to right, not right to left.
print(m1 + m2)
print((m1+m2)*5/4*3/15)
m2.invert()
print(m2)

