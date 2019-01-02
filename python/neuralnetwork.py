import random
import math
from numbers import Number

class NeuralNetwork:
    def __init__(self, input_nodes, layers, learning_rate = 0.01):
        self.input_nodes = input_nodes
        self.layers = layers
        self.learning_rate = learning_rate

        for x in range(len(self.layers)):
            if x == 0:
                nodes = self.input_nodes
            else:
                nodes = self.layers[x - 1].nodes

            self.layers[x].weights = Matrix(self.layers[x].nodes, nodes)
            self.layers[x].bias = Matrix(self.layers[x].nodes, 1)

            self.layers[x].weights.random(0, 1)
            self.layers[x].bias.random(0, 1)

    def predict(self, input_arr):
        inputs = Matrix.from_arr(input_arr)

        prev_layer_values = inputs

        for layer in self.layers:
            layer.layer_values = Matrix._multiply(layer.weights, prev_layer_values)
            layer.layer_values.add(layer.bias)
            layer.layer_values.map(layer.activation_function.function)

            prev_layer_values = layer.layer_values

        outputs = prev_layer_values
        
        return Matrix.to_arr(outputs)

    def train(self, input_arr, target_arr):
        inputs = Matrix.from_arr(input_arr)

        prev_layer_values = inputs

        for layer in self.layers:
            layer.layer_values = Matrix._multiply(layer.weights, prev_layer_values)
            layer.layer_values.add(layer.bias)
            layer.layer_values.map(layer.activation_function.function)

            prev_layer_values = layer.layer_values

        outputs = prev_layer_values
        targets = Matrix.from_arr(target_arr)

        prev_error = None

        for x in range(len(self.layers)):
            index = len(self.layers) - 1 - x

            layer = self.layers[index]

            if prev_error == None:
                error = Matrix._subtract(targets, outputs)
            else:
                next_weights = self.layers[index + 1].weights
                next_weights_transposed = Matrix.transpose(next_weights)

                error = Matrix._multiply(next_weights_transposed, prev_error)
        
            gradient = Matrix._map(layer.layer_values, layer.activation_function.dfunction)
            gradient.multiply(error)
            gradient.multiply(self.learning_rate)

            prev_values = self.layers[index - 1].layer_values
            if index == 0:
                prev_values = inputs
            
            prev_values_transposed = Matrix.transpose(prev_values)
            layer_weights_deltas = Matrix._multiply(gradient, prev_values_transposed)

            layer.weights.add(layer_weights_deltas)
            layer.bias.add(gradient)

            prev_error = error

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate


class Layer:
    def __init__(self, nodes, activation_function):
        self.nodes = nodes
        self.weights = None
        self.bias = None
        self.layer_values = None

        if activation_function == 'sigmoid':
            self.activation_function = ActivationFunction(sigmoid, dsigmoid)
        else:
            self.activation_function = ActivationFunction(sigmoid, dsigmoid)

class ActivationFunction:
    def __init__(self, function, dfunction):
        self.function = function
        self.dfunction = dfunction
        
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

        self.matrix = []

        for x in range(self.rows):
            self.matrix.append([])
            for y in range(self.cols):
                self.matrix[x].append(0)
    
    def random(self, _min, _max):
        for x in range(self.rows):
            for y in range(self.cols):
                self.matrix[x][y] = random.random() * (_max - _min) + _min
    
    def map(self, f):
        for x in range(self.rows):
            for y in range(self.cols):
                value = self.matrix[x][y]
                self.matrix[x][y] = f(value)

    def add(self, n):
        if isinstance(n, Matrix):
            if self.rows != n.rows or self.cols != n.cols:
                raise Exception("Given matrices are not the same size")

            for x in range(self.rows):
                for y in range(self.cols):
                    self.matrix[x][y] = self.matrix[x][y] + n.matrix[x][y]
        
        else:
            for x in range(self.rows):
                for y in range(self.cols):
                    self.matrix[x][y] = self.matrix[x][y] + n

    def subtract(self, n):
        if isinstance(n, Matrix):
            if self.rows != n.rows or self.cols != n.cols:
                raise Exception("Given matrices are not the same size")

            for x in range(self.rows):
                for y in range(self.cols):
                    self.matrix[x][y] = self.matrix[x][y] - n.matrix[x][y]
        
        else:
            for x in range(self.rows):
                for y in range(self.cols):
                    self.matrix[x][y] = self.matrix[x][y] - n
    
    def multiply(self, n):
        if isinstance(n, Matrix):
            if self.rows != n.rows or self.cols != n.cols:
                raise Exception("Given matrices are not the same size")

            for x in range(self.rows):
                for y in range(self.cols):
                    self.matrix[x][y] = self.matrix[x][y] * n.matrix[x][y]
        
        else:
            for x in range(self.rows):
                for y in range(self.cols):
                    self.matrix[x][y] = self.matrix[x][y] * n

    @staticmethod
    def transpose(m):
        result = Matrix(m.cols, m.rows)

        for x in range(result.rows):
            for y in range(result.cols):
                result.matrix[x][y] = m.matrix[y][x]

        return result

    @staticmethod
    def _add(m, n):
        result = Matrix(m.rows, m.cols)

        if isinstance(n, Matrix):
            if m.rows != n.rows or m.cols != n.cols:
                raise Exception("Given matrices are not the same size")

            for x in range(result.rows):
                for y in range(result.cols):
                    result.matrix[x][y] = m.matrix[x][y] + n.matrix[x][y]

        else:
            for x in range(result.rows):
                for y in range(result.cols):
                    result.matrix[x][y] = m.matrix[x][y] + n

        return result

    @staticmethod
    def _subtract(m, n):
        result = Matrix(m.rows, m.cols)

        if isinstance(n, Matrix):
            if m.rows != n.rows or m.cols != n.cols:
                raise Exception("Given matrices are not the same size")

            for x in range(result.rows):
                for y in range(result.cols):
                    result.matrix[x][y] = m.matrix[x][y] - n.matrix[x][y]

        else:
            for x in range(result.rows):
                for y in range(result.cols):
                    result.matrix[x][y] = m.matrix[x][y] - n

        return result

    @staticmethod
    def _multiply(a, b):
        if isinstance(a, Number) and isinstance(b, Number):
            return a * b
        
        if isinstance(a, Matrix) and isinstance(b, Matrix):
            if a.cols == b.rows:
                result = Matrix(a.rows, b.cols)

                for x in range(result.rows):
                    for y in range(result.cols):
                        _sum = 0
                        for k in range(a.cols):
                            _sum += a.matrix[x][k] * b.matrix[k][y]

                        result.matrix[x][y] = _sum

                return result
            
            elif a.rows == b.rows and a.cols == b.cols:
                result = Matrix(a.rows, a.cols)

                for x in range(a.rows):
                    for y in range(a.cols):
                        result.matrix[x][y] = a.matrix[x][y] * b.matrix[x][y]

                return result
            
            else:
                raise Exception("Given matrices cannot be multiplied")
        
        else:
            m = a
            n = b

            if isinstance(a, Number):
                m = b
                n = a
            
            result = Matrix(m.rows, m.cols)

            for x in range(m.rows):
                for y in range(m.cols):
                    result.matrix[x][y] = m.matrix[x][y] * n

            return result

    @staticmethod
    def _map(m, f):
        result = Matrix(m.rows, m.cols)

        for x in range(m.rows):
            for y in range(m.cols):
                value = m.matrix[x][y]
                result.matrix[x][y] = f(value)

        return result

    @staticmethod
    def from_arr(arr, cols = 1):
        rows = math.ceil(len(arr) / cols)

        result = Matrix(rows, cols)

        for x in range(rows):
            for y in range(cols):
                index = x * cols + y
                if not isinstance(arr[index], Number):
                    result.matrix[x][y] = 0
                else:
                    result.matrix[x][y] = arr[index]

        return result

    @staticmethod
    def to_arr(m):
        arr = []

        for x in range(m.rows):
            for y in range(m.cols):
                arr.append(m.matrix[x][y])

        return arr