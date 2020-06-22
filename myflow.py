import numpy as np
import pdb

class Operation():

    def __init__(self, input_nodes = []):
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in input_nodes:
            node.output_nodes.append(self)

        _default_graph.operations.append(self)

class Variable():

    def __init__(self, initial_value):
        self.output = initial_value
        self.output_nodes = []
        self.delta_val = 0
        _default_graph.variables.append(self)

    def predict(self, feed_dict):
        pass

    def fit(self, delta_val):
        self.delta_val = self.delta_val + delta_val

    def deltazero(self, batch = 1):
        self.output = self.output - self.delta_val / batch
        self.delta_val = 0

class Placeholder():

    def __init__(self):
        self.output_nodes = []
        _default_graph.placeholders.append(self)

    def predict(self, feed_dict):
        self.output = feed_dict[self]
        return self.output

    def fit(self, delta_val):
        pass

class multiply(Operation):

    def __init__(self, input_nodes= []):
        super().__init__(input_nodes = input_nodes)

    def predict(self, feed_dict):
        for input_node in self.input_nodes:
            input_node.predict(feed_dict)
        inputs = [input_node.output for input_node in self.input_nodes]

        self.output = np.multiply(*inputs)
        return self.output

    def fit(self, delta_val):
        for input_node in self.input_nodes:
            input_node.fit(delta_val * self.output / input_node.output)

class matmul(Operation):

    def __init__(self, input_nodes= []):
        super().__init__(input_nodes = input_nodes)
  
    def predict(self, feed_dict):
        for input_node in self.input_nodes:
            input_node.predict(feed_dict)
        inputs = [input_node.output for input_node in self.input_nodes]

        self.output = np.dot(*inputs)
        return self.output

    def fit(self, delta_val):
        self.input_nodes[1].fit(np.dot(np.transpose(self.input_nodes[0].output), delta_val))
        self.input_nodes[0].fit(np.dot(delta_val, np.transpose(self.input_nodes[1].output)))

class add(Operation):

    def __init__(self, input_nodes= []):
        super().__init__(input_nodes = input_nodes)

    def predict(self, feed_dict):
        for input_node in self.input_nodes:
            input_node.predict(feed_dict)
        inputs = [input_node.output for input_node in self.input_nodes]

        self.output = np.add(*inputs)
        return self.output

    def fit(self, delta_val):
        for input_node in self.input_nodes:
            input_node.fit(delta_val)

class subtract(Operation):

    def __init__(self, input_nodes= []):
        super().__init__(input_nodes = input_nodes)

    def predict(self, feed_dict):
        for input_node in self.input_nodes:
            input_node.predict(feed_dict)
        inputs = [input_node.output for input_node in self.input_nodes]

        self.output = np.subtract(*inputs)
        return self.output

    def fit(self, delta_val):
        self.input_nodes[0].fit(delta_val)
        self.input_nodes[1].fit(-delta_val)

class square_error(Operation):

    def __init__(self, input_nodes = []):
        super().__init__(input_nodes = input_nodes)
        
    def predict(self, feed_dict):
        for input_node in self.input_nodes:
            input_node.predict(feed_dict)
        inputs = [input_node.output for input_node in self.input_nodes]

        self.output = np.square(*inputs).sum()
        return self.output

    def fit(self, delta_val):
        for input_node in self.input_nodes:
            input_node.fit(2*delta_val*input_node.output)

class Session():

    def predict(self, node, feed_dict = {}):
        return node.predict(feed_dict)

    def fit(self, node, l_rate, batch = 1, feed_dict = {}):
        _default_graph.iterator = 1 + _default_graph.iterator
        node.predict(feed_dict)
        node.fit(l_rate)
        if _default_graph.iterator == batch:
            _default_graph.iterator = 0
            for variable in _default_graph.variables:
                variable.deltazero(batch)

class Graph():

    def __init__(self):
        self.iterator = 0
        self.variables = []
        self.placeholders = []
        self.operations = []

    def set_as_default(self):
        global _default_graph
        _default_graph = self


g = Graph()
g.set_as_default()

a = Variable(np.array([[2,3],[1,4]]))
b = Variable(np.array([[1],[6]]))
x = Placeholder()
y = matmul([a,x])
z = add([y,b])
y_true = Placeholder()
loss = square_error([subtract([y_true, z])])

sess = Session()

A = np.array([[3,5],[6,9]])
B = np.array([[6],[2]])

print("Earlier:     ")
print(a.output)
print(b.output)

for i in range(5000):
    temp = np.random.rand(2,1)
    temp_ = np.add(np.dot(A, temp), B)
    sess.fit(node = loss, l_rate = np.array([0.04]), batch = 1, feed_dict={x:temp, y_true:temp_})


print("Now    :     ")
print(a.output)
print(b.output)


# Foot notes
# is it right to do fit(a*b) or it should be fit(dot(a, b))


