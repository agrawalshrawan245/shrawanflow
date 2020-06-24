import numpy as np
import math
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
        if _default_graph.debug_val != 0: print(self, self.output, sep = '\n')

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
        if _default_graph.debug_val != 0: print(self, self.output, sep = '\n')
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
        if _default_graph.debug_val != 0: print(self, self.output, sep = '\n')
        return self.output

    def fit(self, delta_val):
        for input_node in self.input_nodes:
            if _default_graph.debug_val != 0: print(self, delta_val * self.output / input_node.output, sep = '\n')
        for input_node in self.input_nodes:
            input_node.fit(delta_val * self.output / input_node.output)

class sigmoid(Operation):

    def __init__(self, input_nodes= []):
        super().__init__(input_nodes = input_nodes)

    def sig(self, x):
        return (1 / (1 + math.exp(-x)))

    def sig_diff(self, x):
        return self.sig(x) - self.sig(x) * self.sig(x)
    
    def predict(self, feed_dict):
        for input_node in self.input_nodes:
            input_node.predict(feed_dict)
        inputs = [input_node.output for input_node in self.input_nodes]

        sig_v = np.vectorize(self.sig)
        self.output = sig_v(*inputs)
        if _default_graph.debug_val != 0: print(self, self.output, sep = '\n')
        return self.output

    def fit(self, delta_val):
        sig_diff_v = np.vectorize(self.sig_diff)
        temp = np.multiply(sig_diff_v(self.output), delta_val)
        if _default_graph.debug_val != 0: print(self, temp, sep = '\n')
        self.input_nodes[0].fit(temp)

class inorout(Operation):
    
    def __init__(self, input_nodes= []):
        super().__init__(input_nodes = input_nodes)

    def iomath(self,x):
        if x < 0: return 0.0
        else: return 1.0
        
    def predict(self, feed_dict):
        for input_node in self.input_nodes:
            input_node.predict(feed_dict)
        inputs = [input_node.output for input_node in self.input_nodes]

        iomath_v = np.vectorize(self.iomath)
        self.output = iomath_v(*inputs)
        if _default_graph.debug_val != 0: print(self, self.output, sep = '\n')
        return self.output

    def fit(self, delta_val):
        iomath_v = np.vectorize(self.iomath)
        temp = np.multiply(iomath_v(self.output), delta_val)
        if _default_graph.debug_val != 0: print(self, temp, sep = '\n')
        self.input_nodes[0].fit(temp)

class matmul(Operation):

    def __init__(self, input_nodes= []):
        super().__init__(input_nodes = input_nodes)
  
    def predict(self, feed_dict):
        for input_node in self.input_nodes:
            input_node.predict(feed_dict)
        inputs = [input_node.output for input_node in self.input_nodes]

        self.output = np.dot(*inputs)
        if _default_graph.debug_val != 0: print(self, self.output, sep = '\n')
        return self.output

    def fit(self, delta_val):
        if _default_graph.debug_val != 0: print(self, np.dot(np.transpose(self.input_nodes[0].output), delta_val), sep = '\n')
        if _default_graph.debug_val != 0: print(self, np.dot(delta_val, np.transpose(self.input_nodes[1].output)), sep = '\n')
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
        if _default_graph.debug_val != 0: print(self, self.output, sep = '\n')
        return self.output

    def fit(self, delta_val):
        for input_node in self.input_nodes:
            if _default_graph.debug_val != 0: print(self, delta_val, sep = '\n')
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
        if _default_graph.debug_val != 0: print(self, self.output, sep = '\n')
        return self.output

    def fit(self, delta_val):
        if _default_graph.debug_val != 0: print(self, delta_val, sep = '\n')
        if _default_graph.debug_val != 0: print(self, -delta_val, sep = '\n')
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
        if _default_graph.debug_val != 0: print(self, self.output, sep = '\n')
        return self.output

    def fit(self, delta_val):
        for input_node in self.input_nodes:
            if _default_graph.debug_val != 0: print(self, 2*delta_val*input_node.output, sep = '\n')
        for input_node in self.input_nodes:
            input_node.fit(2*delta_val*input_node.output)

class Session():

    def predict(self, node, feed_dict = {}):
        return node.predict(feed_dict)

    def fit(self, node, l_rate, batch = 1, feed_dict = {}):
        _default_graph.iterator = 1 + _default_graph.iterator

        if _default_graph.debug_val != 0: print("\nForward prop.")
        loss = node.predict(feed_dict)
        print(loss)
        _default_graph.loss.append(loss)

        if _default_graph.debug_val != 0: print("\nBackward prop.")
        node.fit(l_rate)
        if _default_graph.iterator == batch:
            _default_graph.iterator = 0
            for variable in _default_graph.variables:
                variable.deltazero(batch)
        
        if _default_graph.debug_val != 0: print("--------------------------------------------------")
        
        if _default_graph.debug_val != 0: _default_graph.debug_val = _default_graph.debug_val - 1

        return loss

class Graph():

    def __init__(self, debug_val = 0):
        self.iterator = 0
        self.variables = []
        self.placeholders = []
        self.operations = []
        self.loss = []
        self.debug_val  = debug_val

    def set_as_default(self):
        global _default_graph
        _default_graph = self


g = Graph()
g.set_as_default()

a = Variable(np.array([[-20,-30],[-10,-40]]))
b = Variable(np.array([[-10],[-60]]))
x = Placeholder()
y = matmul([a,x])
z = add([y,b])
z_sig = inorout([z])
y_true = Placeholder()
loss = square_error([subtract([y_true, z_sig])])

sess = Session()

A = np.array([[3,4],[2,7]])
B = np.array([[1],[5]])

print("Earlier:     ")
print(a.output)
print(b.output)

def iomath(x):
    if x < 0: return 0.0
    else: return 1.0
sig_v = np.vectorize(iomath)


for i in range(500):
    temp = np.random.rand(2,1)
    temp_ = sig_v(np.add(np.dot(A, temp), B))

    sess.fit(node = loss, l_rate = np.array([0.7]), batch = 2, feed_dict={x:temp, y_true:temp_})


print("Now    :     ")
print(a.output)
print(b.output)


# Foot notes
# is it right to do fit(a*b) or it should be fit(dot(a, b))
# global not working for debug_val
# place names of nodes, also make debugger proper by adding outgoing nodes


