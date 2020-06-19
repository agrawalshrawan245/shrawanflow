import numpy as np
import pdb

class Operation():

    def __init__(self, input_nodes = []):
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in input_nodes:
            node.output_nodes.append(self)

class Variable():

    def __init__(self, initial_value):
        self.output = initial_value
        self.output_nodes = []

    def fit(self, del_val):
        self.output = self.output - del_val

class Placeholder():

    def __init__(self):
        self.output_nodes = []

    def fit(self, del_val):
        pass

class multiply(Operation):

    def __init__(self, input_nodes= []):
        super().__init__(input_nodes = input_nodes)

    def predict(self, x_var, y_var):
        
        return np.multiply(x_var, y_var)

    def fit(self, del_val):
        for input_node in self.input_nodes:
            input_node.fit(del_val * self.output / input_node.output)

class add(Operation):

    def __init__(self, input_nodes= []):
        super().__init__(input_nodes = input_nodes)

    def predict(self, x, y):
        return np.add(x, y)

    def fit(self, del_val):
        for input_node in self.input_nodes:
            input_node.fit(del_val * self.output / input_node.output)


class reduced_mean(Operation):

    def __init__(self, input_nodes = []):
        super().__init__(input_nodes = input_nodes)

    def predict(self, feed_dict):
        a = input_nodes[0]
        b = input_nodes[1]
        return np.square(a.output - b.output)

    def fit(self, del_val):
        a = input_nodes[0]
        b = input_nodes[1]
        delta = 2 * (a.output - b.output)
        a.fit(del_val * delta)
        b.fit(-del_val * delta)


class Session():

    def __init__(self):
        self.iterator = 0
        self.variable_lis = []

    def predict(self, node, feed_dict = {}):

        if type(node) == Placeholder:
            node.output = feed_dict[node]

        elif type(node) == Variable:
            pass
        else:
            for input_node in node.input_nodes:
                self.predict(input_node, feed_dict)
            
            inputs = [input_node.output for input_node in node.input_nodes]
            node.output = node.predict(*inputs)


    def fit(self, node, l_rate, feed_dict = {}):
        self.predict(node, feed_dict)
        node.fit(l_rate)



A = Variable(np.array([5]))
b = Variable(np.array([2]))
x = Placeholder()
y = multiply([A,x])
z = add([y,b])
y_true = Placeholder()
loss = reduced_mean([y_true, z])

sess = Session()
for _ in range(30):
    temp = np.random.rand()
    sess.fit(node = z, l_rate = np.array([1]), feed_dict={x:[2*temp], y_true:[temp + 3]})

print(A.output)
print(b.output)


# A = Variable(np.array([6]))
# x = Placeholder()
# y = multiply([A,x])

# predict(y, feed_dict = {x:np.array([3])})

# fit(y, np.array([1]), feed_dict = {x:np.array([3])})

# print(A.output)




























