import numpy as np

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
        self.delta_val = 0

    def predict(self, feed_dict):
        pass

    def fit(self, delta_val):
        self.delta_val = self.delta_val + delta_val

    def deltazero(self, epochs = 1):
        self.output = self.output - self.delta_val / epochs
        self.delta_val = 0

class Placeholder():

    def __init__(self):
        self.output_nodes = []

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


class reduced_mean(Operation):

    def __init__(self, input_nodes = []):
        super().__init__(input_nodes = input_nodes)

    def predict(self, feed_dict):
        for input_node in self.input_nodes:
            input_node.predict(feed_dict)
        inputs = [input_node.output for input_node in self.input_nodes]

        self.output = np.square(inputs[0] - inputs[1])
        return self.output

    def fit(self, delta_val):
        a = self.input_nodes[0]
        b = self.input_nodes[1]
        delta = 2 * (a.output - b.output)
        a.fit(delta_val * delta)
        b.fit(-delta_val * delta)


class Session():

    def __init__(self, variable_lis = []):
        self.iterator = 0
        self.variable_lis = variable_lis

    def predict(self, node, feed_dict = {}):
        return node.predict(feed_dict)

    def fit(self, node, l_rate, epochs = 1, feed_dict = {}):
        self.iterator = 1 + self.iterator
        node.predict(feed_dict)
        node.fit(l_rate)
        if self.iterator == epochs:
            self.iterator = 0
            for variable in self.variable_lis:
                variable.deltazero(epochs)



a = Variable(np.array([3]))
b = Variable(np.array([2]))
x = Placeholder()
y = multiply([a,x])
z = add([y,b])
y_true = Placeholder()
loss = reduced_mean([y_true, z])
sess = Session([a,b])


for _ in range(500):
    temp = np.random.rand()
    sess.fit(node = loss, l_rate = np.array([0.4]), epochs = 5, feed_dict={x:[temp], y_true:[2 * temp + 8]})


print("Earlier: [3.] * " + str(x.output) + " + [2.] = " + str(np.add(np.multiply(3, x.output), 2)))
print("Now:     " + str(a.output) + " * " + str(x.output) + " + " + str(b.output) + " = " + str(y.output))



