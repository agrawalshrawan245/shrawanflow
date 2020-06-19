# import pdb

# def transform(x, y):
#     y = y**2
#     x *= 2
#     z = x + y
#     return z

# x = 50
# y = 60
# z = 5
# n = 1000

# pdb.set_trace()
# transform(5, 10)
# print("z = " + str(z))


import numpy as np

class Operation():

    def __init__(self, input_nodes = []):
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in input_nodes:
            node.output_nodes.append(self)

    def compute(self):

        pass

class add(Operation):

    def __init__(self, input_nodes=[]):
        super().__init__(input_nodes=input_nodes)
    
    def compute(self, x_var, y_var):

        return np.add(x_var, y_var)

    def backprop(self, del_val):
        
        for input_node in self.input_nodes:
            input_node.backprop(del_val)


# class matmul(Operation):

#     def __init__(self, input_nodes=[]):
#         super().__init__(input_nodes=input_nodes)

#     def compute(self, x_var, y_var):

#         return np.matmul(x_var, y_var)

#     def backprop(self, del_val):
        
#         for input_node in self.input_nodes:
#             input_node.backprop(del_val * self.output / input_node.output)

class multiply(Operation):

    def __init__(self, input_nodes=[]):
        super().__init__(input_nodes=input_nodes)

    def compute(self, x_var, y_var):

        return np.multiply(x_var, y_var)

    def backprop(self, del_val):
        
        for input_node in self.input_nodes:
            input_node.backprop(del_val * self.output / input_node.output)

class reduced_mean(Operation):
    def __init__(self, input_nodes_y, input_nodes_l):
        self.y_true = input_nodes_y
        self.logits = input_nodes_l
        super().__init__(input_nodes=[*input_nodes_y, *input_nodes_l])

    def compute(self, y_val, l_val):
        output = np.square(np.subtract(y_val, l_val))
        return output

    def backprop(self, del_val):
        diff = np.subtract(self.y_true, self.l_true)
        for i, _ in enumerate(diff):
            self.y_true[i].backprop(del_val * diff[i] * 2)
            self.l_true[i].backprop(del_val * diff[i] * (-2))
        

class Placeholder():

    def __init__(self):

        self.output_nodes = []

    def backprop(self, del_val):
        
        pass

class Variable():

    def __init__(self, value):

        self.del_val = 0
        self.value = value
        self.output_nodes = []
    
    def backprop(self, del_val):
        self.del_val += del_val

class Session():
    
    def __init__(self):
        self.iterator = 0
        self.variable_lis = []

    def predict(self, node, feed_dict = {}):

        if type(node) == Placeholder:

            node.output = feed_dict[node]

        elif type(node) == Variable:

            node.output = node.value
            self.variable_lis.append(node)

        else:

            for input_node in node.input_nodes:
                self.predict(input_node, feed_dict)
            
            inputs = [input_node.output for input_node in node.input_nodes]
            node.output = node.compute(*inputs)

    def fit(self, node, learning_rate, epoch, feed_dict = {}):
        
        self.predict(node, feed_dict)
        node.backprop(learning_rate)
        self.iterator += 1
        if self.iterator % epoch == 0:
            self.iterator = 0
            for var in self.variable_lis:
                var.output -= var.del_val
        self.variable_lis = []



A = Variable(np.array([5]))
b = Variable(np.array([2]))
x = Placeholder()
y = multiply([A,x])
z = add([y,b])
y_true = Placeholder()
loss = reduced_mean([y_true], [z])

sess = Session()
for _ in range(30):
    temp = np.random.rand()
    sess.fit(node = z, learning_rate = np.array([1]), epoch = 5, feed_dict={x:[temp], y_true:[temp + 3]})

print(A.output)
print(b.output)
































