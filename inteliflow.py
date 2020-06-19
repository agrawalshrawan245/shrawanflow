import numpy as np
import pdb

class Operation():
    
    def __init__(self, input_nodes = []):
        self.input_nodes = input_nodes # The list of input nodes
        self.output_nodes = [] # List of nodes consuming this node's output
        
        for node in input_nodes:
            node.output_nodes.append(self)

    def compute(self):
        
        pass

class Optimizer():

    def __init__(self, learning_rate, input_nodes = []):
        self.input_nodes = input_nodes
        self.learning_rate = learning_rate

        for node in input_nodes:
            node.output_nodes.append(self)

    def back_prop(self):

        pass


class add(Operation):
    
    def __init__(self, x, y):
         
        super().__init__([x, y])
    def compute(self, x_var, y_var):
         
        return x_var + y_var

class multiply(Operation):
     
    def __init__(self, a, b):
        
        super().__init__([a, b])
    
    def compute(self, a_var, b_var):
         
        return np.multiply(a_var, b_var)

class matmul(Operation):
     
    def __init__(self, a, b):
        
        super().__init__([a, b])
    
    def compute(self, a_mat, b_mat):
         
        return np.matmul(a_mat, b_mat)

class Placeholder():
    
    def __init__(self):
        
        self.output_nodes = []
        

class Variable():
    
    def __init__(self, initial_value = None):
        
        self.output = initial_value
        self.output_nodes = []
        

def run(node, feed_dict = {}):
    
    if type(node) == Placeholder:
        
        node.output = feed_dict[node]

    elif type(node) == Variable:
        node.output = np.array(node.output)
        
    else: # Operation
        node.inputs = []
        for input_node in node.input_nodes:
            node.inputs.append(run(input_node, feed_dict))
            node.output = node.compute(node.inputs)
            
        return node.output

# def train(self, losses, operation, feed_dict = {}):




A = Variable(np.array([[10,20],[30,40]]))
b = Variable(np.array([1,1]))
x = Placeholder()
t = Variable(np.array([10,20]))
y = matmul(A,x)
z = add(y,b)
result = run(node = z,feed_dict={x:10})
print(result)
# print(y.compute(np.array([[10,20],[30,40]]), np.array([10,20])))




