# standard library
import sys

# user library
#sys.path.append('./src')
from src import mnist_loader
from src import network2
training_data, validation_data, test_data = \
    mnist_loader.load_data_wrapper()
net = network2.Network([784, 10])
net.SGD(training_data[:1000], 30, 10, 10.0, lmbda = 1000.0, \
    evaluation_data=validation_data[:100], monitor_evaluation_accuracy=True)