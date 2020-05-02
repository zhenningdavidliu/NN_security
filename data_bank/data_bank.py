from .Data_loader_shades import Data_loader_shades
from .Data_loader_lines import Data_loader_lines
from .Data_loader_existing_lines import Data_loader_existing_lines
from .Data_loader_grad_lines import Data_loader_grad_lines
from .Data_loader_grad_shades import Data_loader_grad_shades
from .Data_loader_stripe import Data_loader_stripe_test, Data_loader_stripe_train
from .Data_loader_lines2 import Data_loader_lines2
from .Data_loader_shades2 import Data_loader_shades2
from .Data_loader_shades3 import Data_loader_shades3
from .Data_loader_snake import Data_loader_snake

def data_selector(data_name, arguments):
    """ Select a data loader based on `data_name` (str).
Arguments
---------
data_name (str): Name of the data loader
arguments (dict): Dictionary given to the constructor of the data loader

Returns
-------
Data loader with name `data_name`. If not found, an error message is printed
and it returns None.
"""
    if data_name.lower() == "shades":
        return Data_loader_shades(arguments)
    if data_name.lower() == "shades2":
        return Data_loader_shades2(arguments)
    if data_name.lower() == "shades3":
        return Data_loader_shades3(arguments)
    elif data_name.lower() == "lines":
        return Data_loader_lines(arguments)   
    elif data_name.lower() == "lines2":
        return Data_loader_lines2(arguments)   
    elif data_name.lower() == "load_lines":
        return Data_loader_existing_lines(arguments) 
    elif data_name.lower() == "glines":
        return Data_loader_grad_lines(arguments) 
    elif data_name.lower() == "gshades":
        return Data_loader_grad_shades(arguments) 
    elif data_name.lower() == "stripe_train":
        return Data_loader_stripe_train(arguments) 
    elif data_name.lower() == "stripe_test":
        return Data_loader_stripe_test(arguments) 
    elif (data_name.lower() == "lines2"):
        return Data_loader_lines2(arguments)   
    elif (data_name.lower() == "shades2"):
        return Data_loader_shades2(arguments)
    elif (data_name.lower() == "snake"):
        return Data_loader_snake(arguments)
    else:
        print('Error: Could not find data loader with name %s' % (data_name))
        return None;

