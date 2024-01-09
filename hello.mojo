from python import Python


   
#The 'raises' is needed because loading a module 
#is a function that might raise an error, so 
#its parent function needs to be able to aswell
fn main() raises:
	print("Hello, world!")

	#This is the equivalent to 'import numpy as np':
	let np = Python.import_module("numpy")
	#Now numpy can be used as if this were Python
	let array = np.array([1,2,3])
	print(array)

	#This is the equivalent to 'import torch as torch':
	let torch = Python.import_module("torch")
	#Now torch can be used as if this were Python
	let x = torch.rand(5,3)
	print(x)