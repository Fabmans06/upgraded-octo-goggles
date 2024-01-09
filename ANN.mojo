from python import Python



fn main() raises:
    let dataPath = "archive/DatsetFraud.csv"

    let np = Python.import_module("numpy")
    let torch = Python.import_module("torch")
    let pd = Python.import_module("pandas")

    #Use of var so that dataset variable is mutable
    var dataset = pd.read_csv(dataPath)

    #TODO: Figure out how function calls work, like say I want to say:
    dataset = pd.DataFrame(dataset, columns=["isFlaggedFraud"])
    #That should filter the dataset to not include the 'isFlaggedFraud' column
    #However a call like 'columns = isflaggedfraud' is not allowed
    #nor is any call that specifies something this way, another example is: 
    #dataset = np.loadtxt(dataPath, delimiter=',')
    #This should load the data using numpy, but doesn't work because I can't specify the delimiter that way
    #So the todo is to figure out how to specify something like this

    

    print(dataset) 
    #let Input = dataset[:,0:8]
    #let Fraudulent = dataset[:,8]