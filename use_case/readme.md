The function ```make_simulator``` sets everything needed to evaluate one layout. The most important thing that is retrned is the callable function ```run_sim```, that executes one simulation and returns the fitness value. This function can be called later with different layout parameters.

This line of code:

run_sim, model, data = make_simulator()

must be placed inside the for loop in case of implementation of the theta switch. You must modify it toreceive as input the set of clusters it wants.