# RL-ABM
ABM financial market simulation environment for RL agent training.

The ABM has been calibrated using the set-up-calibration notebook and model_calibration python file. The model can be run using the Run model.ipynb notebook. There you can observe the output prices, fundamental, volume, and individual profits. 

The main structure of the model code is as follows. The model is a function in the model.py file. Before the model is run is always has to be initialized using the function in the initialize_model.py file. 

The model uses a couple of Python objects that are in the objects folder. The most important of these are the agent object and the orderbook object. These objects tend to store their own information during the simulations. The orderbook object stores price information. 

The model also uses a few helper functions that can be found in the helper function. 
