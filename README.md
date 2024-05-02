# Geopolitics-ML-model


The sources of data used to train these models are as follows:

The V-Dem and V-Party Datasets
The Geopolitical Risk index
The OECD's Historical Population Dataset
The Augmented Human Development Index
The Natural Resource Rent dataset
The International Disaster Database
Cities Database By Country.

The data was preprocessed using the functions in preprocessing.py, such that it was ordered 
with years going across and countries going down

In order to create and run a model, use main.py to create an instance of Instance().
Then, simply call Run() to train the model for a number of epochs, and TestModel() to find its accuracy.
You should not need to interact with any other files.