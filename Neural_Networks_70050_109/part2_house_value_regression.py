import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold, train_test_split

class Regressor():

    def __init__(self, x, minibatch_size = 1000,
                 hidden_layers = [64, 64], 
                 activations = nn.ReLU, 
                 nb_epoch = 400, optimizer = optim.Adam, lr = 0.01):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own       

        self.lb= LabelBinarizer()
        self.norm_min_x= 0
        self.norm_max_x= 0
        self.norm_min_y= 0
        self.norm_max_y= 0
        self.impute_x=pd.DataFrame()

        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 

        self.loss_fn = nn.MSELoss()
        self.minibatch_size = minibatch_size        # batch is the whole dataset, minibatch is for batch learning
        layer_sizes = [X.shape[1]] + hidden_layers + [1]
        self.activations = [activations]*(len(layer_sizes) - 2)

        seq_input = np.array([[nn.Linear(layer_sizes[i], layer_sizes[i + 1]), self.activations[i]()] for i in range(len(layer_sizes[:-2]))]).flatten().tolist() + [nn.Linear(layer_sizes[-2], layer_sizes[-1])]
        self.model = nn.Sequential(*seq_input)

        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.loss_list = []

        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        if training==True:
            
            self.lb.fit(x["ocean_proximity"]) #save labelBinarizer 
            
            self.norm_min_x=x.iloc[:,:-1].min(skipna=True)
            self.norm_max_x=x.iloc[:,:-1].max(skipna=True)

            if not y is None:
                self.norm_min_y=y.min()
                self.norm_max_y=y.max()

            self.impute_x=pd.concat([
                x.iloc[:,:-1].mean(),
                x.iloc[:,-1].mode()
            ], axis=0).set_axis(x.columns) #impute_x is the parameter we store 

       
        x=x.fillna(self.impute_x)
        #min max normalisation 
        x.iloc[:,:-1]= (x.iloc[:,:-1]- self.norm_min_x)/ (self.norm_max_x-self.norm_min_x)

        if y is not None:
            y= (y-self.norm_min_y)/(self.norm_max_y-self.norm_min_y)    

        x=x.join(pd.DataFrame(self.lb.transform(x["ocean_proximity"]), columns=self.lb.classes_, index=x.index)) #apply the labelbinarizer
        x=x.drop(columns=["ocean_proximity"])
       
      
        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None
        return x.to_numpy(), (y.to_numpy() if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget

        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)

        for epoch in range(self.nb_epoch):
            for inputs, labels in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                self.loss_list.append(loss.item())
                loss.backward()
                self.optimizer.step()

            # print(f'Epoch {epoch+1}/{self.nb_epoch}, Loss: {loss.item()}')
        
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(X_tensor)

        self.model.train()

        y_reverted = predictions.numpy() * (self.norm_max_y.to_numpy()[0]-self.norm_min_y.to_numpy()[0]) + self.norm_min_y.to_numpy()[0]
        return y_reverted

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # _, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        predictions = self.predict(x)

        return mse(y, predictions, squared=False)


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
    
    def get_loss(self):
        return self.loss_list


def save_regressor(trained_model): 

    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def example_main():

    output_label = "median_house_value"


    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    x, x_test, y, y_test = train_test_split(x_train,y_train,test_size=0.2,train_size=0.8)
    x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.25,train_size =0.75)
    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_val, y_val)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

