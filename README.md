# Effect of top-down connections in Hierarchical sparse coding

This github repository reproduces all the results of the paper entitled "Effect of top-down connections in Hierarchical sparse coding" published at Neural Computaiton ([ArXiv link here](https://arxiv.org/abs/2002.00892)).

This python repository is organized as follow : 
- The folder /SPC_2L contains the package where the network 2L-SPC network, its training, and all necessary tools are coded.
- The folder /Savings contains most of the results of the simulation (so that you don't need to spend hours to retrain the network). When running the notebook, be carrefully to keep the variable 'Save' to False, otherwise it'll erase the previously saved results
- The notebooks:
    - The notebook with a a name starting with "1" are related to the training of the networks of the 4 tested databases ( ATT, MNIST, STL, CFD).
    - The notebook with a a name starting with "2" are related to ..
    
    
## Overview of the main results 

### Top-down connection allows a mitigation of the prediction error 
Results on CFD database when varying the first layer sparsity (see arXiv paper for other database and also the effect of varying the second layer sparsity):
![Prediction Breakdown on CFD when varying lbda1](/Savings/Fig/Fig2-a-ii.png "Prediction breakdown when varying the first layer sparsity")

### Top-down connection allows a faster convergence of the inference process
Results on CFD :

### Top-down connection refines the prediction made by the second layer
Results on STL :

### Top-down connection speeds-up the training of the network
Results on CFD :

### Top-down connections allows the learning of extended and more contextual RFs
Results on CFD : 
