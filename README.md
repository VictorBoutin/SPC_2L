# Effect of top-down connections in Hierarchical sparse coding

This github repository reproduces all the results of the paper entitled "Effect of top-down connections in Hierarchical sparse coding" published at Neural Computaiton ([ArXiv link here](https://arxiv.org/abs/2002.00892)).

This python repository is organized as follow : 
- The folder /SPC_2L contains the package where the network 2L-SPC network, its training, and all necessary tools are coded.
- The folder /Savings contains most of the results of the simulation (so that you don't need to spend hours to retrain the network). When running the notebook, be carrefully to keep the variable 'Save' to False, otherwise it'll erase the previously saved results
- The notebooks. The notebook with a a name starting with "1" are related to the training of the network of the 4 tested database ( ATT, MNIST, STL, CFD).
