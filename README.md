# DA6401 A3
#### Apoorva Prashanth (CE21B020)

This project implements a sequence-to-sequence neural network (RNN/GRU/LSTM) to perform character-level transliteration from English to Malayalam.

-------

Set Up (Option 1 Local Environment)
- Clone this repository<br>

1. Clone the repository:
   ```bash
   git clone https://github.com/Apoorva-Prashanth/CE21B020_A3)
   cd <repo-folder>
   ```
   <br>
  Means: go into the folder that was created when you cloned the GitHub repository.
- ```pip install torch tqdm wandb``` <br>

- ```import wandb```<br>
```wandb.login(key="your-api-key")```

--------
Q1: Run the following command to build the model with desired parameters:

`python Q1.py --embedding_dim 128 --hidden_dim 256 --num_layers 2 --rnn_type GRU`

You can use any parameter of your choice. This question focuses on building the model architecture.

----- 
Q2 Hyperparameter Sweep with W&B

- Edit the sweep configuration if needed (in Q2.py).<br>
- Login to your Weights & Biases account.<br>
- Run the code from your terminal: <br>
```python Q2.py``` <br><br>

-----
Q3 Theory, explained in W&B report<br>

-----
Q4 Best Model & Confusion Matrix
- Sweep configuration is set to the best model’s hyperparameters.
-  Code to generate the confusion matrix is included.
-   Outputs of this section include Figure 1, 2, and 3.
- Run ```python Q4.py``` in terminal

------
    
Q5 and Q6 after following the set up  <br>
- Follow the same procedure as Q2 and Q4:
- Ensure dataset paths are correct
- Modify the sweep configuration if required
-  Run the corresponding scripts for Q5 and Q6

_For all questions, set the data set the correct local path file of the data set_


-------
Set Up (Option 2, Using Cloud)
- Upload the ```a3_da6401.ipynb``` in Google Colab <br>
- Enable T4 GPUs ```Runtime > Change runtime type > Hardware accelerator: GPU```
- Upload the data set manually
- Set the path to data set correctly
- Run the cells


