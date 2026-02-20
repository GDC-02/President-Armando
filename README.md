# President-Armando
President Armando is an ANN I realized in January 2025 as part of a group project for an exam; its objective is predicting the risk of developping Parkinson's disease using as inputs voice recordings.

---

# The dataset
The dataset we have used is a compendium of the voice recordings of 31 patients, some with Parkinson’s disease and some without, for a total of 195 entries, represented by the rows of our dataset. The columns are the features considered when analyzing the recordings; they are: 

• Name: ASCII subject name and recording number;

• MDVP:Fo(Hz): Average vocal fundamental frequency;

• MDVP:Fhi(Hz): Maximum vocal fundamental frequency;

• MDVP:Flo(Hz): Minimum vocal fundamental frequency;

• MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP: Several measures of variation in fundamental frequency;

• MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA: Several measures of variation in amplitude;

• NHR, HNR: Two measures of the ratio of noise to tonal components in the voice;

• Status: The health status of the subject that we have taken as the output of our neural network (One= Sick, has P.D. and Zero= Healthy);

• RPDE, D2: Two nonlinear dynamical complexity measures;

• DFA: Signal fractal scaling exponent;

• Spread1, spread2, PPE: Three nonlinear measures of fundamental frequency variation. 

We have divided the dataset, following the 80%-20% rule, into a training set of 155 data points and a test set of 40. This is to ensure that we have some actual data to confront the performance of the system in the phase of evaluation. 

---

# Data selection

From here, we developed the graphs and, by looking closely at the data, we took some choices in the sense 
of: 

• Cutting all features with relevance <10%;

• Cutting redundant features.

---

# Final result


We then normalized the data and programmed the Neural Network. President Armando has: 

• 9 input neurons (one for each relevant feature); 

• A first hidden layer composed of 5 neurons; 

• A second hidden layer composed of 3 neurons; 

• 1 output neuron (Has Parkinson? Y/N).

The model showed a test loss of  0.0505 and a test accuracy of 0.9750

---

# User experience

Finally, we have programmed a UI for a better user experience. 
To inquiry President Armando you need to input 9 different features taken from the voice recording of a patient. Evidently, you need a machine capable of detecting the same 9 features the network requires to run.

---

# Issues 

In the process of developing the network we have encountered a few issues. Firstly, regarding the dataset. 
We previously had a more exhaustive and diverse dataset, but it was lacking in clarity. It had over 200.000 datapoints but it was hard to decipher the meaning of the feature selected for the inquiry and a lot of the data was not taken homogeneously. For these reasons, we opted to look for a better and more reliable backbone for our network. The one we are using required little manipulation since it was very clear and tidy in the first place. The features are set out comprehensively, the data are gathered coherently with one another. Even though we must point out that this dataset is far from being considered complete and exhaustive. For the predictions of the network to be considered reliable, the network should be fed with a lot more data we are lacking. Still, we found the results to be satisfying enough for this project. A second type of obstacle we faced has to do with designing the network itself. Initially we struggled choosing the number of neurons and hidden layers composing the ANN as well as deciding the type of activation function for the neurons (sigmoid, linear etc.). To overcome this difficulty, we used a heuristic approach. That is to say that we kept in mind some specific parameters given by the machine (loss function, accuracy, learning rate) to evaluate the performance of the machine in relation to the changes we made. We found the current build to be the most stable and best performing.  The third notable difficulty was found in the choice of graphs. Easily overlooked, the graphical representation molds our perception of the data and their respective salience. Throughout the project we decided to change dispersion graphs for histograms or cartesian diagrams, as they gave us a better picture. 

In the end, we think we accomplished a very satisfying project. Despite the epistemic problem posed by the range of the dataset, our neural network is a good display of the capabilities and advantages of ANN. 
