# Document-Classification-with-CNN

#### Objective
Preprocess docs, train multiple ConvNets to classify docs with achieve upto 70% accuracy etc.

#### Problem Statement

    There are total 18828 documents (text files) and those belongs to 20 different types
    File name contains the label. The string before _
    Specific document to be classified in to one of the class

#### Solved By
Pradip Dharam

#### Covered Roadmap
* Part 1: Preprocessing of all 18828 files. Features preprocessed_email, preprocessed_subjects, preprocessed_texts to be extracted.
* Part 2: Multiple ConvNets models are trained.
* Part 3: Multiple ConvNets models are trained. Another architecture than previous part 2.



### Solution Observations
#### Preprocessing Insights

  * Removed the stemming and lemmatization because GloVe embedding does not have stemmed and lemmatized words

#### Model 1
##### Behavioral

  * Do not be in hurry to build model architecture to save the time, it will cost you more time debugging
  * Do not feel tense and overwhelmed if the model architecture defined by you is not working.
  * Instead, have cup of coffee or go for one day long drive, feel sensible and continue altering the model architecture.
  * Try to focus on something which you find more daunting or more easy. There is high chance that issue lies there.
  * Dont give up because you feel frustrated, tense and less confident.

##### Technical

  * Always print the results to ensure whether its working the way you expect.
  * I wanted to check word count but, I was checking character count by mistake. Scentence length was later reduced from 2000 to aroung 250 words. By mistake, I was considering 2000 as maximum words in sequence, those are in fact number of characters. This was introducing more number of zeros abd less sequence numbers in the 'number sequences' created from sequences. It was then hampering the neural networks training .
  * By mistake, I was flattening the already flattened layer. Model was really misbehaving. High accuracy 0.85 and low f1 score 0.05 around. And vice versa when learning rate was altered. Faced the issue of exploding gradients, controled exploding gradients by gradient clipping by option of keras Adam optimizer clipnorm=1.
  * Model training stopped misbehaving when one of the Flatten later was removed. Just kept onr Flatten layer in the model.
  * Less kernel size captures more detailed word level features. Larger kernel size captures the global meaning but not the detailed meaning. Less kernel size is used which is 2 since Conv1D, if it was Conv2d then the kernel size could be 2x2. Kernel size was changed from 5 to 2 here.
  * ReLU activation and 'He Uniform' weight initializer was provided to all Conv1D
  * Do not create the instance of 'He Uniform' kernel initializer first and do not assign that instance to all Conv1D layers; because, it initializes all Conv1D kernels to same values, hence wont help capture the different aspects of features from the input data
  * Instead, directly assign to kernel_initializer argument as below.
  * i1 = Conv1D(8, 2, padding='same', strides = 1, activation='relu', kernel_initializer = tf.keras.initializers.HeUniform())(x)
  * Writing custom keras metric does not accept normal python variables, it accepts to calculate on top of tensors.
  * Total words in vocablary of entire training corpus are 86583. In the GloVe vector; the 34781 numbers of words not found, zero value used for all dimensions in the embedding metrics.
  * Use the activation function ReLu with He initializer. And try to keep default learning rate 0.001 for Adam optimizer, adam in most of the cases works better with that default learning rate
  * For almost 18% of the documents in training set, class cannot be predicted.
  * For almost 24% of the documents in test set, class cannot be predicted.
  * Model is bit overfitted but can give acceptable results
  * train_acc: 0.857, test_acc: 0.666
  * train_fbeta_score: 0.840, test_fbeta_score: 0.668


#### Model 2

  * Always print the results to ensure whether its working the way you expect.
  threshold=0.5 for f1 beta score is removed, let it consider the argmax so that I can see class labels after predictions.
  * Since no class is able to be prected since probabilities found are below 0.5 and we find no class predicted, though I find almost 10% accuracy.
  * Model highly overfits with kernel size 2; made kernel size 10 for first 2 conv1d and 5 for next 2 conv1d.
  * Number of filters for first 2 conv1d are redulced to 5, for next 2 conv1d those are reduced to 2; this is to control overfitting.
  * Drop out rate was increased from 0.1 ro 0.3 to control the overfitting.
  * Model 2 looks overfitted. Training data has good performance metrics over test data.
  * train_acc: 0.219, test_acc: 0.111
  * train_fbeta_score: 0.219, test_fbeta_score: 0.111


#### Disclaimer: This problem statement is been solved purely for purpose of learning and not to earn money. Dataset obtained from public domain.
