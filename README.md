# Sentiments Analysis of IMDB Reviews

Sentiment analysis is one of applications where Machine Learning shows promising performance. In this project, we will conduct sentiment classification using several ML methods. Decision Tree, SVM, Linear Regression and KNN are used in this repository. For each of these, we have done a grid search to find the best parameters in terms of the performance. 
## Dataset
You can download the dataset from [here](https://drive.google.com/open?id=1kiRwts8yhw4E-MM82_VcIRkRhlXsBnWG) and extract them to the main folder of the project. You should see a train and a test folder. 
## Preprocessing
In order to make the results better, we have cleaned up the stopwords from all the reviews. You can find the list of stopwords in "methods.py" and even modify that according to your requirements.
## Getting Started
In order to run the code, you can use a Python IDE like PyCharm or Spyder. 
### Prerequisites
Install the following packages: 
* os == 1.3
* Scikit-Learn
* NLTK
* csv (if you want to export the test set predictions)
* NumPy
### Experiment
Download the repository in your local computer. 
```
git clone https://github.com/PouriaCh/IMDB-Sentiment-Analysis.git
```
If all of the required packages are installed, you will have the results. Feel free to modify the parameters in the code.
## Acknowledgement
This project was done as part of the COMP551 (Applied Machine Learning) course in McGill University.
