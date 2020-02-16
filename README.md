# Sentiments Analysis of IMDB Reviews

Sentiment analysis is one of applications where Machine Learning shows promising performance. In this project, we will conduct sentiment classification using several ML methods. <br>


## Dataset
The dataset is a .json file of a NumPy list containing 12000 comments. Each comment is a dictionary with the following keys:
Please download the dataset files from [here](https://drive.google.com/open?id=1kiRwts8yhw4E-MM82_VcIRkRhlXsBnWG) and extract it to the main folder of the project.
## Preprocessing
As the comments are taken from real discussions, we need to remove irrelevant information (such as numbers, punctuations, etc.) 
from each comment. This increases the performance of the regressor.
## Getting Started
In order to run the code, you can either use a Python3 kernel in your Jupyter Notebook or any Python IDE. 
### Prerequisites
Install the following packages: 
* os == 1.3
* Scikit-Learn
* NLTK
* csv (if you want to export the predictions for test set)
* NumPy
### Experiment
Download the repository in your local computer. 
```
git clone https://github.com/PouriaCh/IMDB-Sentiment-Analysis.git
```
If all of the required packages are installed, you will have the results. 
## Acknowledgement
This project was done as part of the COMP551 (Applied Machine Learning) course in McGill University.
