'''
1 select the training set, 
5 differentiate categorical features from numeric features, 
2 create feature matrix, 
3 create class label vector, 
4 pick out low variance feature(s), 

6 transform skewed feature(s) by Box-Cox Transformation, 
7 perform the dimensionality reduction by Principal Component Analysis (PCA), 
and filter out redundant (numeric) features according to their correlation. 

Finally, tell me what kind of unsupervised learning you would like to do, 
and what we could expect from such exploratory work.

===============================================================================
ANSWER:
    We already have the annotated data with well labeled class for each 
    tuple and we have 1009 samples, therefore it is common for us to adopt
    supervised learning to explore the f(x) with 100 predictors as input and 
    cell type as output (ws or ps). In this case, i would design a MLP with 
    2 hidden layers:
        the first layer is the input layer, size 100;
        the seconde layer is the first hidden layer, size 5;
        the third layer is the second hidden layer, size 4;
        the fourth layer is the output layer, size 1.
        in summary, there are around 100*5 + 5*4 + 4*1 = 524 parameters,
        which is about the half of the number of sampls.
    using Relu as activate function and the loss function will be:
        LOSS = cross entropy of two possibility distributions, one is the target,
        i.e. (0,1) or (1,0)
        another one is the predicted possibility, i.e. (0.4,0.6) or (0.99,0.01)
        
        
    About the unsupervised learning we can apply to explore the relationship 
    between these predictors and the cell class, i would like to adopt VAE.
    However, we cannot expect good result from this model because the number of 
    parameters for VAE is about the twice of MLP. And due to the limited number 
    of samples, the problem of overfitting is hard to avoid.
    
===============================================================================

'''
import pandas as pd
import matplotlib.pylab as plt
from sklearn import preprocessing
from scipy.stats import skew

# read the data 
cell = pd.read_csv('segmentationOriginal.csv')
#check the columns
cell.head()
cell.describe() # =summary in R

# check if there are any missing values
cell.isnull().values.any() # ->false (means no mising value)

# check the values for 'Case'
cell['Case'].unique() # -> array(['Test','Train'], dtype=object)

## 1, select the taining set
cell_train = cell.loc[(cell['Case']=='Train')]
cell_train.head()
cell_data= cell_train # for later to create class label

## 2, create the feature matix
# 2.1 Remove the first three columns (identifier columns) because they are not related with the feature.
cell_train = cell_train.drop(['Cell','Class','Case'], axis=1)
cell_train # 166 columns left

# 2.2 only mantain the numeric data
## 5, differentiate categorical features from numeric features
series1=pd.Series(cell_train.columns).str.contains("Status")
index1=cell_train.columns[series1]
len(index1) # 58
numeric_cell_train=cell_train.drop(index1, axis=1)   
numeric_cell_train # 58 columns left
numeric_cell_train.describe()



## 3, create class label vector
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

le_class=LabelEncoder().fit(cell_data['Class'])
Class_label=le_class.transform(cell_data['Class'])
Class_label.shape # (1009, )

## 4, pick out low variance features
cell_train.var()
# there are two predictors with 0 variance
# so delete the columns with name 
#'MemberAvgAvgIntenStatusCh2' and 'MemberAvgTotalIntenStatusCh2'
cell_train4=cell_train.drop(['MemberAvgAvgIntenStatusCh2'],axis=1)
cell_train4=cell_train.drop(['MemberAvgTotalIntenStatusCh2'],axis=1)

## 6, transform skewed feature(s) by Box-Cox Transformation

skness = skew(cell_train4) # the largest is 12.879 [64] none of them are largger than 20

centered_scaled_data = preprocessing.scale(cell_train4)
type(centered_scaled_data) # numpy.array

from scipy.stats import boxcox

l=cell_train4.iloc[0:1,0:114]
l2=cell_train4.iloc[:,64]
type(l2) # Series


#boxcox_transformed_data = boxcox(cell_train4.iloc[:,64]+2) # type : tuple
cell_train6=pd.DataFrame()
# do box cox transformation for each column
for i in range(0,115):    
    boxcox_transformed_data = preprocessing.scale(boxcox(cell_train4.iloc[:,64]+2)[0])
    cell_train6.insert(i,i,boxcox_transformed_data)

# check the skewness again
skness2=skew(cell_train6)
                             
## 7, PCA
from matplotlib.mlab import PCA
results = PCA(cell_train6)




