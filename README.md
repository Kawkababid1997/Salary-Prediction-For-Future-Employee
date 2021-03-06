# Predicting Future Employee Salary Based on Current Employee Attributes

Introduction:

The goal of this project was to cconstruct a model that would  predict the salary of a new job posting based on job attributes like jobId, companyId, jobType, degree, education level with area of study, length of experience and miles from metropolis. The model was then tested on a test data set to validate its accuracy.

The three datasets used for this project were train_features, train_salaries and test_features. The train feature dataset originally contained jobId, companyId, jobtype, major, degree, industry. The train salaries dataset contained the corresponding jobId associated with the salary. Then the two data sets were merged as train dataset using the jobId illustrated below.  
<img src = "image/train_df.png">
There is also a test dataset given without salary of employee which we will be determining throughout this model. 
<img src = "image/test_df.png">
The tool used was Python 3 along with its libraries and packages such as numpy, pandas, matplotlib, seaborn and sklearn to do data manipulation, data visualization and build the predictive model.

## Data Cleaning:
The data was found with no missing and duplicate values but there were five entries with salary <= 0. Therefore, these data were removed from the dataset. 
<img src = "image/zero-salary.png">
## Exploratory data analysis(EDA):
Numerical and categorical variables were identified and summarized separately. There are two numerical features - yearsExperience and milesFromMetropolis. The features jobId and companyId were not used to build the model. The categorical features are jobType with 8 unique values, degree, major and industry with 5,9,7 unique values respectively.

### Visualizing Target(Salary) Variable:
<p align = "center">
<img src = "image/salary-distribuition.png" width = 600, height = 300>
</p>
Based on the target variable's plot there were some suspicious potential outliers. Using Statistical Inter-Quartile Range, we found the upper and lower bound of suspected outliers. There were 20 Junior positions with salary above the upper bound 220.5. After investigating the data, it is clear that those data should be good to use since those employees have above 20 years of experience and most of them have masters or phd degree.
<img src = "image/upper_bound_salary.png">

### Relationship between Target and Input Variable:  
### From the EDA we can see that:
<p align = "center">
<img src = "image/salary-jobType.png" width = 600, height = 500>
</p>
There is a clear positive correlation between job type and salary. 
<p align = "center">
<img src = "image/salary-degree.png" width = 600, height = 500>
 </p>
More advanced degrees correspond to higher salaries. 
<p align = "center">
<img src = "image/salary-major.png" width = 600, height = 500>
</p>
People with majors of engineering, business and math generally have higher salaries. 
<p align = "center">
<img src = "image/salary-industry.png" width = 600, height = 500>
</p>
As for industries, oil, finance and web industries generally pay better. 
<p align = "center">
<img src = "image/salary-experience.png" width = 600, height = 500> 
</p>
In general there is a clear correlation between salary and years of experience. As the years of experience increase, salary increases.  
<p align = "center">
<img src = "image/salary-milesFromMetapolis.png" width = 600, height = 500>  
</p>
Salary decreases as the distance to metropolies increases. 
Apart from this to get an idea about the correlation between features, a heatmap was plotted.
<p align = "center">
<img src = "image/heatmap.png" width = 600, height = 500>
</p>

### Feature Engineering
The training data was cleaned, shuffled and reindexed and using one hot encoding categorical data was encoded to obtain the final training and test dataframes.
We concluded from Exploratory data Analaysis Heatmap that:
There is a weak positive relationship (0.38) between salary and yearsExperience. There is a weak negative relationship between (-0.3) salary and milesFromMetropolis. This prediction will be unreliable due to the weak correaltion. Therefore, we engineered new features to enhance model performance.

**New Features:Calculate descriptive statistics by aggregating categorical features (Eg: Group_mean, Group_min, Group_max, Group_std)**

### Model Selection and Evaluation:
The three different regreesion algorithms selected were 1.Linear Regression  2. RandomForest Regressor 3.Gradient Boosting Regressor

Mean Squared Error(MSE) was selected as the evaluation metric. The model with lowest MSE was selected as the best model.

### Best Model:
After doing 2 fold cross validation on each selected models, the following MSE was measured for corresponding models

1. Linear Regression - 384.49 (base model) However, it was reduced to 358.16 after implementing new feature to the train data set   

2. RandomForest Regressor - 314.88. 

3. Gradient Boosting Regressor - 313.36

So Gradient Boosting Regressor with the lowest MSE was selected as the best model. The model was trained on the entire data set and predictions were created based on the test data. Key predictors for this model were Group_mean followed by yearsExperience as shown in the Feature Importances plot.

#### Feature Importance:
<p align = "center">
<img src = "image/feature-importance.png" width = 600, height = 500>
</p>

## Conclusion:
The Predictive model is working fine and is able to predict salaries for the test dataset. The evaluation metric considered was MSE(Mean Squared Error). The MSE obtained for the model is 313.36.
