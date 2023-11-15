# Real Estate Price Prediction

This repository contains code for predicting house prices using **Multiple Linear Regression**, **Support Vector Regression**, and **Random Forest Regression**.

## Table of Contents
- [Introduction](https://github.com/thisarakaushan/Real-Estate-Price-Prediction/edit/main/README.md)
- [Data](./dataset/data.csv)
- [Models](https://github.com/thisarakaushan/Real-Estate-Price-Prediction/tree/main/notebook)

## Introduction

In this project, we explore the task of predicting real estate house prices using three different Regression techniques: Multiple Linear Regression, Support Vector Regression, and Random Forest Regression. The aim is to compare the performance of these models and understand their strengths and weaknesses in predicting house prices.

## Data

We used a dataset containing various features of houses, such as the number of bedrooms, square footage, location, etc. The dataset was preprocessed to handle missing values and perform feature scaling.

## Usage

1. **Clone the Repository:**
   
```
   git clone https://github.com/thisarakaushan/Real-Estate-Price-Prediction.git
```

## Models

### [Multiple Linear Regression](https://www.javatpoint.com/multiple-linear-regression-in-machine-learning)

Multiple linear regression is a basic regression technique that models the relationship between the independent variables and the dependent variable.

Multiple linear regression is a widely used method for predicting continuous target variables. It assumes that the relationship between the target variable and the independent variables is linear, meaning that a change in the value of one independent variable is associated with a constant change in the value of the target variable. Multiple linear regression models are relatively easy to interpret and can provide insights into the specific relationships between the independent variables and the target variable.

To use multiple linear regression for real estate price prediction, you would first need to gather a dataset that includes information on the various factors that influence real estate prices, such as location, size, number of rooms, age of the property, and so on. Once you have collected this data, you would split it into a training set and a test set. You would then use the training set to build a multiple linear regression model that predicts the price of a property based on its various features. Once the model is built, you would evaluate its performance using the test set, which would give you an idea of how well the model generalizes to new data.

### [Support Vector Regression](https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/)

Support vector regression is a regression technique that aims to find a hyperplane that best fits the data points while allowing a certain amount of error.

Support vector machine (SVM) is a machine learning algorithm that can be used for both classification and regression problems. Unlike multiple linear regression, SVM is a non-linear model that can capture more complex relationships between the independent variables and the target variable. SVM works by finding the hyperplane that best separates the data into different classes. For regression problems, the hyperplane is a line that provides the best fit to the data.

To use SVM for real estate price prediction, you would follow a similar process to that for multiple linear regression. You would first gather a dataset that includes information on the various factors that influence real estate prices. You would then split the data into a training set and a test set and use the training set to build an SVM model that predicts the price of a property based on its various features. Once the model is built, you would evaluate its performance using the test set.

When comparing multiple linear regression and SVM for real estate price prediction, there are several factors to consider. Multiple linear regression is a simpler model that assumes a linear relationship between the independent variables and the target variable. This can be an advantage if the relationship between the variables is actually linear, as it makes the model easier to interpret and understand. However, if the relationship between the variables is non-linear, then multiple linear regression may not perform well. SVM, on the other hand, can capture non-linear relationships between the variables, making it more flexible and potentially more accurate. However, SVM can be computationally intensive and may be more difficult to interpret.

Ultimately, the choice between multiple linear regression and SVM will depend on the specific requirements of your project. If the relationship between the independent variables and the target variable is known to be linear, then multiple linear regression may be the best choice. If the relationship is suspected to be non-linear or the data is noisy or high-dimensional, then SVM may be a better choice. It is important to carefully evaluate the strengths and weaknesses of each method and select the one that best fits your specific needs.

### [Random Forest Regression](https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/)

Random forest regression is an ensemble technique that creates multiple decision tree regressors and averages their predictions to improve overall accuracy.

To analyze house price prediction using the random forest model, you will need to consider the following main parts:

1. Dataset Preparation: This involves collecting, cleaning, and preparing the data to train the model. You will need to identify the features and target variable for the model and ensure the data is formatted properly for the random forest algorithm.

2. Feature Selection: You will need to determine which features are most important for predicting house prices. This can be done using techniques such as correlation analysis and feature importance scores.

3. Random Forest Model Training: This involves training the random forest model on the prepared dataset. You will need to select appropriate hyperparameters such as the number of trees, maximum depth, and minimum samples per leaf to optimize the model's performance.

4. Model Evaluation: You will need to evaluate the performance of the model on a separate test set to assess its accuracy and determine whether any overfitting has occurred.

5. Fine-tuning the Model: If the model is not performing as well as expected, you may need to fine-tune the model by adjusting the hyperparameters or adding or removing features from the dataset.

6. Predictions and Interpretations: Finally, once you have trained and evaluated the model, you can use it to make predictions on new data and interpret the results to gain insights into what features are driving the predicted house prices.

## Results

After training and evaluating the three models, we found that...

We used the Boston Housing Dataset and the SVM method to predict the median value of owner-occupied homes. We first explored the dataset using histograms and scatter plots to gain insights and identify patterns. We then split the dataset into training and testing sets, and trained a SVM model on the training set. We evaluated the model's performance on the testing set using the RMSE, and visualized the relative importance of each feature using a bar chart. The results suggest that the SVM method can be an effective approach for real estate price prediction, and that the number of rooms, the proportion of lower status of the population, and the distances to employment centers are important factors to consider.

There are many other machine learning algorithms and techniques that can be used, and the choice depends on the specific requirements and constraints of the problem. Additionally, feature engineering, data cleaning, and hyperparameter tuning are important steps that can improve the performance of the model.

Based on the results, you can choose the best method for your problem. Generally, the method with the lowest mean squared error and highest R-squared value is considered the best. However, it's also important to consider factors such as model complexity, interpretability, and computational efficiency.

In this case, based on the performance metrics, it appears that SVM is the best method for real estate price prediction on the Boston Housing Dataset. However, it's always a good idea to try multiple methods and evaluate their performance on multiple metrics to ensure that you have made the best choice for your specific problem.

MSE measures the average squared difference between the predicted and actual values. RMSE is simply the square root of the MSE, which gives a metric that is more interpretable in the same units as the target variable. R-squared measures the proportion of variance in the target variable that is explained by the model. A higher R-squared value indicates a better fit.

After fitting each model to the training data and making predictions on the test data, I printed the performance metrics for each algorithm. Based on these results, I concluded that SVM performed the best on this particular dataset, as it had the lowest MSE and RMSE values and the highest R-squared value.

However, it's important to note that the choice of the best algorithm can depend on various factors, such as the size of the dataset, the complexity of the problem, and the specific goals of the analysis. Therefore, it's always a good idea to try multiple methods and evaluate their performance on multiple metrics to ensure that you have made the best choice for your specific problem.

Based on the analysis of the three algorithms (Random Forest, Multiple Linear Regression, and Support Vector Machine), the SVM algorithm appears to be the best method for real estate price prediction on this dataset. Here are some important points to note:

The SVM algorithm had the lowest MSE and RMSE values and the highest R-squared value compared to the other two algorithms. This indicates that it produced more accurate predictions.

### Conclusion:

Based on these metrics, it looks like the SVM model outperforms the other two models in terms of predictive accuracy, as it has the lowest MSE and RMSE, and the highest R-squared score. The Multiple Linear Regression model has a lower predictive accuracy compared to the other two models, as it has the highest MSE and RMSE, and the lowest R-squared score, but it is still able to capture some of the patterns in the data, as evidenced by its non-zero R-squared score. T

Based on the performance metrics you have provided, the SVM model appears to be the best model for the Boston Housing dataset, as it has the lowest Mean Squared Error (MSE) and Root Mean Squared Error (RMSE), and the highest R-squared score. These metrics indicate that the SVM model is able to make more accurate predictions than the other two models you evaluated (Multiple Linear Regression and Random Forest). Therefore, if you are looking for the most accurate model for predicting housing prices in Boston, the SVM model would be the recommended choice based on your analysis.

## Feel Free to Contribute

If you find any issues with the code or have suggestions for improvements, feel free to open an issue or submit a pull request. Contributions from the community are highly welcome and appreciated. Let's work together to enhance the accuracy and applicability of our house price prediction models.

Happy coding!
