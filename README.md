# BANK-MARKETING-EFFECTIVENESS-PREDICTION
The dataset is from a banking institution's direct marketing campaigns where they made phone calls to potential clients to promote their term deposit. The goal of this project is to build machine learning models that can predict whether a client will subscribe to the term deposit or not. To achieve this, we used several machine learning algorithms like Logistic Regression, Decision Tree, Random Forest Classifier, K nearest Neighbour Classifier, and Naive Bayes Algorithm.

We started by importing the dataset and analyzing it by checking its dimensions, datatype, and statistical description. We also counted the null values of each column and replaced them with appropriate values to make the dataset ready for exploratory data analysis. We performed several types of analysis such as Univariate, Bivariate, and Multivariate analysis to get a better idea of the dataset. We formulated some Hypothesis and did hypothesis testing using different statistical tests like Z-test, Chi Square test, and accepted or rejected the hypothesis based on the p-values.

Then we proceeded to feature engineering where we manipulated some columns, imputed missing values, handled outliers using Z score method, and converted data into numerical features using techniques like One Hot encoding, Binary encoding, and get dummies. We split the data into 70:30 using the train-test split method and handled data imbalance using the SMOTE technique. We also scaled the data and selected important features for modeling.

We applied machine learning classification models like Decision Trees, Random forest, Logistic regression, KNN, and Naive one by one and used hyperparameter tuning to increase the efficiency of the models. We evaluated our model using various evaluation metrics such as Confusion matrix, Accuracy, Precision, Recall, F1 Score, and ROC-AUC Curve.

After comparing the performance of all the models, we found that Random Forest was the most accurate with 87% accuracy, 39% Precision, and 33% Recall values. But for our purpose, Logistic Regression was the best performing model with 77% accuracy, 80% Precision, and 73% recall since we wanted to predict most true positives.

Finally, we used a new dataset to test our model's prediction and provided some business insights helpful for the banking institution.
