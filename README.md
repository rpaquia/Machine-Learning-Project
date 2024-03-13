# CSE 151 Project - Cinematic Success Unveiled: A Data-Driven Analysis of the Film Industry

## Introduction

In the captivating world of cinema, movies not only serve as a source of entertainment but also as significant cultural and economic entities. The dynamics of movie success are influenced by a myriad of factors ranging from budget allocations to audience reception and critical acclaim. With the advent of data science and machine learning, the film industry now has the tools to unravel these complexities. Our project, leveraging advanced analytical techniques and machine learning models, aims to dissect the intricate web of variables that contribute to a movie's commercial success and critical reception. By analyzing comprehensive datasets, we strive to uncover the hidden patterns and relationships that can inform future filmmaking and marketing strategies, paving the way for a data-driven approach to cinematic art and commerce. Our investigation encompasses the following primary datasets,where it provides a valuable insights into different aspects of the movie world:

 **Kaggle Movies Dataset by Daniel Grijalvas**: This dataset contains extensive information about movies, including data on budget, revenue, and ratings.

Our objective is to employ a rigorous analytical approach to uncover intricate relationships between various factors in the film industry. Specifically, we seek to:

- Investigate the correlation between movie budgets and box office performance, offering insights into the financial dynamics of movie production.
- Analyze the impact of both critical and commercial reception on a movie's financial outcomes, identifying what makes a movie resonate with audiences and critics alike.
- Uncover trends in audience preferences across different eras, contributing to a better understanding of changing tastes and expectations.
- Employ machine learning models to predict commercial success and audience reception, aiding stakeholders in making informed decisions.

## Research Goals

Through this multidimensional analysis, our research group endeavors to provide a holistic understanding of the factors contributing to a movie's success, encompassing both its commercial and critical acclaim. Our findings aim to offer valuable insights to stakeholders involved in the filmmaking process, marketing, and industry analysis.

## Figures

In our study, we explore the nexus of film economics and critical success within the realm of entertainment analytics. Our focus is on deciphering how budget allocations and gross revenue impact movie ratings, with the goal of unveiling patterns and insights that could guide producers, marketers, and stakeholders in navigating the financial and creative aspects of the film industry. By employing both a polynomial regression model and a neural network, we aim to dissect the intricate dynamics at play, providing a comprehensive understanding that goes beyond mere surface-level correlations. This endeavor not only enriches academic discussions in film studies but also equips industry practitioners with data-driven insights for strategic decision-making in movie production and distribution.

For a detailed examination of our analytical models, please refer to the following resources:

- Neural Network Analysis: [Explore the Neural Network Model](https://colab.research.google.com/drive/1C9Mwf1J2ril1Q4l6n2BjQMb8YaFySG5_?usp=sharing#scrollTo=2zFcNpKzjEhY)
- Polynomial Regression Insights: [Dive into the Polynomial Regression Model](https://colab.research.google.com/drive/1qJACT9ZokWtD22lQeOtTRRe_XmRPq0q_?usp=sharing)
- Decision Tree Classifier Observations:
  - [View Decision Tree Classifier Model 1](https://colab.research.google.com/drive/1PKweOlRDaFI8GZ222odxu2pztfsBV98l?usp=sharing)
  - [Explore Decision Tree Classifier Model 2](https://colab.research.google.com/drive/1OptVsB2DknM0rG3sQdZ-BIz7Vseu5fum?usp=sharing)

This streamlined approach enhances readability and accessibility, ensuring that your audience can easily navigate to the specific models and analyses that underpin your study.

## Machine Learning Model

For the purpose of this research, we have employed a robust machine learning model, the **Polynomial Regression**, **Neural Network Model**, and **Decision Tree Classifier**  to aid in our analysis and predictions. This model enables us to make data-driven decisions and draw meaningful conclusions from the complex datasets under scrutiny.

### Model 1: Polynomial Regression

### Evaluate our First Model Compare Training vs Test Error

All three degrees 2, 3, and 4 are showing a good fit as seen by the testing MSE being lower than or very close to the training MSE. There is no evidence of overfitting, where we would expect the testing MSE to be significantly higher than the training MSE due to the model capturing noise in the training data.

### Model Fit In The Fitting Graph

Based on our MSE values, as the polynomial degree increases from 2 to 4, both training and testing MSEs decrease. This indicates that our model is capturing more of the data's underlying pattern with increased complexity, improving its performance. Our model with polynomial degree 4 have the lowest MSEs suggests an optimal balance between bias and variance, making it the best fit among the ones that we tested. There's no sign of overfitting as both training and testing error decreases together. In conclusion, our polynomial regression model with degree 4 provides the best balance of complexity and performance based on our data.

### Next Model Considerations

In light of the insights derived from the performance of our current neural network model, we are considering the exploration of two additional machine learning models to potentially enhance our predictive capabilities and address the identified limitations:

- **Neural Network**: Neural networks are indeed well-suited to handle the complexity inherent in modeling the relationship between a movie's budget and its gross revenue, thanks to their ability to model nonlinear relationships and interactions between features without the need for manual feature engineering. Unlike polynomial regression, which requires choosing the degree of polynomials a priori and risks overfitting with higher degrees, neural networks can learn complex patterns through their hidden layers and neurons. They do this by adjusting weights and biases through backpropagation based on the error rate, allowing them to capture both high-level and subtle nuances in data. Moreover, neural networks can automatically discover the interaction between variables, making them a powerful tool for capturing the multifaceted dynamics of movie revenues. The flexibility and adaptability of neural networks, combined with techniques to prevent overfitting such as dropout and regularization, make them an attractive option for improving upon traditional regression models in predicting outcomes with complex, non-linear relationships.

- **Decision Tree Classifier**: Despite the simplicity of decision trees, they are powerful for classification tasks and provide clear visualization of the decision-making process. A key advantage of using a Decision Tree Classifier is its interpretability; it allows for easy understanding of how decisions are made, which is invaluable for analyzing which features most significantly affect a movie's success. Furthermore, decision trees can handle non-linear data effectively and are less susceptible to outliers than regression models, making them a suitable choice for further exploration.

These models were chosen with the intention of addressing specific challenges observed in our initial approach. Polynomial regression will allow us to test the hypothesis that a more nuanced modeling of relationships between variables could yield better predictive performance. On the other hand, the Decision Tree Classifier offers a different approach to classification, with the potential for higher accuracy.

### Model 2: Neural Network Model 

### Fitting Graph Analysis
Based on our MSE values, as the polynomial degree increases from 2 to 4, both training and testing MSEs decrease. This indicates that our model is capturing more of the data's underlying pattern with increased complexity, improving its performance. Our model with polynomial degree 4 have the lowest MSEs suggests an optimal balance between bias and variance, making it the best fit among the ones that we tested. There's no sign of overfitting as both training and testing error decreases together. In conclusion, our polynomial regression model with degree 4 provides the best balance of complexity and performance based on our data.

#### Training and Testing Performance
The initial performance of our neural network on the test set yielded a high accuracy of approximately 91.2%, with a mean squared error (MSE) of 0.0649. This indicates that the model is highly accurate in classifying the movies into the correct rating categories based on their budget and gross revenue. However, the confusion matrix reveals that the model primarily predicts the majority class, as shown by the significant number of true positives (487) for one class and zero true positives for the other two classes. This skew towards a single class suggests a high bias towards the most frequent rating category in the dataset.

The precision of 0.8317 reflects the model's ability to return relevant instances, while the recall of 0.9120 demonstrates the model's proficiency in identifying all relevant instances within the majority class. The relatively high precision and recall further corroborate the model's tendency to favor the predominant class in its predictions.

#### Cross-validation Results
Cross-validation was employed to validate the model's stability and generalizability across different subsets of the data. The 10-fold cross-validation approach produced an overall average accuracy of approximately 89.7% and an average MSE of 0.0516. These results are consistent with the model's performance in the initial test, indicating a robust model across different data splits. However, the variation in accuracy and MSE across folds (with accuracy ranging from 86.9% to 91.4% and MSE from 0.0432 to 0.0655) suggests some variability in model performance depending on the data split, hinting at potential overfitting to certain subsets of the data.

#### Hyperparameter Tuning
Hyperparameter tuning identified the optimal configuration for the first hidden layer to be 312 units, with a learning rate of approximately 0.0002. This optimized setup is expected to enhance the model's ability to learn from the data without overfitting, balancing the complexity of the model with its predictive power. The tuning process is crucial for refining the model's architecture and improving its accuracy and efficiency in handling unseen data. 

#### Conclusion of Model
The evaluation of the neural network model through training vs. test error analysis, cross-validation, and hyperparameter tuning provides a nuanced understanding of its performance. While the model demonstrates high accuracy, the reliance on the majority class for predictions raises questions about its ability to generalize across more evenly distributed classes. The cross-validation results affirm the model's robustness, though they also highlight the necessity of further adjustments to reduce variance and improve model generalizability.


### Evaluate Your Model: Training vs. Test Error
**Model Accuracy graph**: The accuracy for both training and validation shoots up sharply and reaches a high level after the first epoch. After the initial jump, the accuracy for both remains approximately constant. It's noteworthy that the validation accuracy is slightly higher than the training accuracy, which is unusual but not necessarily a problem if the variance is small. This could sometimes happen due to the specific samples in the validation set or regularization effects.

**Model Loss graph**: The loss for both training and validation decreases sharply after the first epoch and continues to decrease gradually over subsequent epochs. The training and validation losses are very close to each other and converge towards a similar value, which indicates that the model is generalizing well without overfitting or underfitting. There's no sign of divergence, which is good.

**Confusion Matrix**: The confusion matrix indicates that our model did not correctly predict any of the samples for the first two classes, but predicted all samples that belonged to the third class. The model correctly identified 487 samples of the third class, which indicates that it has a strong bias towards this class.

**MSE**: The MSE is around 0.074, which might seem low, but MSE is not typically the best metric for classification problems, especially when dealing with categorical data.

**Accuracy**: The accuracy which is about 91.2% which is a bit misleading in this context, as it seems the model has learned to predict the majority class very well but fails to recognize the other classes.

**Precision**: The precision score is about 83.2%. This suggests that when the model predicts a class, it's correct about 83.2% of the time. However, this metric is weighted and might be influenced by the class imbalance.

**Recall**: The recall is equal to the accuracy. This means that the model is very good at detecting the positives of the third class. However, for the first two classes, the recall is 0% because the model failed to identify any true positives for these classes.

In conclusion, the model is likely suffering from a class imbalance issue, where it predicts the majority class well but fails to predict the minority classes. This is evidenced by the lack of true positives for the two classes in the confusion matrix.

### Fitting Graph Analysis, How does it compare to your first model

The model has a high accuracy, precision and recall on both the training and validation data, which would suggest a good fit. This suggests that our second model performs well in predicting the "group ratings" based on the "Gross Revenue" and "Budget" of the movie and we can determine that "Gross Revenue" and "Budget" are good features in order to predict our target variable "group rating". 

The fitting graphs with MSE values for the polynomial regression indicate that the model's performance varies with the degree of the polynomial. As the degree increases, the model fits the training data better but could potentially overfit. While our second model shows high classification accuracy, the fitting graphs from our first model indicate the regression model's capability to predict a continuous outcome based on its input. They serve different purposes and thus, their performances are not directly comparable.

In conclusion, for the specific task of classifying movies into 'group ratings', your second model seems to have found a good balance and performs well according to the classification metrics. For the task of predicting 'Gross Revenue' from 'Budget', the regression analysis of our first model indicates a variable fit depending on the polynomial degree, and its performance should be assessed by how well it generalizes to new, unseen data.

### Did you perform hyper parameter tuning? K-fold Cross validation? Feature expansion? What were the results? 

We performed, Hyper parameter tuning and K-fold Corss validation, but we didn't did any feature expansion as what we did was grouping scores. What we did with Grouping scores mean that We transformed the continuous 'Score' variable into a categorical 'group_rating' variable. This process is a form of feature engineering, where we're creating a new feature based on existing data.

1. **K Fold Cross-Validation**
- Cross-Validation: Applied Repeated K-Fold cross-validation to assess the model's performance across different subsets of the data, ensuring the model's effectiveness and generalization capability. This approach helps in evaluating the model's stability and reliability across different data splits.
- Performance Metrics: The model's accuracy and mean squared error (MSE) were evaluated, providing insights into its classification performance and how close the predicted ratings are to the actual ratings, respectively.
- **Results**: The K-fold cross-validation results indicate that the neural network model exhibits consistent performance across different subsets of the data, with accuracies ranging from approximately 86.9% to 91.4% across ten folds and an overall average accuracy of 89.7%. The mean squared error (MSE) values, averaging at 0.0516, suggest the model's predictions are reasonably close to the actual values. This demonstrates the model's robustness and generalizability, confirming its ability to perform well across diverse data segments without significant overfitting or underfitting to the training data.

2. **Hyperparameter Tuning**
- Hyperparameter Search: Employed a hyperparameter tuning process (using Keras Tuner) to find the optimal model architecture and learning rate, which are crucial for achieving the best possible model performance.
- GridSearch: A GridSearch approach was selected, systematically exploring a range of predefined hyperparameter values to find the best combination, focusing on maximizing validation accuracy.
- Optimization Results: The process identified the optimal number of units in each layer and the learning rate for the SGD optimizer, which were then used to build and evaluate the best model configuration.
- **Results**: The hyperparameter tuning results reveal the identification of an optimal model configuration with 312 units in the first hidden layer and a learning rate of approximately 0.000202. This configuration resulted in a significant improvement in the model's performance during the initial training phases, showcasing high validation accuracies that indicate the model's capacity to make accurate predictions. The process of hyperparameter tuning has effectively pinpointed the most conducive parameters for maximizing the model’s accuracy, underlining the critical role of tuning in enhancing the predictive power and efficiency of machine learning models.

### Next Model Considerations

In light of the insights derived from the performance of our current neural network model, we are considering the exploration of two additional machine learning models to potentially enhance our predictive capabilities and address the identified limitations: 

- **Decision Tree Classifier**: Despite the simplicity of decision trees, they are powerful for classification tasks and provide clear visualization of the decision-making process. A key advantage of using a Decision Tree Classifier is its interpretability; it allows for easy understanding of how decisions are made, which is invaluable for analyzing which features most significantly affect a movie's success. Furthermore, decision trees can handle non-linear data effectively and are less susceptible to outliers than regression models, making them a suitable choice for further exploration.

The Decision Tree Classifier offers a different approach to classification, with the potential for high interpretability and the ability to capture non-linear patterns without the need for transformation.

### Model 3: Decision Tree Classifier

### Training and Testing Performance
**Model Accuracy Graph**: Similar to the previous models, we want to see how the accuracy and training test react as we overfit by changing the hyperparameters of the decision tree. According to our graphs, our training accuracy increases as we increase the maximum depth of the decision tree while the testing accuracy decrease. These the training and testing data intersect when the maximum depth is around 6. As we increase the number of min splits, the training accuracy decreases while the testing accuracy initially decreases, but then increases. The highest accuracy is achieved when number of min split is 2. Lastly, as the number of min sample leaf increases, the training accuracy decreases while the testing accuracy increases. The testing and training data intersect at a value of 7.5 This might indicate that the decision tree performs well at a higher value of maximum depth and min sample leaf, and a lower value of min split.

**Model Loss Graph**: The error/loss for our graph is simply loss = 1 - accuracy. As we increase the max depth of the decision tree, the testing loss increases while our training set decreases which intersects at around a maximum depth of 6.

**Accuracy**: Our decision tree yielded an accuracy of 100% with our training set which indicates overfitting of the data. This is evident in the result of our testing set which yielded an accuracy of about 83%.

**Precision**: Our decision tree yielded an precision of 100% with our training set which indicates overfitting of the data. This is evident in the result of our testing set which yielded an weighted precision of about 84%.

**Recall**: Our decision tree yielded an recall of 100% with our training set which indicates overfitting of the data. This is evident in the result of our testing set which yielded an weighted recall of about 83%.

### Did you perform hyper parameter tuning? K-fold Cross validation? Feature expansion? What were the results?

**1. Hyperparameter Tuning**
- Hyperparameter tuning was conducted using GridSearchCV, focusing on the criterion, max depth, min samples split, and min samples leaf
- The tuning was conducted with 5 cross validations and scored using accuracy
- Based on our tuning, using Gini with a max depth of 1, min sample leaf of 1, and min sample split of 2 yielded the best accuracy
- CV=5 mitigates the risk of overfitting and ensures the performance metrics not to be overly optimistic
- Our testing accuracy increased from about 83% to 90%, indicating a 7% improvement from our base model

## Project Structure

The project is organized into the following sections:

1. **Data Collection and Preprocessing**: Details about how the datasets were obtained and prepared for analysis. In the preprocessing step, we clean the data by handling missing values, normalize numerical values to ensure uniform scales, and encode categorical variables as necessary. For the Kaggle Movies Dataset, we also adjust budget and revenue figures for inflation to enable accurate comparisons over time. Our Jupyter notebook, [Data Preprocessing and Exploration](https://github.com/arandersen/CSE_151_Project/blob/main/Project%20(2).ipynb), provides a detailed walkthrough of these steps.

2. **Exploratory Data Analysis (EDA)**: Comprehensive exploration and visualization of the datasets to gain initial insights.

3. **Feature Engineering**: The process of selecting and transforming relevant features for our analysis.

4. **Machine Learning Model**: A detailed explanation of the Neural Network Model and how it is utilized in our research. [Machine Learning Model: Polynomial Regression, Neural Network, Decision Tree Classifier](https://github.com/arandersen/CSE_151_Project/blob/main/training_model.ipynb), provides a detailed walkthrough of these steps.

5. **Results and Interpretation**: Presentation of our findings, analysis of results, and their implications.

### Data Exploration

The dataset consists of 5,336 movies with data spanning from the year 1980 to 2020. The average movie score is approximately 6.4, with a standard deviation of 0.96, indicating a moderate level of variance in movie ratings. The financial aspect shows a substantial standard deviation in both budget and gross revenue, suggesting a wide disparity in movie budgets and earnings. Votes for movies have a wide range, indicating significant differences in popularity or viewership, with an average of around 114,229 votes. The distribution of 'Votes' is heavily right-skewed, meaning most movies receive relatively few votes, while a small number receive a vast majority, likely reflecting blockbuster hits. The distribution of 'Gross Revenue' is also right-skewed, with most movies earning relatively less and a few earning substantially more, which is typical of box office earnings. 

The Q-Q plots for both gross revenue and budget before and after transformation show significant deviation from the line of normality at the tails, suggesting that the distribution of these financial variables is heavily skewed. This skewness indicates that while a few movies earn extremely high revenues and have large budgets, the majority do not. The histogram for the distribution of the budget shows a right-skewed distribution, reinforcing the Q-Q plot findings that most movies have a relatively low budget, with the frequency sharply decreasing as the budget increases. Bar charts displaying the mean score by rating and by genre provide insightful patterns about the average performance of movies within each category. Movies with different ratings and genres have distinct average scores, indicating that certain ratings and genres might be associated with higher or lower average movie scores.

The correlation matrix reveals relationships between numerical variables such as the year of release, score, votes, budget, gross revenue, and runtime. Notable observations include a moderate positive correlation between gross revenue and votes, implying that more popular movies tend to earn higher revenues. There's also a weaker, yet significant, positive correlation between the budget and gross revenue, suggesting that higher budgets are somewhat associated with higher gross revenues.

### Data Preprocessing

The data preprocessing was used to refine the input for the neural network. We made a function called classify_score to segment movie scores into categorical bands, 'very bad' for scores up to 3, 'bad' for those up to 5, 'mid' for scores up to 8, and 'good' for any higher. This categorization was actualized by generating a new column, 'group_rating', applying the classification function to the existing 'score' data. The analysis then proceeded to isolate 'Budget' and 'Gross Revenue' as the feature set, positing these financial metrics as predictors, while 'group_rating' was established as the target variable, reflecting the categorized movie ratings.

To accommodate the categorical nature of the target variable within a neural network's computational framework, one-hot encoding was employed, transforming 'group_rating' into a binary matrix essential for model interpretation. Concurrently, feature scaling was conducted through min-max normalization on the 'Budget' and 'Gross Revenue' variables, thereby enhancing the algorithm's convergence efficiency. Through these steps, the dataset was optimally preprocessed to support the subsequent neural network's learning and predictive accuracy.

### Model 1: Polynomial Regression

Our polynomial regression analysis involved assessing the relationship between movie budgets and gross revenue across polynomial degrees 2, 3, and 4. The results are encapsulated in the following figures:

1. **Polynomial Degree 2**: The scatter plot with a second-degree polynomial fit reveals the basic quadratic relationship between the budget and gross revenue.
2. **Polynomial Degree 3**: Increasing the polynomial degree to 3 showed a more flexible fit to the data, indicating a more complex relationship.
3. **Polynomial Degree 4**: The fourth-degree polynomial regression provided the most nuanced fit among the tested models, suggesting a more detailed underlying pattern.

#### Training vs. Test Error Evaluation

- **Degree 2**: The evaluation of the second-degree polynomial model showed a mean squared error (MSE) of 0.51227 for training data and 0.48057 for testing data.
- **Degree 3**: For the third-degree polynomial model, the training MSE was 0.49835 and the testing MSE was 0.46815, suggesting a better fit than the second degree.
- **Degree 4**: The most complex model, with a fourth-degree polynomial, achieved a training MSE of 0.49328 and a testing MSE of 0.46525.

#### Model Fit in the Fitting Graph

The fitting graph analysis clearly showed that as the polynomial degree increases from 2 to 4, both training and testing MSEs decreased. This trend implies that our models are effectively capturing more of the complexity inherent in the data with each incremental degree. The fourth-degree polynomial regression, having the lowest MSEs, was deemed the best fit among the tested models. Importantly, there was no evidence of overfitting, as indicated by the parallel decrease of both training and testing errors. The fourth-degree polynomial model emerged as the optimal balance between complexity and performance based on the dataset.

These findings suggest that higher-degree polynomial regression can effectively model the non-linear relationship between movie budgets and gross revenue. However, the careful selection of polynomial degree is crucial to avoid overfitting while still capturing the underlying data pattern with sufficient complexity. The results from our polynomial regression models inform our understanding of the financial dynamics in the movie industry, providing valuable insights into how budgetary decisions can influence a movie's commercial success.

### Model 2: Neural Network

#### Data Preprocessing
Our neural network model's preprocessing involved creating a new target variable 'group_rating' derived from the 'Score' feature, which classifies movies into 'very bad,' 'bad,' 'mid,' and 'good' categories. We selected 'Budget' and 'Gross Revenue' as feature inputs, applied one-hot encoding to the target variable, and performed MinMax scaling on the inputs to normalize the data.

#### Training and Testing Performance
The neural network was trained and tested, revealing the following results:

- Confusion Matrix indicated a strong bias towards one class, with the model correctly predicting all samples of the third class but none for the first two.
- Accuracy was high at 91.2%, which should be interpreted cautiously due to the apparent class imbalance.
- Mean Squared Error (MSE) stood at 0.0931, less indicative due to the classification nature of the problem.
- Precision was recorded at 83.2%, suggesting the model was mostly correct when it made predictions, albeit likely skewed by class imbalance.
- Recall matched the accuracy, again reflecting the bias towards the third class.

#### K-Fold Cross-Validation
K-Fold cross-validation demonstrated the model's consistent performance, with an overall average accuracy of approximately 89.7% and an average MSE of 0.0516, indicating robustness across different data segments. The variance in accuracy and MSE suggested some susceptibility to overfitting on particular subsets of the data.

#### Hyperparameter Tuning
Hyperparameter tuning identified an optimal configuration for the neural network model, indicating that 312 units for the first hidden layer and a learning rate of about 0.0002 are best suited for this analysis.

#### Model Accuracy and Loss Over Epochs
Training and validation accuracy graphs showed an initial surge followed by a plateau, with the validation accuracy peculiarly higher than training accuracy. Loss graphs for both sets decreased consistently over epochs, indicative of good generalization and no evidence of overfitting or underfitting.

#### Analysis
The model demonstrated high classification accuracy for the 'group ratings' based on 'Gross Revenue' and 'Budget'. However, the analysis indicates a potential class imbalance, as shown by the model's inability to predict the minority classes accurately.

In conclusion, while the neural network model exhibits strong predictive accuracy, it is prone to bias toward the majority class, a challenge that future iterations of the model must address to improve class-specific predictive performance.

### Model 3: Decision Tree Classifier

#### Data Preprocessing
Our decision tree preprocessing involved creating a new target variable 'group_rating' derived from the 'Score' feature, which classifies movies into 'very bad,' 'bad,' 'mid,' and 'good' categories, and applied one-hot encoding to the target variable. We selected 'Budget' and 'Gross Revenue' as feature inputs and performed MinMax scaling on the feature inputs to normalize the data.

#### Training and Testing Performance
The decision tree was trained and tested, revealing the following results:

- Confusion Matrix indicated a tendency of the model to misclassify "mid" movies as "bad" and vice versa.
- Accuracy was high at about 90%, which should be interpreted cautiously due to the apparent class imbalance.
- Precision was recorded at about 84%, suggesting the model was mostly correct when it made predictions, albeit likely skewed by class imbalance.
- Recall was recorded at about 83%, again reflecting the bias towards the "mid" and "bad" movies.

#### Hyperparameter Tuning
Hyperparameter tuning identified that using Gini with a max depth of 1, min sample leaf of 1, and min sample split of 2 yielded the best accuracy, increasing our testing accuracy from about 83% to 90%

#### Analysis
The model demonstrated high classification accuracy for the 'group ratings' based on 'Gross Revenue' and 'Budget'. However, the analysis indicates a potential class imbalance, as shown by the model tend to misclassify "mid" movies as "bad" and vice versa, which is a further issue to address to improve the predictive performance of the model.

6. ## Conclusion and Future Directions of our model

### Conclusion of the 1st Model

The improvement in MSE as the degree of the polynomial increases suggests that the relationship between budget and gross revenue is complex and potentially non-linear, with higher-degree polynomials capturing this complexity more effectively. The consistent decrease in both training and testing errors indicates that the model is not yet suffering from overfitting at the fourth degree.

However, it's important to note that while the improvements in MSE are consistent, they are also marginal, especially when moving from degree 3 to degree 4. This diminishing return suggests that there is a limit to how much more complexity (in terms of polynomial degree) can beneficially be added to the model without overfitting. The observation of diminishing returns as we increase the polynomial degree suggests a critical insight into the nature of modeling complex relationships, such as that between a movie's budget and its gross revenue. It emphasizes the inherent trade-offs in model development, especially between capturing the underlying data patterns (reducing bias) and maintaining a model's ability to generalize well to unseen data (avoiding overfitting).

The fact that the model is not yet overfitting at the fourth degree is encouraging, indicating there's still some, albeit limited, scope for complexity increase without sacrificing model performance on new data. However, the marginal gains observed caution us against pursuing higher degrees of polynomial without careful consideration. It suggests that we are approaching, if not already at, the point of optimal complexity where the model is sufficiently complex to capture the relevant patterns in the data but not so complex that it becomes overly specialized to the training set.

This situation underscores the importance of exploring alternative strategies for model improvement that do not solely rely on increasing model complexity through higher-degree polynomials. Techniques such as incorporating domain knowledge to engineer more relevant features, employing regularization methods to penalize unnecessary complexity, and exploring other forms of model validation like cross-validation to ensure that improvements are robust and generalizable, become paramount.

Moreover, this context also highlights the potential utility of exploring other modeling approaches that might inherently balance complexity and generalizability better. Machine learning models, such as random forests, gradient boosting machines, or neural networks, offer sophisticated mechanisms to model non-linear relationships and interactions without manually specifying the form of the model. These models come with their mechanisms to control overfitting, such as depth limitations in trees or dropout in neural networks, potentially providing a more effective way to capture the complexities of the relationship between movie budgets and gross revenue while maintaining good performance on unseen data.

### Conclusion of the 2nd Model

The second model demonstrates good performance in terms of accuracy, precision, and recall on both the training and validation datasets. These metrics are indicative of a model that is correctly identifying the majority of instances across the classes it was trained to predict. The consistent high performance on unseen validation data suggests that the model has generalized well beyond the training dataset.

However, it is important to consider the confusion matrix, which reveals that while the model excels at predicting a certain class, it may not be performing equally well across all classes. This could be a sign of class imbalance or that the model's predictive features such as 'Gross Revenue' and 'Budget' are particularly informative for one class but less so for others.

In conclusion, the second model is a robust classifier for the 'group ratings' based on the 'Gross Revenue' and 'Budget' features. It presents a high degree of accuracy, precision, and recall, which are strong indicators of its reliability.

### Conclusion of the 3rd Model

The third model demonstrates strong performance across various evaluation metrics, including accuracy, precision, and recall, on both the training and testing datasets. These metrics suggest that the model effectively identifies the majority of instances across its predicted classes, indicating its proficiency in capturing underlying patterns within the data. Moreover, the model exhibits consistent high performance on the testing data, implying that it has successfully generalized beyond the constraints of the training dataset.

However, a detailed examination of the confusion matrix reveals certain tendencies of the model to misclassify "mid" movies as "bad" and vice versa. Despite these occasional misclassifications, the model's overall performance remains robust. These insights underscore the importance of considering nuanced aspects of model performance beyond aggregate metrics, allowing for a more comprehensive evaluation of its predictive capabilities.

In conclusion, the third model emerges as a reliable classifier for predicting 'group ratings' based on the 'Gross Revenue' and 'Budget' features. Its high degree of accuracy, precision, and recall are strong indicators of its reliability and effectiveness, making it a good model.

### Improvements

To refine our approach and enhance the model's predictive accuracy and reliability, we propose several strategies:

**Model 1 Improvement Suggestion**

- **Regularization**: To prevent overfitting, especially when using higher-degree polynomials, consider applying regularization techniques such as Ridge or Lasso regression. These methods can help control the complexity of the model by penalizing large coefficients.

- **Feature Engineering**: Besides polynomial features, explore other forms of feature engineering. For instance, interaction terms between budget and other variables might provide additional insights. Also, normalizing or scaling the features might help, especially when moving towards models that use regularization.

- **Alternative Models**: Consider exploring non-linear models beyond polynomials, such as decision trees, random forests, or gradient boosting machines, which might capture the data's complexity in different ways.

- **Hyperparameter Tuning**: Use grid search or random search to find the optimal combination of hyperparameters, such as the degree of the polynomial and regularization strength. This systematic approach can help in identifying the best model configuration.

- **Non-Linear Transformations**: Before applying polynomial features, consider non-linear transformations on the input features, such as logarithmic, square root, or exponential transformations. These transformations can help in linearizing relationships between features and the target variable.

**Model 2 Improvement Suggestion**

- **Data Augmentation**: Expanding our dataset with more variables or by integrating additional datasets could provide a richer context for analysis, helping the model to uncover more nuanced relationships between features and outcomes.

- **Advanced Feature Engineering**: Delving deeper into feature selection and transformation to emphasize more predictive variables. This could involve more sophisticated techniques to extract or combine features in ways that better capture the complexities of movie success.

- **Model Complexity Adjustment**: Experimenting with the neural network's architecture, such as layer depth and neuron count, to strike an optimal balance between model complexity and overfitting. This includes evaluating different activation functions, optimizers, and regularization methods.

- **Ensemble Methods**: Considering ensemble techniques, such as bagging or boosting, to improve model stability and accuracy. These methods can aggregate predictions from multiple models to reduce variance and bias.

- **Exploration of Alternative Models**: As previously mentioned, we plan to explore Polynomial Regression and Decision Tree Classifier models. These alternatives could offer new perspectives and methodologies for addressing the dataset's challenges, potentially overcoming limitations observed in the neural network model.

**Model 3 Improvement Suggestion**

- **Increase Cross-Validation Folds:** Increasing the number of folds in cross-validation can lead to a more reliable estimate of model performance. By using more folds, each instance in the dataset gets to be in the test set exactly once and in the training set K−1 times (where K is the number of folds), providing a more comprehensive evaluation of the model's performance across different subsets of the data.

- **Expand the Parameter Grid:** Extending the range or adding new parameters to the param_grid can potentially lead to finding a better model. 

- **Implement Randomized Search:** Instead of using GridSearchCV, we can use RandomizedSearchCV which samples a given number of candidates from the parameter space with a specified distribution. This approach can be more efficient than Grid Search and provide a good approximation of the best parameters with significantly less computational time.

- **Feature Engineering and Selection:** We could create new features from existing ones through domain knowledge and select the most relevant features in the data. By doing so, we can implement feature importance scores for feature selection which can lead to a more effective and efficient model.

## Discussion

Our exploration into the intricate world of cinema through the lens of data science has led us to some interesting findings. In this section, we delve deep into the interpretation, significance, and limitations of our results and approach framing them within the broader context of film analytics. This discussion is rooted in our initial objective: to unravel the complex relationship between movie budgets, box office performance, critical reception, and audience preferences.

#### Data Preprocessing

The creation of the `classify_score` method and the categorization of movies into 'very bad', 'bad', 'mid', and 'good' based on their scores was a critical step in our analysis. This preprocessing allowed us to transition from continuous to categorical analysis, facilitating our exploration into classification models. However, this simplification also introduces a potential limitation: the reduction of the rich variability found in movie ratings into broad categories might mask subtler trends and insights within the data. Furthermore, the reliance on 'Budget' and 'Gross Revenue' as the primary predictors overlooks other potentially influential factors, such as genre, directorial influence, and critical acclaim, which could provide a more nuanced understanding of movie success. But it is also notable to think about the fact that 'Budget' and 'Gross Revenue' had the highest correlation to group ratings and that the other features did not have any impactful correlation to group rating.

#### Model 1: Polynomial Regression

The polynomial regression analysis highlighted a non-linear relationship between movie budgets and gross revenue, with higher-degree polynomials capturing this complexity more effectively. This finding underscores the multifaceted nature of movie success, suggesting that simply increasing a movie's budget does not guarantee proportionate returns in revenue. However, the diminishing returns observed as we increased the polynomial degree to 4 raise questions about the practical limits of this model. This leads us to ponder the balance between model complexity and interpretability, and whether other variables not included in our model could better explain the variance in movie revenue.

#### Model 2: Neural Network

The neural network's high classification accuracy in predicting 'group ratings' from 'Budget' and 'Gross Revenue' is compelling, showcasing the model's ability to discern patterns within the data. Yet, the strong bias towards one class and the apparent class imbalance it revealed point to a significant challenge in modeling: ensuring that our model is accurate across all categories. This suggests the need for more sophisticated techniques to handle class imbalance, such as SMOTE or cost-sensitive learning. Moreover, the peculiar phenomenon of higher validation accuracy than training accuracy prompts us to question the distribution of our data splits and whether this could artificially inflate our model's perceived performance.

#### Model 3: Decision Tree Classifier

Our decision tree classifier, while providing high accuracy, precision, and recall, also exhibited a tendency to misclassify movies between 'mid' and 'bad' categories. And desipte having a slightly lower score for accuracy compared to the neural network model, it is notable that this model does not have bias towards one class. Furthermore, the decision tree model provided a more generalized performance across multiple classes, as opposed to the neural network's specialization in accurately predicting a single class. In scenarios where it's important to maintain balanced performance across various categories, the decision tree's ability to offer a more even classification might be preferred. Continuing on, the misclassification our decision tree model raises important considerations about the distinctiveness and overlap of these categories, as well as the decision tree's sensitivity to the nuances within the data. The improvement in testing accuracy following hyperparameter tuning is a proof to the effectiveness of these techniques. Yet, it also highlights the necessity of a careful and informed search for optimal model parameters to prevent overfitting and ensure generalizability. 

#### General Discussion

Across all models, the challenge of class imbalance emerged as a recurring theme, indicating a broader issue in machine learning applications to diverse datasets. This underscores the importance of employing balanced datasets or adjusting models to account for this imbalance. Furthermore, our results' believability hinges on the rigorousness of our preprocessing, model selection, and validation techniques. While we've made efforts to ensure accuracy and reliability, the inherent limitations of our dataset and chosen features invite caution in over-generalizing our findings.

#### Limitations and Future Work

Our analysis is not without its limitations. The decision to focus primarily on 'Budget' and 'Gross Revenue' as predictors may have oversimplified the complex factors influencing movie success. Future studies could enrich this analysis by incorporating additional variables, such as social media sentiment, critic reviews, or actor popularity. Additionally, exploring ensemble models or more advanced machine learning techniques could provide further insights and potentially more accurate predictions.

## Conclusion

Reflecting on our journey through the landscape of cinema analytics, it's clear that our exploration has opened more doors than it has closed. The fusion of data science and movie analytics not only highlights the potential for uncovering hidden patterns in the film industry but also underscores the complexities and challenges that come with it.

Our decision to focus on 'Budget' and 'Gross Revenue' as primary predictors was driven by their apparent influence on a movie's commercial success. However, the journey revealed the nature relationship between these financial metrics and a movie's critical reception and audience preferences. The polynomial regression model provided valuable insights into the non-linear nature of this relationship, suggesting that movie success is not merely a matter of financial investment. Meanwhile, the neural network and decision tree models delved into classification, offering a lens through which to view the categorization of movies based on their success metrics.

One reflection from our work is the critical role of data preprocessing and the choice of features in modeling. The simplification into categorical bands, while necessary for our analysis, may have masked subtler distinctions within the data. Future explorations could benefit from a more granular approach, possibly by incorporating a broader array of features that capture the multifaceted nature of movie success more comprehensively.

Another key takeaway is the challenge posed by class imbalance, especially evident in the neural network model's results. This issue underscores the need for sophisticated methods to ensure balanced model performance across categories, highlighting an area for further methodological innovation.

#### What Could Have Been Done Differently?

Expanding our feature set to include qualitative aspects such as genre, directorial influence, or social media sentiment could have enriched our analysis, offering deeper insights into what drives a movie's success. Additionally, employing more advanced techniques to address class imbalance or exploring ensemble methods could have potentially enhanced the robustness and accuracy of our predictions. A different approach we could have explored further involves scrapping additional data from various sources, cleaning it, and matching it with our existing dataset. This approach would have allowed us to proceed with data that have matching records, potentially enriching our analysis with a wider array of variables and insights.

#### Future Directions

The path forward is rich with opportunities for deeper exploration. Integrating natural language processing to analyze movie reviews or leveraging social network analysis to understand the impact of celebrity networks on movie success are just a few avenues that could yield fascinating insights. Moreover, the application of cutting-edge machine learning techniques, such as deep learning or reinforcement learning, could offer new perspectives on predicting movie success.

#### Final Thoughts

Our project stands as a proof to the power and potential of applying data science to the arts and entertainment sector. While we've uncovered some patterns and relationships within the film industry, the complexity of the subject matter ensures that there is much more to explore. The intersection of cinema and data science remains a fertile ground for research, promising to enhance our understanding of cultural products and the factors that drive their success. As we close this chapter, we look forward to the next phase of discovery, confident that the future holds even more exciting insights at the nexus of film economics, critical success, and data analytics.

## Contributors

- Arthur Andersen
- Carlson Salim
- Kenneth Hidayat
- Ryan Paquia
- Bryant Tan
- Steven Sahar
- Noah Jaurigue

## License

This project is licensed under the [License Name] License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

We would like to express our gratitude to the creators of the datasets used in this research, as well as to our instructors and mentors who guided us throughout the project.

