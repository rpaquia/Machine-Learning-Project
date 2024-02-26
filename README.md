# CSE 151 Project - Analyzing Movie Datasets

## Introduction

In this research project, we aim to conduct a comprehensive analysis of datasets pertaining to the film industry. Our investigation encompasses three primary datasets, each providing valuable insights into different aspects of the movie world:

1. **Kaggle Movies Dataset by Daniel Grijalvas**: This dataset contains extensive information about movies, including data on budget, revenue, and ratings.

2. **Movie Dataset by UCI Machine Learning Repository**: This dataset offers a deeper understanding of various movie attributes, including their types and ratings.

3. **Box Office Collections Dataset from Kaggle**: This dataset provides insights into the commercial performance of movies.

Our objective is to employ a rigorous analytical approach to uncover intricate relationships between various factors in the film industry. Specifically, we seek to:

- Examine the correlation between a movie's budget and its box office performance.
- Assess the influence of critical and commercial reception on financial outcomes.
- Identify patterns in audience preferences across different time periods.

## Research Goals

Through this multidimensional analysis, our research group endeavors to provide a holistic understanding of the factors contributing to a movie's success, encompassing both its commercial and critical acclaim. Our findings aim to offer valuable insights to stakeholders involved in the filmmaking process, marketing, and industry analysis.

## Machine Learning Model

For the purpose of this research, we have employed a robust machine learning model, the **Neural Network Model**, to aid in our analysis and predictions. This model enables us to make data-driven decisions and draw meaningful conclusions from the complex datasets under scrutiny.

### Model Result

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

### Fitting Graph Analysis

The model has a high accuracy on both the training and validation data, which would typically suggest a good fit. However, the confusion matrix reveals that the model has learned to predict only the majority class and fails to recognize any instances of the minority classes. This isn't the traditional notion of overfitting where the model memorizes the training data but rather a case where the model is biased and does not have the capability to generalize across all classes.
The situation indicates that although the model's overall error rates (like accuracy) are low, it's actually failing to learn meaningful patterns across all classes due to class imbalance or perhaps a lack of representational capacity to differentiate between classes.

In conclusion, your model's fit is superficially good but fundamentally poor due to its inability to classify more than one class. This is a form of overfitting to the majority class, which is often seen in cases of severe class imbalance.

### Next Model Considerations

In light of the insights derived from the performance of our current neural network model, we are considering the exploration of two additional machine learning models to potentially enhance our predictive capabilities and address the identified limitations:

**Polynomial Regression**: This model is particularly appealing for its ability to capture the non-linear relationships between the features and the target variable, which is a common scenario in complex datasets like ours. Unlike linear regression, polynomial regression can model the intricate patterns observed in the budget, gross revenue, and their impact on a movie's rating category, potentially offering more accurate predictions by fitting a curved line through our data points.

**Decision Tree Classifier**: Despite the simplicity of decision trees, they are powerful for classification tasks and provide clear visualization of the decision-making process. A key advantage of using a Decision Tree Classifier is its interpretability; it allows for easy understanding of how decisions are made, which is invaluable for analyzing which features most significantly affect a movie's success. Furthermore, decision trees can handle non-linear data effectively and are less susceptible to outliers than regression models, making them a suitable choice for further exploration.

These models were chosen with the intention of addressing specific challenges observed in our initial approach. Polynomial regression will allow us to test the hypothesis that a more nuanced modeling of relationships between variables could yield better predictive performance. On the other hand, the Decision Tree Classifier offers a different approach to classification, with the potential for high interpretability and the ability to capture non-linear patterns without the need for transformation.

## Project Structure

The project is organized into the following sections:

1. **Data Collection and Preprocessing**: Details about how the datasets were obtained and prepared for analysis. In the preprocessing step, we clean the data by handling missing values, normalize numerical values to ensure uniform scales, and encode categorical variables as necessary. For the Kaggle Movies Dataset, we also adjust budget and revenue figures for inflation to enable accurate comparisons over time. Our Jupyter notebook, [Data Preprocessing and Exploration](https://github.com/arandersen/CSE_151_Project/blob/main/Project%20(2).ipynb), provides a detailed walkthrough of these steps.

2. **Exploratory Data Analysis (EDA)**: Comprehensive exploration and visualization of the datasets to gain initial insights.

3. **Feature Engineering**: The process of selecting and transforming relevant features for our analysis.

4. **Machine Learning Model**: A detailed explanation of the Neural Network Model and how it is utilized in our research. [Machine Learning Model: Neural Network](https://github.com/arandersen/CSE_151_Project/blob/main/training_model.ipynb), provides a detailed walkthrough of these steps.

5. **Results and Interpretation**: Presentation of our findings, analysis of results, and their implications.

6. ## Conclusion and Future Directions

### Conclusion of the 1st Model

Our model demonstrates high accuracy (approximately 91.2%) on the dataset used. However, the performance metrics and the confusion matrix suggest that this accuracy is primarily due to the model's ability to correctly identify one class while failing to recognize the others. The model has learned to predict the majority class almost exclusively, resulting in a high number of false negatives for the minority classes. This situation is indicative of a model that has not learned the distinguishing features of the minority classes. The training and validation loss graphs, as well as the accuracy graphs, indicate that the model's parameters are being optimized effectively. Nonetheless, the model's real-world utility is limited due to its lack of generalizability across different classes.

### Improvements

To refine our approach and enhance the model's predictive accuracy and reliability, we propose several strategies:

- **Data Augmentation**: Expanding our dataset with more variables or by integrating additional datasets could provide a richer context for analysis, helping the model to uncover more nuanced relationships between features and outcomes.

- **Advanced Feature Engineering**: Delving deeper into feature selection and transformation to emphasize more predictive variables. This could involve more sophisticated techniques to extract or combine features in ways that better capture the complexities of movie success.

- **Model Complexity Adjustment**: Experimenting with the neural network's architecture, such as layer depth and neuron count, to strike an optimal balance between model complexity and overfitting. This includes evaluating different activation functions, optimizers, and regularization methods.

- **Ensemble Methods**: Considering ensemble techniques, such as bagging or boosting, to improve model stability and accuracy. These methods can aggregate predictions from multiple models to reduce variance and bias.

- **Exploration of Alternative Models**: As previously mentioned, we plan to explore Polynomial Regression and Decision Tree Classifier models. These alternatives could offer new perspectives and methodologies for addressing the dataset's challenges, potentially overcoming limitations observed in the neural network model.

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

