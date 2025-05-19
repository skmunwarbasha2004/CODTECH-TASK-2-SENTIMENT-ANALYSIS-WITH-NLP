# CODTECH-TASK-2-SENTIMENT-ANALYSIS-WITH-NLP


COMPANY : CODTECH IT SOLUTION

NAME : SHAIK MUNWAR BASHA

INTERN ID : CT06DM431

DOMAIN : Machine Learning

MENTOR : Neela Santosh

DURATION : 6 weeks

Sentiment Analysis on Customer Reviews Using TF-IDF and Logistic Regression

Sentiment analysis is a natural language processing (NLP) technique used to determine the emotional tone behind a body of text, such as customer reviews. It classifies text as positive, negative, or neutral, helping businesses understand customer opinions, improve products, or monitor brand perception. In this case, we’re analyzing a dataset of customer reviews to classify them as positive (1) or negative (0) using TF-IDF vectorization and Logistic Regression.

Procedure Overview
The code performs sentiment analysis on a dataset (sentiment_train) through several steps: data loading, preprocessing, feature extraction, model training, evaluation, and interpretation. Let’s break down each step.<br/>

1.Loading and Inspecting the Data: The dataset is loaded using pandas from a tab-delimited file (sentiment_train). It contains two columns: text (review text) and sentiment (0 for negative, 1 for positive). The first five rows are displayed to understand the data structure, and sentiment.value_counts() checks the class distribution (e.g., how many positive vs. negative reviews). A count plot visualizes this distribution, revealing if the dataset is balanced. A balanced dataset ensures the model isn’t biased toward one class.<br/>
2.Text Preprocessing with TF-IDF: Raw text data needs to be converted into numerical features for machine learning. We use TF-IDF (Term Frequency-Inverse Document Frequency) vectorization, which weighs words based on their frequency in a document relative to the entire dataset, highlighting important terms while downplaying common ones. Custom stop words (e.g., "movie," "harry") are added to remove irrelevant terms, and a Porter Stemmer reduces words to their root form (e.g., "running" to "run") to improve consistency. The TfidfVectorizer is configured with a maximum of 1000 features to limit dimensionality, and the stemmed_words function ensures stemming and stop-word removal during vectorization. The top 15 features by TF-IDF sum are printed to show the most influential terms.<br/>
3.Splitting the Data: The dataset is split into training (70%) and testing (30%) sets using train_test_split. The features (train_ds_features) are the TF-IDF vectors, and the target (sentiment) is the label. A random_state=42 ensures reproducibility.<br/>
4.Training the Logistic Regression Model: Logistic Regression is chosen as the classifier because it performs well with TF-IDF features and provides interpretable coefficients. It’s trained on the training set (train_X, train_y) using LogisticRegression with random_state=42 for consistency. The model learns to predict sentiment based on the TF-IDF features.<br/>
5.Evaluating the Model: The trained model predicts sentiments for the test set (test_X), and performance is evaluated using a classification report, which includes precision, recall, F1-score, and accuracy for both classes. For example, an accuracy of 0.84 indicates 84% of predictions were correct. A confusion matrix is plotted as a heatmap, showing true positives, true negatives, false positives, and false negatives (e.g., 885 true negatives, 80 true positives).<br/>
6.Feature Importance: Logistic Regression coefficients indicate feature importance. Positive coefficients suggest features associated with positive sentiment (e.g., "great"), while negative coefficients indicate negative sentiment (e.g., "terrible"). The top 10 features for each are printed, providing insights into what drives sentiment predictions.<br/>

The code transforms raw review text into numerical features using TF-IDF, trains a Logistic Regression model to classify sentiments, and evaluates its performance. The process ensures the model learns meaningful patterns (e.g., positive words like "excellent" vs. negative ones like "awful") while handling noise through stop words and stemming. The evaluation and feature importance steps help assess the model’s effectiveness and interpret its decisions, making it a robust approach for sentiment analysis..<br/>
#output.<br/>
![Image](https://github.com/user-attachments/assets/850fb50d-4ce3-4735-afb0-c3d55b0f72b6)

![Image](https://github.com/user-attachments/assets/83145405-4419-4314-83ce-8b8305a51a20)
