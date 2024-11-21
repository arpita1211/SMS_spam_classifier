# SMS_spam_classifier

Dataset
The project uses a dataset containing SMS messages labeled as Spam or Ham. It includes:

Columns:
target: Indicates whether the message is spam (1) or ham (0).
text: The content of the SMS message.
The dataset was cleaned by removing unnecessary columns, renaming fields for clarity, and dropping duplicate entries.

Model
This project utilizes Natural Language Processing (NLP) techniques and machine learning models to classify SMS messages. The primary steps include:

Text Preprocessing:

Lowercasing text.
Removing special characters, punctuation, and stopwords.
Applying stemming for word normalization.
Feature Extraction:

Used TF-IDF Vectorization to convert text into numerical features.
Classification Algorithms:

Multiple models were tested, including:
Naive Bayes (MultinomialNB)
Support Vector Machines (SVC)
Random Forest
The final model used was Multinomial Naive Bayes, stored as a pickle file (model.pkl).
Key Libraries
Data Processing:
pandas, numpy
NLP:
nltk for tokenization and stopword removal.
WordCloud for data visualization.
Machine Learning:
scikit-learn for TF-IDF vectorization, model building, and evaluation.
Web Application:
streamlit for creating an interactive interface.
Workflow
Data Cleaning:

Removed unnecessary columns.
Dropped duplicate rows.
Exploratory Data Analysis (EDA):

Visualized the distribution of spam and ham messages.
Analyzed text features (e.g., number of characters, words, and sentences).
Text Preprocessing:

Normalized text and tokenized it.
Removed stopwords and punctuation.
Applied stemming for word simplification.
Feature Extraction:

Transformed text using TF-IDF Vectorization for numerical representation.
Model Selection:

Tested several classifiers for accuracy and precision.
Finalized the Multinomial Naive Bayes model.
Deployment:

Created a web-based application using Streamlit.

Installation
Clone the repository:
git clone https://github.com/your-repo/sms-spam-detection.git
cd sms-spam-detection
Install dependencies:
pip install -r requirements.txt
Download the NLTK data required:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
Ensure the model.pkl and vectorizer.pkl files are present in the working directory.

Running the Streamlit App
Launch the application:
streamlit run app.py

Enter the SMS message in the text area and click Predict to classify it as Spam or Not Spam.
