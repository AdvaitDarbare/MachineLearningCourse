from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Fetch the entire 20 Newsgroups dataset
emails = fetch_20newsgroups()
print(emails.target_names)  # Print the target categories

# Fetch specific categories: 'rec.autos' and 'rec.sport.hockey'
emails = fetch_20newsgroups(categories=['rec.autos', 'rec.sport.hockey'])
print(emails.target_names)  # Print the target categories
print(emails.data[5])  # Print the content of the 6th document
print(emails.target[5])  # Print the target label of the 6th document

# Split the dataset into training and test sets
train_emails = fetch_20newsgroups(categories=['rec.autos', 'rec.sport.hockey'], subset='train', shuffle=True, random_state=108)
test_emails = fetch_20newsgroups(categories=['rec.autos', 'rec.sport.hockey'], subset='test', shuffle=True, random_state=108)

# Initialize the CountVectorizer
counter = CountVectorizer()

# Fit the vectorizer on the combined training and test data to build the vocabulary
counter.fit(train_emails.data + test_emails.data)

# Transform the training and test data into matrices of token counts
train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)

# Initialize the Multinomial Naive Bayes classifier
classifier = MultinomialNB()

# Train the classifier using the training data and their corresponding labels
classifier.fit(train_counts, train_emails.target)

# Evaluate the classifier on the test data and print the accuracy
print(classifier.score(test_counts, test_emails.target))
