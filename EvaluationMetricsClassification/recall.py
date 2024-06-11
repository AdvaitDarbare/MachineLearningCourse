actual = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]
predicted = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

for i in range(len(predicted)):
  if actual[i] == 1 and predicted[i] == 1:
    true_positives += 1
  if actual[i] == 0 and predicted[i] == 0:
    true_negatives += 1
  if actual[i] == 0 and predicted[i] == 1:
    false_positives += 1
  if actual[i] == 1 and predicted[i] == 0:
    false_negatives += 1

recall = true_positives/(true_positives + false_negatives)

print(recall)


'''
Recall is a vital metric when the cost of false negatives is high and ensuring the detection of all positive instances is critical. It provides a different perspective from other metrics like accuracy and precision, helping to create a more balanced and effective classification model.

Recall = (True Positives) / (True Positives + False Negatives)

'''