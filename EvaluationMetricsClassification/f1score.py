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

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

f_1 = 2*precision*recall/(precision+recall)

print(f_1)


'''
It is often useful to consider both the precision and recall when attempting to describe the effectiveness of a model. This is F1 score. 
'''
