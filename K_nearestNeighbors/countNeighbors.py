# Our goal now is to count the number of good movies and bad movies in the list of neighbors.
# If more of the neighbors were good, then the algorithm will classify the unknown movie as good.
# Otherwise, it will classify it as bad.

from movies import movie_dataset, movie_labels

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def classify(unknown, dataset, labels, k):
  distances = []
  #Looping through all points in the dataset
  for title in dataset:
    movie = dataset[title]
    distance_to_point = distance(movie, unknown)
    #Adding the distance and point associated with that distance
    distances.append([distance_to_point, title])
  distances.sort()
  #Taking only the k closest points
  neighbors = distances[0:k]

  num_good, num_bad = 0, 0
  for movie in neighbors:
    title = movie[1]
    if labels[title] == 0:
      num_bad += 1;
    else:
      num_good += 1 

  res = -1
  if num_good > num_bad:
    res = 1
  else:
    res = 0

  return res

print(classify([.4, .2, .9], movie_dataset, movie_labels, 5))
