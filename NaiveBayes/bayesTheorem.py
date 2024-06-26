from reviews import neg_counter, pos_counter

review = "This crib was amazing"

percent_pos = 0.5
percent_neg = 0.5

# number of words in pos and neg
total_pos = sum(pos_counter.values())
total_neg = sum(neg_counter.values())

pos_probability = 1
neg_probability = 1

# list of review splitting by each word
review_words = review.split()

# iterate through review words
for word in review_words:
  # retrive the count from dictionary of specifc word appaering
  word_in_pos = pos_counter[word]
  word_in_neg = neg_counter[word]


  pos_probability =  pos_probability * (word_in_pos / total_pos)

  neg_probability =  neg_probability * (word_in_neg / total_neg)

print(pos_probability)
print(neg_probability)
