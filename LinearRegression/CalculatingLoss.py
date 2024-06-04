x = [1, 2, 3]
y = [5, 1, 3]

#y = x
m1 = 1
b1 = 0

y_predicted1 = [m1*x_values + b1 for x_values in x]

print(y_predicted1)

total_loss1 = 0

for i in range(len(y)):
  total_loss1 += (y[i] - y_predicted1[i]) ** 2

print(total_loss1)


#y = 0.5x + 1
m2 = 0.5
b2 = 1

y_predicted2 = [m2*x_values + b2 for x_values in x]

total_loss2 = 0

for i in range(len(y)):
  total_loss2 += (y[i] - y_predicted2[i]) ** 2

print(total_loss2)

better_fit = -1

if total_loss1 < total_loss2:
  better_fit = 1
else:
  better_fit = 2
