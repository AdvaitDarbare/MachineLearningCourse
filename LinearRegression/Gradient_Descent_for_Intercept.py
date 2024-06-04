def get_gradient_at_b(x, y, m, b):
  diff = 0
  N = len(x)
  b_gradient = -2/N
  for i in range(len(x)):
    diff += (y[i] - (m*x[i]+b))
  b_gradient = b_gradient * diff
  return b_gradient
