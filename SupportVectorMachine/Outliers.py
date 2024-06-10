import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from graph import points, labels, draw_points, draw_margin

classifier = SVC(kernel='linear', C = 1)
classifier.fit(points, labels)

draw_points(points, labels)
draw_margin(classifier)

plt.show()
