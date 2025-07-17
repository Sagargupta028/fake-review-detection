import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np

# Create a simple dummy model that can be loaded
# This is just to make the app run - it won't be accurate but will prevent the error
model = LogisticRegression()
# Create some dummy data to fit the model
X_dummy = np.random.rand(100, 10)
y_dummy = np.random.randint(0, 2, 100)
model.fit(X_dummy, y_dummy)

# Save the model
pickle.dump(model, open('model.pkl', 'wb'))
print("Created model.pkl file") 