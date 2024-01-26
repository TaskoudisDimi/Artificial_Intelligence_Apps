
#############################################################################################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv("C:/Users/chris/Desktop/Dimitris/Tutorials/AI/data/mnist_train.csv")
test_data = pd.read_csv("C:/Users/chris/Desktop/Dimitris/Tutorials/AI/data/mnist_test.csv")



round(train_data.drop('label', axis=1).mean(), 2)

## Separating the X and Y variable

y = train_data['label']

## Dropping the variable 'label' from X variable 
X = train_data.drop(columns = 'label')

## Printing the size of data 
print(train_data.shape)


## Normalization

X = X/255.0
test_data = test_data/255.0

print("X:", X.shape)
print("test_data:", test_data.shape)


# scaling the features
from sklearn.preprocessing import scale
X_scaled = scale(X)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, train_size = 0.2 ,random_state = 10)

model = SVC(C=10, gamma=0.001, kernel="rbf")

# fit
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# metrics
print("accuracy", metrics.accuracy_score(y_test, y_pred), "\n")
print(metrics.confusion_matrix(y_test, y_pred), "\n")



model_filename = 'C:/Users/chris/Desktop/Dimitris/Tutorials/AI/Computational-Intelligence-and-Statistical-Learning/WebApp/Models/SVM_model_Mnist.pkl'
joblib.dump(model, model_filename)



# # Plot some random images
# num_images_to_plot = 5
# fig, axes = plt.subplots(1, num_images_to_plot, figsize=(15, 3))

# for i in range(num_images_to_plot):
#     # Randomly select an image from the dataset
#     random_idx = np.random.randint(0, len(X_scaled))
#     image = X_scaled[random_idx].reshape(28, 28)
    
#     # Get the corresponding label
#     label = y[random_idx]
    
#     # Plot the image
#     axes[i].imshow(image, cmap='gray')
#     axes[i].set_title(f"Label: {label}")
#     axes[i].axis('off')

# plt.show()



