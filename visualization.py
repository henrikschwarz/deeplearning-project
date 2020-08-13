from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

model_used = "my_model-20200813-215626.h5"
model = load_model("models/" + model_used)

img = ImageDataGenerator(rescale=1. / 255)
test_gen = img.flow_from_directory(
    directory='./dataset_catdog/dataset/test',
    target_size=(150, 150),
    color_mode='rgb'
)

labels = test_gen.labels
result = np.round(model.predict(test_gen))
# incorrect_labeled = result != labels

print(result)
#for i in range(len(result)):
#    print(f"Result predict: {result[i]}, actual label {labels[i]}")

# print(incorrect_labeled)

# print(test_gen[0][0][0].shape)
# plt.imshow(test[0][0][0])
# plt.show()