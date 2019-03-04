
#%%
from keras import layers
from keras import models
from keras import losses
from keras import optimizers

import matplotlib.pyplot as plt
import numpy as np


sizeMb = np.array([0.080, 9.000, 0.001, 0.100, 8.000, 5.000, 0.100, 6.000, 0.050, 0.500, 0.002, 2.000, 0.005, 10.00, 0.010, 7.000, 6.000, 5.000, 1.000, 1.000,
        5.000, 0.200, 0.001, 9.000, 0.002, 0.020, 0.008, 4.000, 0.001, 1.000, 0.005, 0.080, 0.800, 0.200, 0.050, 7.000, 0.005, 0.002, 8.000, 0.008])

timeSec = np.array([0.135, 0.739, 0.067, 0.126, 0.646, 0.435, 0.069, 0.497, 0.068, 0.116, 0.070, 0.289, 0.076, 0.744, 0.083, 0.560, 0.480, 0.399, 0.153, 0.149,
        0.425, 0.098, 0.052, 0.686, 0.066, 0.078, 0.070, 0.375, 0.058, 0.136, 0.052, 0.063, 0.183, 0.087, 0.066, 0.558, 0.066, 0.068, 0.610, 0.057])

train_input = sizeMb[:20]
train_labels = timeSec[:20]
test_input = sizeMb[20:]
test_labels = timeSec[20:]

model = models.Sequential()
model.add(layers.Dense(1, input_shape=(1,)))

#%%
model.compile(optimizer="sgd", loss="mae")

#%%
model.fit(train_input, train_labels, epochs=200)

#%%

print(model.evaluate(test_input, test_labels))

#%%
smallFileMb = np.array([1])
predictedTime = model.predict(smallFileMb)
print("Prediction for a file of size {}MB is {} second(s)".format(smallFileMb, predictedTime))

#%%
plt.scatter(train_input, train_labels)
plt.scatter(test_input, test_labels)
plt.title("File download duration")
plt.xlabel("size (MB)")
plt.ylabel("time (sec)")
plt.show()