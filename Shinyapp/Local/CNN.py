import numpy as np
import matplotlib.pyplot as plt



import tensorflow as tf
#from tensorflow.keras.preprocessing.image import load_img, img_to_array, smart_resize


# can_test1, cardboard_test1, glass_bottle_test1, plastic_bottle_test1
#img = load_img("plastic_bottle_test3.png") # load image as grayscale
img = tf.keras.preprocessing.image.load_img("plastic_bottle_test3.png") # load image 

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('model_2_94p.h5')

def predictions(image):
     img_array = tf.keras.preprocessing.image.img_to_array(image) # convert the image to an NumPy array
     img_array_resized = tf.keras.preprocessing.image.smart_resize(img_array, size = (128, 128)) / 255.0 # if necessary, resize the image to 128 by 128
     reshaped_image = np.expand_dims(img_array_resized, axis=0) # reshape the image from (height, width, 3) to (1, height, width, 3) as an input to the CNN model
     probabilities = new_model.predict(reshaped_image)
     probabilities = np.round(probabilities[0,:],4 ) * 100
     return probabilities

def probability_chart(probabilities):
     x = np.array([0, 1, 2, 3])  # X-axis values
     labels = ['Can', 'Cardboard', 'Glass bottle', 'Plastic Bottle']  # Replace with your labels
     plt.bar(x, probabilities)
     plt.xticks(x, labels)  # Assigning labels to x-axis ticks
     plt.xlabel('Object')
     plt.ylabel('Probability')
     plt.title('Bar Chart')
     for i, value in enumerate(probabilities):
         chart = plt.text(i, value, str(value), ha='center', va='bottom')
     return plt.show()

#p = predictions(img)
#probability_chart(p)







