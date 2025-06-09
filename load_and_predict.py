import tensorflow as tf
import numpy as np

# Cargar el modelo
model = tf.keras.models.load_model('asl_alphabet.h5')

# Ejemplo: Predecir una imagen dummy (debes reemplazar esto por una imagen real preprocesada)
# Para un modelo entrenado con imágenes 64x64 en escala de grises:
dummy_image = np.random.rand(64, 64, 1)  # Imagen aleatoria para test
dummy_image = np.expand_dims(dummy_image, axis=0)  # Añadir dimensión batch

pred = model.predict(dummy_image)
predicted_class = np.argmax(pred)

print(f'Clase predicha: {predicted_class}')
