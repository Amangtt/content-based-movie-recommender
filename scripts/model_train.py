import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Layer
from keras.saving import register_keras_serializable
from model_preprocess import preprocess
import mlflow
from logger import SetupLogger
logger = SetupLogger(log_file='./logs/notebooks.log').get_logger()
# Define the custom L2Normalize layer
@register_keras_serializable()  # Register the custom layer for serialization
class L2Normalize(Layer):
    def __init__(self, **kwargs):
        super(L2Normalize, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.linalg.l2_normalize(inputs, axis=1)

def train(user_path, movies_path, rating_path):
    try:
        mlflow.set_experiment("Movie Recommender - Training")

        
        user = pd.read_csv(user_path)
        movies = pd.read_csv(movies_path)
        rating = pd.read_csv(rating_path)
        num_user_features = user.shape[1] - 3  # remove userid, rating count, and ave rating during training
        num_item_features = movies.shape[1] - 1  # remove movie id at train time

        u_s = 3  # start of columns to use in training, user
        i_s = 1
        user_train, user_test, movie_train, movie_test, y_train, y_test, scalerUser, scalerItem,scalerTarget = preprocess(user, movies, rating)
            
        tf.random.set_seed(1)
            
            # Define the user neural network
        User_NN = tf.keras.models.Sequential([
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(32)
            ])
            
            # Define the movies neural network
        movies_NN = tf.keras.models.Sequential([
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(32)
            ])
            
            # Define the input layers
        input_user = tf.keras.layers.Input(shape=(num_user_features,))
        input_item = tf.keras.layers.Input(shape=(num_item_features,))
            
            # Pass inputs through the networks
        vu = User_NN(input_user)
        vm = movies_NN(input_item)
            
            # Normalize the outputs using the custom layer
        vu = L2Normalize()(vu)
        vm = L2Normalize()(vm)
            
            # Compute the dot product
        output = tf.keras.layers.Dot(axes=1)([vu, vm])
            
            # Define the model
        model = tf.keras.Model([input_user, input_item], output)
            
            # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
                        loss=tf.keras.losses.MeanSquaredError())
            
            # Train the model
        history=model.fit([user_train[:, u_s:], movie_train[:, i_s:]], y_train, epochs=30)
            # Log metrics
        mlflow.log_metric("final_loss", history.history["loss"][-1])
            # Save the model
        model.save("./model/model.keras", save_format="keras")
        mlflow.log_artifact("./model/model.keras")
        logger.info('Successfully completed model training')
        return model
    except Exception as e:
        error_message = f"Failed to preprocess model: {e}"
        logger.error(error_message)