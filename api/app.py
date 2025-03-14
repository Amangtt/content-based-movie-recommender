from flask import Flask, jsonify, request,Response
import tensorflow as tf
import numpy as np
import os, sys
from numpy import genfromtxt
import joblib
# Add the 'scripts' directory to the Python path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from scripts import load
from keras.saving import register_keras_serializable

# Define the custom L2Normalize layer
@register_keras_serializable()  # Register the custom layer for serialization
class L2Normalize(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(L2Normalize, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.linalg.l2_normalize(inputs, axis=1)

app=Flask(__name__)
model = tf.keras.models.load_model(
    "./model/model.keras",
    custom_objects={"L2Normalize": L2Normalize}
)
scalerUser=joblib.load('./model/scalerUser.pkl')
scalerItem=joblib.load('./model/scalerItem.pkl')
scalerTarget=joblib.load('./model/scalerTarget.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.json
        print("Received Data:", data)
        user_vec = [
            data.get('new_user_id', 0),
            data.get('new_rating_count', 0),
            data.get('new_rating_ave', 0.0),
            data.get('new_action', 0.0),
            data.get('new_adventure', 0.0),
            data.get('new_animation', 0.0),
            data.get('new_childrens', 0.0),
            data.get('new_comedy', 0.0),
            data.get('new_crime', 0.0),
            data.get('new_documentary', 0.0),
            data.get('new_drama', 0.0),
            data.get('new_fantasy', 0.0),
            data.get('new_horror', 0.0),
            data.get('new_mystery', 0.0),
            data.get('new_romance', 0.0),
            data.get('new_scifi', 0.0),
            data.get('new_thriller', 0.0),
            
        ]
        
        user_vec = np.array(user_vec)
        print("User Vector:", user_vec)
        item_vecs = genfromtxt('./Data/content_item_vecs.csv', delimiter=',')
        movie_dict = load.load_data()
        user_vecs = load.gen_user_vecs(user_vec, len(item_vecs))

        # Scale our user and item vectors
        suser_vecs = scalerUser.transform(user_vecs)
        sitem_vecs = scalerItem.transform(item_vecs)

        # Make a prediction
        y_p = model.predict([suser_vecs[:, 3:], sitem_vecs[:, 1:]])

        # Unscale y prediction 
        y_pu = scalerTarget.inverse_transform(y_p)

        # Sort the results, highest prediction first
        sorted_index = np.argsort(-y_pu, axis=0).reshape(-1).tolist()  # Negate to get largest rating first
        sorted_ypu = y_pu[sorted_index]
        sorted_items = item_vecs[sorted_index]  # Using unscaled vectors for display

        predictions_json = load.predictions_to_json(sorted_ypu, sorted_items, movie_dict)

        return jsonify(predictions_json)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True,port=5000)
