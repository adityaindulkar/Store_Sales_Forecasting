import os
import joblib
import json
import numpy as np
import pandas as pd
import logging
from flask import Flask, request, jsonify

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = Flask(__name__)

# Global variable for the model
model = None

def load_model():
    """Load the model when the Flask app starts"""
    global model
    try:
        model_dir = "/opt/ml/model"
        model_path = os.path.join(model_dir, "model.joblib")
        logger.info(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint"""
    try:
        if model is not None:
            return jsonify({"status": "healthy"}), 200
        else:
            return jsonify({"status": "unhealthy", "reason": "Model not loaded"}), 500
    except Exception as e:
        return jsonify({"status": "unhealthy", "reason": str(e)}), 500

@app.route('/invocations', methods=['POST'])
def invocations():
    """Prediction endpoint"""
    try:
        # Get input data
        input_data = input_fn(request.get_data(), request.content_type)
        
        # Make prediction
        prediction = predict_fn(input_data, model)
        
        # Return response
        return output_fn(prediction, request.content_type)
    except Exception as e:
        logger.error(f"Error during invocation: {str(e)}")
        return jsonify({"error": str(e)}), 500

def input_fn(request_body, request_content_type):
    """Deserialize input data"""
    try:
        if request_content_type == "application/json":
            data = json.loads(request_body)
            logger.info(f"Received input data: {data}")
            
            # Convert to DataFrame and ensure proper feature order
            df = pd.DataFrame([data])
            
            # Log the features received
            logger.info(f"Input features: {list(df.columns)}")
            
            return df
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
    except Exception as e:
        logger.error(f"Error processing input: {str(e)}")
        raise

def predict_fn(input_data, model):
    """Run prediction with residual correction"""
    try:
        # Log the expected feature names
        logger.info(f"Model expects features: {model['feature_names']}")
        
        # Check if all required features are present
        missing_features = set(model['feature_names']) - set(input_data.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Align input with trained features
        features = input_data[model['feature_names']]
        
        # Log the input features being used for prediction
        logger.info(f"Features for prediction: {features.to_dict()}")
        
        # Step 1: Ensemble prediction
        ensemble_pred = model['ensemble'].predict(features)
        logger.info(f"Ensemble prediction (log scale): {ensemble_pred}")
        
        # Step 2: Residual correction
        residual_pred = model['residual_correction'].predict(features)
        logger.info(f"Residual prediction: {residual_pred}")
        
        # Step 3: Combine and inverse transform
        final_pred_log = ensemble_pred + residual_pred
        final_pred = np.expm1(final_pred_log)
        logger.info(f"Final prediction (exp scale): {final_pred}")
        
        return final_pred.tolist()
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def output_fn(prediction, response_content_type):
    """Serialize output"""
    try:
        if response_content_type == "application/json":
            response = {
                "prediction": prediction,
                "status": "success"
            }
            logger.info(f"Returning prediction: {response}")
            return jsonify(response)
        else:
            raise ValueError(f"Unsupported content type: {response_content_type}")
    except Exception as e:
        logger.error(f"Error formatting output: {str(e)}")
        raise

if __name__ == '__main__':
    # Load model when starting the app
    load_model()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=8080, debug=False)