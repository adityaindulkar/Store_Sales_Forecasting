# Sales Forecasting with AWS SageMaker

## Project Overview
Developed and deployed a ML sales forecasting pipeline that predicts monthly revenue by product category and region. Engineered temporal features (3-month rolling mean, YoY growth, holiday/weekend indicators, lag features) and log transformation to handle skewness, rolling mean was the primary indicator. Optimized multiple ML models, implementing a robust ensemble method that averaged their predictions to enhance accuracy. Deployed the end-to-end solution on AWS cloud infrastructure; utilized S3, SageMaker, ECR and Docker for storage, model development, training, visualization and deployment.

## Tech Stack: 
### Python (Numpy, Pandas, Scikit-Learn, Matplotlib)     
### AWS (Sagemaker, S3, Dockerfile, ECR, Jupyter Notebook)

## Key Features
**Advanced Feature Engineering**: Time-based features, holiday/weekend indicators, cyclical encoding, rolling statistics, lag features, and target encoding              
**Ensemble Modeling**: Combines Gradient Boosting, XGBoost, and LightGBM with residual correction                 
**AWS Integration**: S3 bucket storage, custom Docker container, and SageMaker endpoint deployment           
**Production-Ready**: Includes model serialization, containerization, and REST API endpoint            
**Real-time Predictions**: REST API endpoint for making sales predictions           

## Data Preprocessing
Date parsing and normalization      
State abbreviation mapping        
Holiday detection using US federal and state holidays           
Logarithmic transformation of sales data            
Advanced time feature engineering (cyclical encoding, month indicators)            

## Feature Engineering
Time-based features: year, month, quarter, week, day of week, etc.         
Cyclical encoding for periodic features        
Holiday and weekend indicators           
Rolling statistics (3, 6, 12 months)         
Lag features (1, 2, 3, 6, 12 months)        
Growth rate calculations (YoY, MoM)           
Target encoding for high-cardinality features         

## Modeling Approach
**Ensemble Model**: Voting regressor combining:       
  - Gradient Boosting Regressor           
  - XGBoost Regressor         
  - LightGBM Regressor
      
**Residual Correction**: Secondary model to correct ensemble errors

## Performance Metrics: MAE, RMSE, and MAPE
**MAE** - $6.5 - $7.5      
**MAPE** - 6.5% - 7.5%            
**RMSE** - $12 - $13        

## AWS Infrastructure
S3 bucket for data storage and model artifacts            
Custom Docker container for model serving           
SageMaker endpoint for production inference         
ECR repository for container management     

## Running the Pipeline
Execute the main notebook to process data and train models
The pipeline will automatically:            
  Create S3 bucket       
  Upload data            
  Preprocess and engineer features          
  Train and evaluate models                 
  Build and push Docker container       
  Deploy SageMaker endpoint        
