# Exoplanet Detection Using Machine Learning

## Project Overview
This project focuses on detecting exoplanets using the Transit Method by analyzing the brightness of stars. The Kepler telescope data, which records the brightness levels of distant stars, is utilized. The fluctuations in brightness levels indicate the presence of a planet orbiting the star.

## Methodologies Used

### Data Normalization
Mean normalization was applied to both training and test datasets to standardize the feature values, ensuring each feature contributes equally to the model's performance.

### Fourier Transformation
Fourier Transformation was used to convert the time series data of brightness levels into the frequency domain. This helps in identifying periodic fluctuations in brightness levels, indicative of an exoplanet.

### Oversampling
SMOTE (Synthetic Minority Over-Sampling Technique) was applied to handle the class imbalance in the dataset. This technique synthesizes new data points for the minority class to balance the dataset.

### Machine Learning Models
1. **Random Forest Classifier**
   - Initially deployed to classify stars but failed to detect stars with planets accurately.
2. **XGBoost Classifier**
   - Deployed as an alternative to Random Forest. This model showed better performance in classifying stars with planets.

## Programming Languages Used
- Python

## Libraries and Tools
- pandas
- numpy
- scikit-learn
- imbalanced-learn (for SMOTE)
- xgboost

## Dataset Description
- The dataset contains brightness measurements recorded by the Kepler telescope.
- Features include time-series data of brightness (FLUX) for various stars.
- Target variable: LABEL (1 for stars without a planet, 2 for stars with a planet).

## Steps to Run the Project

1. **Data Loading**
   - Load the training and test datasets.

2. **Data Normalization**
   - Apply mean normalization to standardize the datasets.

3. **Fourier Transformation**
   - Transform the FLUX values from the time domain to the frequency domain.

4. **Oversampling Using SMOTE**
   - Apply SMOTE to balance the class distribution in the training dataset.

5. **Model Training and Evaluation**
   - Train the Random Forest Classifier and evaluate its performance.
   - If Random Forest fails, train the XGBoost Classifier and evaluate its performance.

## Project Structure
- `data/` - Directory containing the dataset files.
- `notebooks/` - Jupyter notebooks for data exploration and model training.
- `src/` - Source code for data processing and model implementation.
- `results/` - Directory to save the results and model evaluation metrics.

## Conclusion
Despite extensive data processing and normalization, the Random Forest Classifier failed to detect exoplanets accurately. The XGBoost Classifier, however, showed better performance. This project demonstrates the importance of selecting appropriate models and data processing techniques in machine learning tasks, especially for imbalanced datasets.

## References
- [How Do Astronomers Find Exoplanets?](#)
- [Transiting Exoplanet Light Curve](#)
