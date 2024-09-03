# Hyperspectral-Imaging
This repository contains code for the visualization, manipulation and classification of hyperspectral images, with a specific focus on the identification of fruit seeds and pulp. The analysis is performed using image processing and machine learning techniques, applied to data from the Giessen database.
The objective of this project is to serve as a beginner's guide on how to work with hyperspectral images. The codes used represent attempts to understand these interesting tools (HSI). Overall, the intention was to explore the detailed information available across multiple spectral bands, which allows for precise differentiation of materials based on their spectral signatures
## Project Features:  
- **Reading and Handling Hyperspectral Images:** Use of HDR and RAW files to load and process hyperspectral images using the spectral library.  
- **Spectra Extraction:** Selection of specific pixels corresponding to seeds and pulp to extract their characteristic spectra.  
- **Statistical Moments Calculation:** Calculation of mean, standard deviation, skewness, and kurtosis of the spectra to use as features in the classification.
- **Classification with Support Vector Machine (SVM):** Implementation of a SVM classifier to differentiate between seed and pulp pixels based on statistical moments.
- **Classification with K-Nearest Neighbors (KNN):** Implementation of a KNN classifier to differentiate between seed and pulp pixels based on statistical moments.  
- **Decision Boundary Visualization:** Plots that show how the KNN classifier separates the data based on different pairs of statistical features.  

## Code Structure: 
- **Metadata Reading:** Read the metadata from the HDR file to get the image dimensions and data type.  
- **Hyperspectral Image Loading:** Load the image using the HDR and RAW files.  
- **Coordinate Selection:** Define and validate the pixel coordinates corresponding to seeds and pulp.  
- **Statistical Moments Calculation:** Calculate statistical moments (mean, standard deviation, skewness, kurtosis) for the spectra of the selected pixels.  
- **Model Training and Evaluation:** Train a SVM and KNN classifier using a portion of the data and evaluate its performance on a test set.  
- **Results Visualization:** Create plots to show the classifier's decision boundaries and the distribution of test data.  

**Requirements**  
- Python 3.x  
- Libraries: `spectral`, `numpy`, `matplotlib`, `scipy`, `sklearn`  

## How to Use:  
1. **Install Dependencies:** Make sure all required libraries are installed.  
2. **Configure Directories and Files:** Adjust the paths to the HDR and RAW files in the code according to your directory structure.  
3. **Run the Code:** Execute the script to perform classification and visualize the results.  

## Future Improvements:  
- Implement other classification algorithms to compare results.  
- Include preprocessing techniques to improve the quality of spectral data.  
- Expand the analysis to other types of materials.
