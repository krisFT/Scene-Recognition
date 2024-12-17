# **Scene Recognition Project**

This project implements a **Scene Recognition Pipeline** to classify images into predefined scene categories using various feature extraction techniques and classifiers. The pipeline explores different methods for feature representation, spatial enhancements, and advanced classification strategies to achieve optimal results.

---

## **Pipeline Overview**

The project includes the following feature extraction and classification methods:

1. **Tiny Image Features**:  
   - `tiny_image + Nearest Neighbor (NN)`  
   - `tiny_image + Support Vector Machine (SVM)`  

2. **Bag of Words (BoW)**:  
   - `BoW + Nearest Neighbor (NN)`  
   - `BoW + SVM`  

3. **Spatial Enhancements**:  
   - Added spatial information using **Gaussian Pyramid** and **Spatial Pyramid BoW**.

4. **Advanced SVM**:  
   - Implemented **SVM with RBF kernel** and tuned it using **Grid Search** to find the best `C` and `gamma` parameters.

5. **Multi-Scale Features**:  
   - Features extracted at multiple scales to improve robustness to size variations.

6. **Combined Features**:  
   - Combined BoW histograms with **GIST descriptors** and **Self-Similarity descriptors** for a richer representation.

7. **Gaussian Mixture Model (GMM)**:  
   - Replaced k-means clustering with GMM for BoW feature quantization using **soft assignment**.

---

## **Results**

After extensive experimentation, the **best performance achieved** is:


| Method                   | Accuracy (%) |
|--------------------------|-------------:|
| Tiny Image + NN          | 20.9         |
| Tiny Image + SVM         | 19.9         |
| BoW + NN                 | 58.5         |
| BoW + SVM                | 70.2         |
| BoW (Multi-Scale) + SVM_RBF (Grid Search) | **75.9**     |

Despite adding **combined features** (BoW + GIST + Self-Similarity) and trying **GMM** for feature quantization, further improvements were not observed.  

### *Why Didn't Combined Features and GMM Improve Accuracy?*

1. **Feature Redundancy**:  
   Combined features (BoW, GIST, and self-similarity) may not provide truly complementary information. For example, both GIST and HOG (used in BoW) capture spatial patterns, leading to feature overlap rather than added discriminative power.

2. **GMM Limitations**:  
   GMM assumes clusters follow a Gaussian distribution, which may not align with the actual feature distribution. For sparse histograms like BoW, k-means (hard assignment) often performs better.

3. **Overfitting**:  
   Combining features or using GMM increases feature dimensionality, which can lead to overfitting, particularly if the training dataset is small relative to the number of features.

---

## **Key Functions in `utils.py`**

The project provides reusable utility functions for feature extraction, vocabulary building, and classification:

### **Feature Extraction**
- `get_tiny_images(image_paths)`  
- `build_vocabulary(image_paths, vocab_size, num_levels=3)` *(k-means-based)*  
- `build_vocabulary_with_gmm(image_paths, n_components)` *(GMM-based)*  
- `get_bags_of_words(image_paths, levels=3)`  
- `get_bags_of_words_with_gmm(image_paths, levels=3)`
- `combined_features(image_paths, levels=3)`

### **Spatial and Multi-Scale**
- `generate_gaussian_pyramid(image, num_levels=3)`  
- `spatial_pyramid_bow(features, vocab, levels=3)`  

### **Complementary Features**
- `compute_gist(image, num_scales=4, num_orientations=8)`  
- `compute_self_similarity(image)`  

### **Classification**
- `nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, k=5)`  
- `svm_classify(train_image_feats, train_labels, test_image_feats)`  
- `svm_classify_rbf(train_image_feats, train_labels, test_image_feats, C=10, gamma=0.1)`  *(Best tuned)*
- `svm_classify_rbf_with_grid_search(train_image_feats, train_labels, test_image_feats, param_grid=None)`

---

## **How to Run**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/scene-recognition.git
   cd scene-recognition
   ```
2. **Install required libraries in your virtual environment**:
   ```bash
   pip install numpy scikit-learn matplotlib opencv-python scikit-image
   ```
3. **Run the pipeline:**
   ```bash
   python Scene_Recognition.py --feature bag_of_words --classifier SVM_RBF
   ```
4. **Experiment with other options:**
   Use the following arguments to explore different combinations:
  
  - **`--feature`**:  
    - `tiny_image`  
    - `bag_of_words`  
    - `bow_gmm`  
    - `combined_features`  
  
  - **`--classifier`**:  
    - `NN`  
    - `SVM`  
    - `SVM_RBF`  
    - `SVM_RBF_GridSearch`  
---

## **Conclusion**
This project explores various methods for scene recognition, with BoW + SVM_RBF achieving the best accuracy of 75.9%. Despite integrating advanced techniques like multi-scale features, GMM, and combined features, further improvements remain elusive.

If you find a way to push this accuracy higher, let me know! 🚀
