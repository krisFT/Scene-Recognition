import numpy as np
from numpy.linalg import norm
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.feature import hog, match_template
from skimage.transform import resize
from skimage.filters import gabor
from sklearn.feature_extraction.image import extract_patches_2d
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
import os
import glob
import matplotlib.pyplot as plt
import warnings
import cv2

warnings.filterwarnings('ignore', '', UserWarning)


def get_tiny_images(image_paths):
    """
    Generates tiny image features by resizing images to a small resolution.

    Args:
        image_paths (list): List of image file paths.
    Returns:
        np.ndarray: Array of tiny image feature vectors (n x d).
    """
    tiny_images = [
        resize(imread(image_path), (16, 16)).flatten() 
        for image_path in image_paths
    ]
    return np.array(tiny_images)


def generate_gaussian_pyramid(image, num_levels=3):
    """
    Generate a Gaussian pyramid for an image.
    Args:
        image (np.ndarray): Input image.
        num_levels (int): Number of pyramid levels.
    Returns:
        list: List of images at different pyramid levels.
    """
    pyramid = [image]
    for _ in range(num_levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid


def build_vocabulary(image_paths, vocab_size, num_levels=3):
    """
    Builds a visual vocabulary using multi-scale HOG features.
    Args:
        image_paths (list): List of image file paths.
        vocab_size (int): Number of clusters for the vocabulary.
        num_levels (int): Number of levels for the Gaussian pyramid.
    Returns:
        np.ndarray: Array of cluster centers (vocab_size x feature_dim).
    """
    cells_per_block = (4, 4)
    pixels_per_cell = (8, 8)
    feature_dim = cells_per_block[0] * cells_per_block[1] * 9

    features = []

    for image_path in image_paths:
        img = imread(image_path)
        
        # Multi-Scale HOG
        pyramid = generate_gaussian_pyramid(img, num_levels=num_levels)
        for level in pyramid:
            hog_features = hog(
                level,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                feature_vector=True
            ).reshape(-1, feature_dim)
            features.append(hog_features)

    # Stack features and cluster with MiniBatchKMeans
    features = np.vstack(features)
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, max_iter=100, n_init=10, random_state=0)
    kmeans.fit(features)

    return kmeans.cluster_centers_

def build_vocabulary_with_gmm(image_paths, n_components):
    """
    Builds a visual vocabulary using GMM for feature quantization.

    Args:
        image_paths (list): List of image file paths.
        n_components (int): Number of GMM components (analogous to vocabulary size).

    Returns:
        GaussianMixture: Fitted GMM model.
    """
    cells_per_block = (4, 4)
    pixels_per_cell = (8, 8)
    feature_dim = cells_per_block[0] * cells_per_block[1] * 9

    features = []  # Collect all HOG features

    for image_path in image_paths:
        img = imread(image_path)
        hog_features = hog(
            img,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            feature_vector=True
        ).reshape(-1, feature_dim)
        features.append(hog_features)

    features = np.vstack(features)  # Stack all features into a single matrix

    # Fit a GMM model
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=0)
    gmm.fit(features)

    return gmm

# def build_vocabulary(image_paths, vocab_size): # no multi-scale
#     """
#     Builds a visual vocabulary by clustering HOG features from training images.

#     Args:
#         image_paths (list): List of image file paths.
#         vocab_size (int): Number of clusters for the vocabulary.
#     Returns:
#         np.ndarray: Array of cluster centers (vocab_size x feature_dim).
#     """
#     cells_per_block = (4, 4)
#     pixels_per_cell = (4, 4)
#     feature_dim = cells_per_block[0] * cells_per_block[1] * 9

#     # Collect HOG features from all images
#     features = np.vstack([
#         hog(imread(image_path), 
#             pixels_per_cell=pixels_per_cell, 
#             cells_per_block=cells_per_block, 
#             feature_vector=True
#         ).reshape(-1, feature_dim)
#         for image_path in image_paths
#     ])

#     # Cluster features to build vocabulary
#     kmeans = MiniBatchKMeans(n_clusters=vocab_size, max_iter=100, random_state=0)
#     kmeans.fit(features)

#     return kmeans.cluster_centers_

# def get_bags_of_words(image_paths): #no pyramid
#     """
#     Computes Bag of Words histograms for a list of images.

#     Args:
#         image_paths (list): List of file paths to images.
#     Returns:
#         np.ndarray: n x d matrix of BoW histograms (n = number of images, d = vocab size).
#     """
#     vocab = np.load('vocab.npy')
#     print('Loaded vocabulary from file.')
    
#     cells_per_block = (4, 4)
#     pixels_per_cell = (4, 4)
#     vocab_size = len(vocab)
    
#     histograms = []

#     for image_path in image_paths:
#         # Extract HOG features
#         img = imread(image_path)
#         feature_vector = hog(
#             img,
#             pixels_per_cell=pixels_per_cell,
#             cells_per_block=cells_per_block,
#             feature_vector=True
#         ).reshape(-1, cells_per_block[0] * cells_per_block[1] * 9)

#         # Find closest visual words
#         distances = cdist(vocab, feature_vector, metric='euclidean')
#         closest_words = np.argmin(distances, axis=0)

#         # Build histogram and normalize
#         histogram, _ = np.histogram(closest_words, bins=vocab_size, range=(0, vocab_size))
#         histogram = histogram / norm(histogram)  # Normalize histogram

#         histograms.append(histogram)

#     return np.array(histograms)

def spatial_pyramid_bow(features, vocab, levels=3):
    """
    Compute Spatial Pyramid Representation for Bag of Words.
    Args:
        features (np.ndarray): Feature descriptors for an image (n x d matrix).
        vocab (np.ndarray): Vocabulary (visual words).
        levels (int): Number of pyramid levels.
    Returns:
        np.ndarray: Spatial pyramid feature vector.
    """
    weights = [1 / (2 ** level) for level in range(levels)]
    vocab_size = vocab.shape[0]
    pyramid_feature = []

    for level in range(levels):
        num_cells = 2 ** level  # Number of cells along each axis
        step_size = features.shape[0] // num_cells  # Divide features into grid

        for i in range(num_cells):
            for j in range(num_cells):
                # Extract features in the current grid cell
                start_idx = i * step_size
                end_idx = (i + 1) * step_size
                cell_features = features[start_idx:end_idx, :]  # Slice features

                # Ensure cell_features matches vocab feature dimension
                if cell_features.shape[1] != vocab.shape[1]:
                    raise ValueError("Feature dimension of cell_features does not match vocabulary.")

                # Compute distances and build histogram
                distances = cdist(cell_features, vocab, metric='euclidean')
                closest_words = np.argmin(distances, axis=1)
                histogram, _ = np.histogram(closest_words, bins=vocab_size, range=(0, vocab_size))
                histogram = histogram / norm(histogram)  # Normalize histogram

                # Weight the histogram and append
                pyramid_feature.append(weights[level] * histogram)

    return np.hstack(pyramid_feature)

def get_bags_of_words(image_paths, levels=3):
    """
    Computes Bag of Words histograms with spatial pyramid for a list of images.
    Args:
        image_paths (list): List of file paths to images.
        levels (int): Number of levels in the spatial pyramid.
    Returns:
        np.ndarray: n x d matrix of BoW histograms (n = number of images, d = vocab size).
    """
    vocab = np.load('vocab.npy')
    print('Loaded vocabulary from file.')

    cells_per_block = (4, 4)
    pixels_per_cell = (8, 8)
    vocab_size = len(vocab)
    feature_dim = cells_per_block[0] * cells_per_block[1] * 9

    histograms = []

    for image_path in image_paths:
        img = imread(image_path)

        # Extract Multi-Scale HOG
        pyramid = generate_gaussian_pyramid(img, num_levels=3)
        hog_features = np.vstack([
            hog(
                level,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                feature_vector=True
            ).reshape(-1, feature_dim)
            for level in pyramid
        ])

        # Compute Spatial Pyramid BoW
        spatial_histogram = spatial_pyramid_bow(hog_features, vocab, levels=levels)
        histograms.append(spatial_histogram)

    return np.array(histograms)


def get_bags_of_words_with_gmm(image_paths, levels=3):
    """
    Computes Bag of Words histograms with GMM-based soft assignment.

    Args:
        image_paths (list): List of image file paths.
        gmm (GaussianMixture): Pre-trained GMM model.
        levels (int): Number of levels in the spatial pyramid.

    Returns:
        np.ndarray: Bag of Words histograms (n x d, where d = number of GMM components).
    """
    gmm = np.load('vocab_gmm.npy')
    print('Loaded vocabulary from file.')

    cells_per_block = (4, 4)
    pixels_per_cell = (8, 8)
    vocab_size = gmm.n_components  # Number of GMM components
    feature_dim = cells_per_block[0] * cells_per_block[1] * 9

    histograms = []

    for image_path in image_paths:
        img = imread(image_path)

        # Generate Gaussian Pyramid for multi-scale HOG
        pyramid = generate_gaussian_pyramid(img, num_levels=levels)

        # Extract multi-scale HOG features
        hog_features = np.vstack([
            hog(
                level,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                feature_vector=True
            ).reshape(-1, feature_dim)
            for level in pyramid
        ])

        # Compute probabilities for each feature vector using GMM
        probabilities = gmm.predict_proba(hog_features)

        # Sum probabilities across all feature vectors to form the histogram
        histogram = np.sum(probabilities, axis=0)

        # Normalize histogram
        histogram = histogram / np.linalg.norm(histogram)

        histograms.append(histogram)

    return np.array(histograms)



def compute_gist(image, num_scales=4, num_orientations=8):
    """
    Compute GIST descriptor for an image using Gabor filters.

    Args:
        image (np.ndarray): Input image.
        num_scales (int): Number of scales (frequency levels) for Gabor filters.
        num_orientations (int): Number of orientations for Gabor filters.

    Returns:
        np.ndarray: GIST descriptor.
    """
    if len(image.shape) == 3:  # Convert RGB to grayscale
        image = rgb2gray(image)

    # Initialize GIST descriptor
    gist_descriptor = []

    # Iterate through scales and orientations
    for scale in range(1, num_scales + 1):
        for orientation in np.linspace(0, np.pi, num_orientations, endpoint=False):
            # Apply Gabor filter
            real, _ = gabor(image, frequency=0.2 * scale, theta=orientation)
            
            # Compute the mean of the filtered response
            gist_descriptor.append(real.mean())

    return np.array(gist_descriptor)


def compute_self_similarity(image):
    """
    Compute self-similarity descriptor for an image.
    Args:
        image (np.ndarray): Input image.
    Returns:
        np.ndarray: Self-similarity matrix.
    """
    patches = extract_patches_2d(image, (16, 16), max_patches=100)
    similarity_matrix = np.zeros((len(patches), len(patches)))

    for i, patch_a in enumerate(patches):
        for j, patch_b in enumerate(patches):
            similarity_matrix[i, j] = match_template(patch_a, patch_b)

    return similarity_matrix.flatten()


def combined_features(image_paths, levels=3):
    """
    Computes combined features (BoW, GIST, Self-Similarity) for a list of images.

    Args:
        image_paths (list): List of image file paths.
        vocab (np.ndarray): Pre-trained vocabulary for BoW.
        levels (int): Number of levels in the spatial pyramid for BoW.

    Returns:
        np.ndarray: Combined feature matrix (n x d, where d = sum of individual feature dimensions).
    """
    vocab = np.load('vocab.npy')
    print('Loaded vocabulary from file.')

    cells_per_block = (4, 4)
    pixels_per_cell = (8, 8)
    feature_dim = cells_per_block[0] * cells_per_block[1] * 9

    combined_features = []

    for image_path in image_paths:
        img = imread(image_path)

        # Extract Multi-Scale HOG
        pyramid = generate_gaussian_pyramid(img, num_levels=3)
        hog_features = np.vstack([
            hog(
                level,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                feature_vector=True
            ).reshape(-1, feature_dim)
            for level in pyramid
        ])

        # Compute Spatial Pyramid BoW
        bow_features = spatial_pyramid_bow(hog_features, vocab, levels=levels)

        # GIST Descriptor
        gist_features = compute_gist(img)

        # Self-Similarity Descriptor
        self_similarity_features = compute_self_similarity(img)

        # Combine features
        combined_feature = np.hstack([bow_features, gist_features, self_similarity_features])
        combined_features.append(combined_feature)

    return np.array(combined_features)


def svm_classify(train_image_feats, train_labels, test_image_feats):
    """
    Classifies test images using a multi-class SVM classifier.

    Args:
        train_image_feats (np.ndarray): n x d matrix of training image features.
        train_labels (list): n x 1 list of training labels (strings).
        test_image_feats (np.ndarray): m x d matrix of test image features.

    Returns:
        np.ndarray: m x 1 array of predicted labels for test images.
    """
    # Initialize Linear SVM with default settings and multi-class support
    l_svc = LinearSVC(random_state=0, tol=1e-5)

    # Train the model on training features and labels
    l_svc.fit(train_image_feats, train_labels)

    # Predict labels for test features
    predictions = l_svc.predict(test_image_feats)

    return predictions


def svm_classify_rbf(train_image_feats, train_labels, test_image_feats, C=10, gamma=0.1):
    """
    Classifies test images using an SVM with an RBF kernel.

    Args:
        train_image_feats (np.ndarray): n x d matrix of training image features.
        train_labels (list): n x 1 list of training labels (strings).
        test_image_feats (np.ndarray): m x d matrix of test image features.
        C (float): Regularization parameter. Default is 1.0.
        gamma (str or float): Kernel coefficient. Default is 'scale'.

    Returns:
        np.ndarray: m x 1 array of predicted labels for test images.
    """
    # Initialize SVM with RBF kernel
    rbf_svm = SVC(kernel='rbf', C=C, gamma=gamma, random_state=0)

    # Train the model on training features and labels
    rbf_svm.fit(train_image_feats, train_labels)

    # Predict labels for test features
    predictions = rbf_svm.predict(test_image_feats)

    return predictions


def svm_classify_rbf_with_grid_search(train_image_feats, train_labels, test_image_feats, param_grid=None):
    """
    Classifies test images using an SVM with RBF kernel and tunes `C` and `gamma` via Grid Search.

    Args:
        train_image_feats (np.ndarray): n x d matrix of training image features.
        train_labels (list): n x 1 list of training labels (strings).
        test_image_feats (np.ndarray): m x d matrix of test image features.
        param_grid (dict): Parameter grid for Grid Search. Defaults to {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}.

    Returns:
        dict: Best parameters and predicted labels for test images.
    """
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1]
        }

    # Perform Grid Search
    print("Performing Grid Search for SVM (RBF)...")
    grid_search = GridSearchCV(SVC(kernel='rbf', random_state=0), param_grid, cv=3, scoring='accuracy', verbose=2)
    grid_search.fit(train_image_feats, train_labels)

    # Best model and parameters
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Predict on test set
    predictions = best_model.predict(test_image_feats)

    print(f"Best Parameters: {best_params}")
    return {'best_params': best_params, 'predictions': predictions}


def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, k=5):
    """
    Classifies test images using k-Nearest Neighbors.

    Args:
        train_image_feats (np.ndarray): n x d array of training image features.
        train_labels (list): List of n training labels (strings).
        test_image_feats (np.ndarray): m x d array of test image features.
        k (int): Number of nearest neighbors to consider (default=5).

    Returns:
        list: Predicted labels for each test image.
    """
    # Compute pairwise distances between test and training features
    distances = cdist(test_image_feats, train_image_feats, metric='euclidean')

    # Find the indices of the k nearest neighbors for each test image
    k_nearest_indices = np.argsort(distances, axis=1)[:, :k]

    # Predict labels based on the mode of the k-nearest neighbors' labels
    predictions = []
    for neighbors in k_nearest_indices:
        k_labels = [train_labels[i] for i in neighbors]  # Get the labels of the k-nearest neighbors
        most_common_label = mode(k_labels, keepdims=True).mode[0]  # Find the most common label
        predictions.append(most_common_label)

    return predictions


def get_image_paths(data_path, categories, num_train_per_cat):
    """
    Returns file paths and labels for training and test images.

    Args:
        data_path (str): Path to the dataset directory.
        categories (list): List of category names (strings).
        num_train_per_cat (int): Number of images per category for training and testing.

    Returns:
        tuple: (train_image_paths, test_image_paths, train_labels, test_labels)
            - train_image_paths (list): List of training image file paths.
            - test_image_paths (list): List of test image file paths.
            - train_labels (list): List of category labels for training images.
            - test_labels (list): List of category labels for test images.
    """
    train_image_paths, test_image_paths = [], []
    train_labels, test_labels = [], []

    for cat in categories:
        # Collect training images and labels
        train_images = glob.glob(os.path.join(data_path, 'train', cat, '*.jpg'))[:num_train_per_cat]
        train_image_paths.extend(train_images)
        train_labels.extend([cat] * len(train_images))

        # Collect test images and labels
        test_images = glob.glob(os.path.join(data_path, 'test', cat, '*.jpg'))[:num_train_per_cat]
        test_image_paths.extend(test_images)
        test_labels.extend([cat] * len(test_images))

    return train_image_paths, test_image_paths, train_labels, test_labels


def create_results_webpage(train_image_paths, test_image_paths,
	train_labels, test_labels,
	categories, abbr_categories, predicted_categories):

	'''
	This function creates a webpage (html and images) visualizing the
	classiffication results. This webpage will contain:
	 (1) A confusion matrix plot
	 (2) A table with one row per category, with 4 columns - training
		 examples, true positives, false positives, and false negatives.
	False positives are instances claimed as that category but belonging to
	another category, e.g. in the 'forest' row an image that was classified
	as 'forest' but is actually 'mountain'. This same image would be
	considered a false negative in the 'mountain' row, because it should have
	been claimed by the 'mountain' classifier but was not.
	This webpage is similar to the one created for the SUN database in
	2010: http://people.csail.mit.edu/jxiao/SUN/classification397.html
	'''

	print('Creating results_webpage/index.html, thumbnails, and confusion matrix.')

	# Number of examples of training examples, true positives, false positives,
	# and false negatives. Thus the table will be num_samples * 4 images wide
	# (unless there aren't enough images)
	num_samples = 2
	thumbnail_height = 75 #pixels
	num_categories = len(categories)

	# Convert everything over to numpy arrays
	categories = np.array(categories)
	predicted_categories = np.array(predicted_categories)
	train_labels = np.array(train_labels)
	test_labels = np.array(test_labels)

	# Delete the old thumbnails, if there are any
	files = glob.glob('results_webpage/thumbnails/*.jpg')
	for f in files:
		os.remove(f)

	if not os.path.isdir('results_webpage'):
		print('Making results_webpage directory.')
		os.mkdir('results_webpage')
	if not os.path.isdir('results_webpage/thumbnails'):
		print('Making thumbnails directory.')
		os.mkdir('results_webpage/thumbnails')

	### Create And Save Confusion Matrix ###
	# Based on the predicted category for each test case, we will now build a
	# confusion matrix. Entry (i,j) in this matrix well be the proportion of
	# times a test image of ground truth category i was predicted to be
	# category j. An identity matrix is the ideal case. You should expect
	# roughly 50-95% along the diagonal depending on your features,
	# classifiers, and particular categories. For example, suburb is very easy
	# to recognize.
	with open('results_webpage/index.html', 'w+') as f:

		# Initialize the matrix
		confusion_matrix = np.zeros((num_categories, num_categories))

		# Iterate over predicted results (this is like, several hundred items long)
		for i,cat in enumerate(predicted_categories):
			# Find the row and column corresponding to the label of this entry
			# The row is the ground truth label and the column is the found label
			row = np.argwhere(categories == test_labels[i])[0][0]
			column = np.argwhere(categories == predicted_categories[i])[0][0]

			# Add 1 to the matrix for that row/col
			# This way we build up a histogram from our labeled data
			confusion_matrix[row][column] += 1;

		# If the number of training examples and test cases are not equal, this
		# statement will be invalid!
		# TODO: That's an old comment left over from the matlab code that I don't
		# think still applies
		num_test_per_cat = len(test_labels) / num_categories
		confusion_matrix = confusion_matrix / float(num_test_per_cat)
		accuracy = np.mean(np.diag(confusion_matrix))

		print('Accuracy (mean of diagonal of confusion matrix) is {:2.3%}'.format(accuracy))

		# plasma is the most easily-interpreted color map I've found so far
		plt.imshow(confusion_matrix, cmap='plasma', interpolation='nearest')

		# We put the shortened labels (e.g. "sub" for "suburb") on the x axis
		locs, labels = plt.xticks()
		plt.xticks(np.arange(num_categories), abbr_categories)

		# Full labels go on y
		locs, labels = plt.yticks()
		plt.yticks(np.arange(num_categories), categories)

		# Save the result
		plt.savefig('results_webpage/confusion_matrix.png', bbox_inches='tight')

		## Create webpage header
		f.write('<!DOCTYPE html>\n');
		f.write('<html>\n');
		f.write('<head>\n');
		f.write('<link href=''http://fonts.googleapis.com/css?family=Nunito:300|Crimson+Text|Droid+Sans+Mono'' rel=''stylesheet'' type=''text/css''>\n');
		f.write('<style type="text/css">\n');

		f.write('body {\n');
		f.write('  margin: 0px;\n');
		f.write('  width: 100%;\n');
		f.write('  font-family: ''Crimson Text'', serif;\n');
		f.write('  background: #fcfcfc;\n');
		f.write('}\n');
		f.write('table td {\n');
		f.write('  text-align: center;\n');
		f.write('  vertical-align: middle;\n');
		f.write('}\n');
		f.write('h1 {\n');
		f.write('  font-family: ''Nunito'', sans-serif;\n');
		f.write('  font-weight: normal;\n');
		f.write('  font-size: 28px;\n');
		f.write('  margin: 25px 0px 0px 0px;\n');
		f.write('  text-transform: lowercase;\n');
		f.write('}\n');
		f.write('.container {\n');
		f.write('  margin: 0px auto 0px auto;\n');
		f.write('  width: 1160px;\n');
		f.write('}\n');

		f.write('</style>\n');
		f.write('</head>\n');
		f.write('<body>\n\n');

		f.write('<div class="container">\n\n\n');
		f.write('<center>\n');
		f.write('<h1>Scene classification results visualization</h1>\n');
		f.write('<img src="confusion_matrix.png">\n\n');
		f.write('<br>\n');
		f.write('Accuracy (mean of diagonal of confusion matrix) is %2.3f\n' % (accuracy));
		f.write('<p>\n\n');

		## Create results table
		f.write('<table border=0 cellpadding=4 cellspacing=1>\n');
		f.write('<tr>\n');
		f.write('<th>Category name</th>\n');
		f.write('<th>Accuracy</th>\n');
		f.write('<th colspan=%d>Sample training images</th>\n' % num_samples);
		f.write('<th colspan=%d>Sample true positives</th>\n' % num_samples);
		f.write('<th colspan=%d>False positives with true label</th>\n' % num_samples);
		f.write('<th colspan=%d>False negatives with wrong predicted label</th>\n' % num_samples);
		f.write('</tr>\n');

		for i,cat in enumerate(categories):
			f.write('<tr>\n');

			f.write('<td>'); #category name
			f.write('%s' % cat);
			f.write('</td>\n');

			f.write('<td>'); # category accuracy
			f.write('%.3f' % confusion_matrix[i][i]);
			f.write('</td>\n');

			# Collect num_samples random paths to images of each type. Training examples.
			train_examples = np.take(train_image_paths, np.argwhere(train_labels == cat))

			# True positives. There might not be enough of these if the classifier is bad
			true_positives = np.take(test_image_paths, np.argwhere(np.logical_and(test_labels == cat, predicted_categories == cat)))

			# False positives. There might not be enough of them if the classifier is good
			false_positive_inds = np.argwhere(np.logical_and(np.invert(cat == test_labels), cat == predicted_categories))
			false_positives = np.take(test_image_paths, false_positive_inds)
			false_positive_labels = np.take(test_labels, false_positive_inds)

			# False negatives. There might not be enough of them if the classifier is good
			false_negative_inds = np.argwhere(np.logical_and(cat == test_labels, np.invert(cat == predicted_categories)))
			false_negatives = np.take(test_image_paths, false_negative_inds)
			false_negative_labels = np.take(predicted_categories, false_negative_inds)

			# Randomize each list of files
			np.random.shuffle(train_examples)
			np.random.shuffle(true_positives)

			# HACK: Well, sort of a hack. We need to shuffle the false_positives
			# and their labels in the same exact order, so we get the RNG state,
			# save it, shuffle, restore, then shuffle the other list so that they
			# shuffle in tandem.
			rng_state = np.random.get_state()
			np.random.shuffle(false_positives)
			np.random.set_state(rng_state)
			np.random.shuffle(false_positive_labels)

			rng_state = np.random.get_state()
			np.random.shuffle(false_negatives)
			np.random.set_state(rng_state)
			np.random.shuffle(false_negative_labels)

			# Truncate each list to be at most num_samples long
			train_examples  = train_examples[0:min(len(train_examples), num_samples)]
			true_positives  = true_positives[0:min(len(true_positives), num_samples)]
			false_positives = false_positives[0:min(len(false_positives), num_samples)]
			false_positive_labels = false_positive_labels[0:min(len(false_positive_labels),num_samples)]
			false_negatives = false_negatives[0:min(len(false_negatives),num_samples)]
			false_negative_labels = false_negative_labels[0:min(len(false_negative_labels),num_samples)]

			# Sample training images
			# Create and save all of the thumbnails
			for j in range(num_samples):
				if j + 1 <= len(train_examples):
					thisExample = train_examples[j][0]
					tmp = imread(thisExample)
					height, width = rescale(tmp.shape, thumbnail_height)
					tmp = resize(tmp, (height, width),
						anti_aliasing=True, mode='wrap')

					name = os.path.basename(thisExample)
					tmp_uint8 = (tmp * 255).astype(np.uint8)
					imsave('results_webpage/thumbnails/' + cat + '_' + name, tmp_uint8, quality=100)
					f.write('<td bgcolor=LightBlue>')
					f.write('<img src="%s" width=%d height=%d>' % ('thumbnails/' + cat + '_' + name, width, height))
					f.write('</td>\n')
				else:
					f.write('<td bgcolor=LightBlue>')
					f.write('</td>\n')

			for j in range(num_samples):
				if j + 1 <= len(true_positives):
					thisExample = true_positives[j][0]
					tmp = imread(thisExample)
					height, width = rescale(tmp.shape, thumbnail_height)
					tmp = resize(tmp, (height, width),
						anti_aliasing=True, mode='wrap')

					name = os.path.basename(thisExample)
					tmp_uint8 = (tmp * 255).astype(np.uint8)
					imsave('results_webpage/thumbnails/' + cat + '_' + name, tmp_uint8, quality=100)
					f.write('<td bgcolor=LightGreen>');
					f.write('<img src="%s" width=%d height=%d>' % ('thumbnails/' + cat + '_' + name, width, height))
					f.write('</td>\n');
				else:
					f.write('<td bgcolor=LightGreen>');
					f.write('</td>\n');

			for j in range(num_samples):
				if j + 1 <= len(false_positives):
					thisExample = false_positives[j][0]
					tmp = imread(thisExample)
					height, width = rescale(tmp.shape, thumbnail_height)
					tmp = resize(tmp, (height, width),
						anti_aliasing=True, mode='wrap')

					name = os.path.basename(thisExample)
					tmp_uint8 = (tmp * 255).astype(np.uint8)
					imsave('results_webpage/thumbnails/' + cat + '_' + name, tmp_uint8, quality=100)
					f.write('<td bgcolor=LightCoral>');
					f.write('<img src="%s" width=%d height=%d>' % ('thumbnails/' + cat + '_' + name, width, height))
					f.write('<br><small>%s</small>' % false_positive_labels[j][0]);
					f.write('</td>\n');
				else:
					f.write('<td bgcolor=LightCoral>');
					f.write('</td>\n');

			for j in range(num_samples):
				if j + 1 <= len(false_negatives):
					thisExample = false_negatives[j][0]
					tmp = imread(thisExample)
					height, width = rescale(tmp.shape, thumbnail_height)
					tmp = resize(tmp, (height, width),
						anti_aliasing=True, mode='wrap')

					name = os.path.basename(thisExample)
					tmp_uint8 = (tmp * 255).astype(np.uint8)
					imsave('results_webpage/thumbnails/' + cat + '_' + name, tmp_uint8, quality=100)
					f.write('<td bgcolor=#FFBB55>');
					f.write('<img src="%s" width=%d height=%d>' % ('thumbnails/' + cat + '_' + name, width, height));
					f.write('<br><small>%s</small>' % false_negative_labels[j][0]);
					f.write('</td>\n');
				else:
					f.write('<td bgcolor=#FFBB55>');
					f.write('</td>\n');

			f.write('</tr>\n');

		f.write('<tr>\n');
		f.write('<th>Category name</th>\n');
		f.write('<th>Accuracy</th>\n');
		f.write('<th colspan=%d>Sample training images</th>\n' % num_samples);
		f.write('<th colspan=%d>Sample true positives</th>\n' % num_samples);
		f.write('<th colspan=%d>False positives with true label</th>\n' % num_samples);
		f.write('<th colspan=%d>False negatives with wrong predicted label</th>\n' % num_samples);
		f.write('</tr>\n');

		f.write('</table>\n');
		f.write('</center>\n\n\n');
		f.write('</div>\n');

		## Create end of web page
		f.write('</body>\n');
		f.write('</html>\n');

	print('Wrote results page to results_webpage/index.html.')

def rescale(dims, thumbnail_height):
	height = dims[1]
	factor = thumbnail_height / height
	left = int(round(dims[0] * factor))
	right = int(round(dims[1] * factor))
	return (left, right)
