import os
import numpy as np
import argparse
from utils import (
    get_tiny_images, build_vocabulary, get_bags_of_words, combined_features,
    svm_classify, nearest_neighbor_classify, get_image_paths,
    svm_classify_rbf, svm_classify_rbf_with_grid_search,
    build_vocabulary_with_gmm, get_bags_of_words_with_gmm,
    create_results_webpage
)

def scene_recognition(feature='tiny_image', classifier='nearest_neighbor'):
    """
    Performs scene recognition by extracting features from images and classifying them
    into predefined categories using specified feature extraction and classification methods.

    Args:
        feature (str): Feature extraction method ('tiny_image' or 'bag_of_words').
        classifier (str): Classification method ('nearest_neighbor' or 'support_vector_machine').
    """
    # Parameters and Paths
    data_path = './data/'
    categories = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office', 'Industrial',
                  'Suburb', 'InsideCity', 'TallBuilding', 'Street', 'Highway', 'OpenCountry',
                  'Coast', 'Mountain', 'Forest']
    abbr_categories = ['Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub', 'Cty', 'Bld', 'St',
                       'HW', 'OC', 'Cst', 'Mnt', 'For']
    num_train_per_cat = 100

    # Load image paths and labels
    print('Loading image paths and labels...')
    train_paths, test_paths, train_labels, test_labels = get_image_paths(data_path, categories, num_train_per_cat)

    # Feature Extraction
    print(f'Extracting features using: {feature}')
    if feature == 'tiny_image':
        train_feats = get_tiny_images(train_paths)
        test_feats = get_tiny_images(test_paths)
    elif feature == 'bag_of_words':
        vocab_file = 'vocab.npy'
        if not os.path.isfile(vocab_file):
            print('Building vocabulary...')
            vocab = build_vocabulary(train_paths, vocab_size=500)
            np.save(vocab_file, vocab)
        train_feats = get_bags_of_words(train_paths)
        test_feats = get_bags_of_words(test_paths)
    elif feature == 'bow_gmm':
        # Build Vocabulary using GMM
        gmm_file = 'vocab_gmm.npy'
        if not os.path.isfile(vocab_file):
            print('Building GMM vocabulary...')
            gmm_size=200
            gmm = build_vocabulary_with_gmm(train_paths, n_components=gmm_size)
            np.save(gmm_file, gmm)
        train_feats = get_bags_of_words_with_gmm(train_paths, levels=3)
        test_feats = get_bags_of_words_with_gmm(test_paths, levels=3)
    elif feature == 'combined_features':
        vocab_file = 'vocab.npy'
        if not os.path.isfile(vocab_file):
            print('Building vocabulary...')
            vocab = build_vocabulary(train_paths, vocab_size=500)
            np.save(vocab_file, vocab)
        train_feats = combined_features(train_paths)
        test_feats = combined_features(test_paths)
    else:
        raise ValueError(f'Unsupported feature type: {feature}')

    # Classification
    print(f'Classifying using: {classifier}')
    if classifier == 'NN':
        predictions = nearest_neighbor_classify(train_feats, train_labels, test_feats)
    elif classifier == 'SVM':
        predictions = svm_classify(train_feats, train_labels, test_feats)
    elif classifier == 'SVM_RBF':
        predictions = svm_classify_rbf(train_feats, train_labels, test_feats)
    elif classifier == 'SVM_RBF_GridSearch':
        results = svm_classify_rbf_with_grid_search(train_feats, train_labels, test_feats)
        predictions = results['predictions']
        print(f"Best Parameters for SVM_RBF: {results['best_params']}")    
    else:
        raise ValueError(f'Unsupported classifier type: {classifier}')

    # Generate Results Webpage
    create_results_webpage(train_paths, test_paths, train_labels, test_labels, categories,
                           abbr_categories, predictions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--feature', default='tiny_image',
                        choices=['tiny_image', 'bag_of_words', 'bow_gmm', 'combined_features'],
                        help='Feature representation method: tiny_image, bag_of_words, bow_gmm or combined_features')
    parser.add_argument('-c', '--classifier', default='NN',
                        choices=['NN', 'SVM', 'SVM_RBF', 'SVM_RBF_GridSearch'],
                        help='Classifier method: NN, SVM, SVM_RBF or SVM_RBF_GridSearch')
    args = parser.parse_args()

    scene_recognition(feature=args.feature, classifier=args.classifier)
