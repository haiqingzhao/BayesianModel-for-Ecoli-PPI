#@title Define Discretized Naive Bayes (Naive, Simple and Independent)
# for PrePPI SM PPI-PSD, or other independent features
# Using Laplace Smoothing for Likelihood Estimation

from collections import Counter
import numpy as np

class DiscretizedNaiveBayes_IndependentFeatures:
    def __init__(self):
        self.class_priors = {}  # Prior probabilities of classes
        self.feature_bin_edges = []     # List of automatically generated bin edges for each feature
        self.n_bins = [] # List of bin numbers for each feature
        self.class_feature_bins_counts = {} # Counts for each bin of each class
        self.class_feature_bins_smoothedlikelihood = {} # Smoothedlikelihood for each bin of each class
        self.class_feature_bins_smoothedlikelihood_lr = [] # Smoothedlikelihood ratio (LR) for each bin

    def discretize_features(self, X):
        # Discretize continuous features into bins
        num_samples, num_features = X.shape

        X_discretized = np.zeros_like(X, dtype=int)
        for feature_index in range(num_features):
            feature_values = X[:, feature_index]
            if len(self.n_bins) == num_features:
                X_discretized[:, feature_index] = np.digitize(feature_values, self.feature_bin_edges[feature_index][1:-1])
            else:
                bin_edges = np.histogram_bin_edges(feature_values, bins='doane') #self.num_bins) #'doane'
                self.feature_bin_edges.append(bin_edges)
                self.n_bins.append(len(bin_edges)-1)
                X_discretized[:, feature_index] = np.digitize(feature_values, bin_edges[1:-1])
        #print('self.feature_bin_edges:',self.feature_bin_edges)
        #print('self.n_bins for each feature in []:',self.n_bins)

        return X_discretized

    def fit(self, X, y):
        # Discretize continuous features
        X_discretized = self.discretize_features(X)
        num_samples, num_features = X.shape
        # Calculate class priors
        unique_classes, class_counts = np.unique(y, return_counts=True)
        print('unique_classes',unique_classes)
        for i, class_label in enumerate(unique_classes):
            self.class_priors[class_label] = class_counts[i] / num_samples
        #print("self.class_priors:",self.class_priors)
        # Calculate the distribution of discretized features for each class
        for class_label in unique_classes:
            class_indices = np.where(y == class_label)
            binned_samples = X_discretized[class_indices]

            # Initialize the bins for this class
            self.class_feature_bins_counts[class_label] = []
            self.class_feature_bins_smoothedlikelihood[class_label] =[]

            for feature_index in range(num_features):
                # Initialize bins for this feature of this class
                feature_bins_counts = np.zeros(self.n_bins[feature_index],dtype=int)

                # Update counts of bins for this feature of this class
                bins_counter = Counter(binned_samples[:,feature_index])
                #print("Class",class_label, "Feature",feature_index,"bins_counter:",bins_counter)
                #print("Class",class_label, "Feature ",feature_index,"sorted(counter):",sorted(bins_counter.items()))
                np.put(feature_bins_counts, list(bins_counter.keys()), list(bins_counter.values()))
                print(bins_counter.keys(),bins_counter.values())

                # Smoothing
                alpha = 1  # Smoothing parameter
                #posterior = (feature_bins_counts ) / (len(binned_samples) )
                smoothed_joint_probability = (feature_bins_counts + alpha) / (num_samples + alpha * (self.n_bins[feature_index]))
                smoothed_likelihood = smoothed_joint_probability / self.class_priors[class_label]
                # Log-probability
                #smoothed_likelihood = np.log(smoothed_posterior) + np.log(self.class_priors[class_label])

                self.class_feature_bins_counts[class_label].append(feature_bins_counts)
                self.class_feature_bins_smoothedlikelihood[class_label].append(smoothed_likelihood)

        for feature_index in range(num_features):
            # Assume labels being 0/1, F/T
            lrs = self.class_feature_bins_smoothedlikelihood[1][feature_index] / self.class_feature_bins_smoothedlikelihood[0][feature_index]
            lrs = np.round(lrs, 4).astype(np.float32)

            # For log-probability
            #lrs = self.class_feature_bins_smoothedlikelihood[0][feature_index] / self.class_feature_bins_smoothedlikelihood[1][feature_index]
            self.class_feature_bins_smoothedlikelihood_lr.append(lrs)
        #print("self.class_feature_bins_smoothedlikelihood_lr:",self.class_feature_bins_smoothedlikelihood_lr)

    def predict(self, X):
        X_discretized = self.discretize_features(X)
        predict_lrs = np.zeros(X.shape[0])

        for index, sample_bin_set in enumerate(X_discretized):
            feature_lrs = []
            for feature_index, bin_id in enumerate(sample_bin_set):
                feature_lrs.append(self.class_feature_bins_smoothedlikelihood_lr[feature_index][bin_id])
            predict_lrs[index] = np.product(feature_lrs)

        predict_lrs = np.round(predict_lrs, 4).astype(np.float32)
        return predict_lrs

#@title Discretized Naive Bayes with Fully-connected Bins of Each Feature
# for PrePPI SM PPI-interface features

class DiscretizedNaiveBayes_JointBins:
    def __init__(self):
        self.class_priors = {}  # Prior probabilities of classes
        self.feature_bin_edges = []     # List of automatically generated bin edges for each feature
        self.n_bins = [] # List of bin numbers for each feature
        self.class_joint_bins_counts = {}  # Joint (or Fully-connected) bins for each class
        self.class_joint_bins_smoothedlikelihood = {}  # Joint (or Fully-connected) bins for each class
        self.class_joint_bins_smoothedlikelihood_lr = np.zeros(tuple(self.n_bins),dtype=int) # Smoothed likelihood ratio for each bin

    def discretize_features(self, X):
        # Discretize continuous features into bins
        num_samples, num_features = X.shape
        X_discretized = np.zeros_like(X, dtype=int)

        for feature_index in range(num_features):
            feature_values = X[:, feature_index]
            if len(self.n_bins) == num_features:
                X_discretized[:, feature_index] = np.digitize(feature_values, self.feature_bin_edges[feature_index][1:-1])
            else:
                bin_edges = np.histogram_bin_edges(feature_values, bins='doane') #self.num_bins) #'doane'
                self.feature_bin_edges.append(bin_edges)
                self.n_bins.append(len(bin_edges))
                X_discretized[:, feature_index] = np.digitize(feature_values, bin_edges[1:-1])
        #print('self.feature_bin_edges:',self.feature_bin_edges)
        #print('self.n_bins:',self.n_bins)
        return X_discretized

    def fit(self, X, y):
        # Discretize continuous features
        X_discretized = self.discretize_features(X)
        num_samples, num_features = X.shape
        # Calculate class priors
        unique_classes, class_counts = np.unique(y, return_counts=True)
        #print('unique_classes',unique_classes)
        for i, class_label in enumerate(unique_classes):
            self.class_priors[class_label] = class_counts[i] / num_samples

        # Calculate the distribution of discretized features for each class
        for class_label in unique_classes:
            class_indices = np.where(y == class_label)
            binned_samples = X_discretized[class_indices]

            # Initialize the joint bins for this class
            self.class_joint_bins_counts[class_label] = np.zeros(tuple(self.n_bins),dtype=int)

            # Count the joint distribution for combinations of bins from each feature for this class #across all features
            for sample_bin_ids in binned_samples:
                self.class_joint_bins_counts[class_label][tuple(sample_bin_ids)] += 1

            total_counts = num_samples #np.sum(self.class_joint_bins_counts[class_label])

            # Initialize Laplace smoothing parameters
            alpha = 1.0  # Smoothing parameter
            n_possible_combinations = np.prod(self.n_bins)
            # Calculate Laplace-smoothed likelihood for the sample
            # Using joint counts for all features in the sample

            smoothed_joint_probability = (self.class_joint_bins_counts[class_label] + alpha) / (total_counts + alpha * (n_possible_combinations))
            #smoothed_posterior = (feature_bins_counts + alpha) / (len(binned_samples) + alpha * (self.n_bins[feature_index]))
            self.class_joint_bins_smoothedlikelihood[class_label] = smoothed_joint_probability / self.class_priors[class_label]

        # Calculate the smoothed likelihood ratio (LR) for each bin
        self.class_joint_bins_smoothedlikelihood_lr = self.class_joint_bins_smoothedlikelihood[unique_classes[1]]/self.class_joint_bins_smoothedlikelihood[unique_classes[0]]

    def predict(self, X):
        X_discretized = self.discretize_features(X)
        predict_lrs = np.zeros(X.shape[0])
        #print(self.class_joint_bins_smoothedlikelihood_lr.shape)
        for index, sample_bin_set in enumerate(X_discretized):
            lr = self.class_joint_bins_smoothedlikelihood_lr[tuple(sample_bin_set)]
            predict_lrs[index] = lr
        return predict_lrs

""";
""";

""";
""";
