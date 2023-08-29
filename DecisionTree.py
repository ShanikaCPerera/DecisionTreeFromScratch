''' DecisionTree.py: Create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.
    __author__      = Shanika Perera'''

import numpy as np
from collections import Counter
import graphviz

''' Class that represents a node in the Decision tree'''
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, best_gain=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.best_gain = best_gain

    def is_leaf_node(self):
        return self.value is not None
    
    def get_feature(self):
        return self.feature
    
    def get_left(self):
        return self.left
    
    def get_right(self):
        return self.right
    
    def get_value(self):
        return self.value
    
    def print_node(self, feature_names):
        if self.is_leaf_node():
            print(self.get_value())
        else:
            print(feature_names[self.get_feature()])

''' Class that represents the Decision tree'''
class DecisionTree:
    
    def __init__(self, min_samples_split=2, max_depth=1000, n_features=None):
        self.min_samples_split = min_samples_split #
        self.max_depth = max_depth 
        self.n_features = n_features #number of features
        self.root = None
        self.tree = []
        self.gain_dict = {}
        
    def get_root(self):
        return self.root

    def fit(self, X, y):
        ''' fit a model while creating a desicion tree
            - X is a 2D-array with predictor variables
            - y is a 2D-array with target variable'''
        # making sure the number of nodes do not exceed the number of features
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        #starting growing the tree from the root
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        ''' create a desicion tree
            - X is a 2D-array with predictor variables
            - y is a 2D-array with target variable
            - depth is the depth of the decision tree'''
        #getting number of samples and number of features
        n_samples, n_feats = X.shape
        #getting number of labels
        n_labels = len(np.unique(y))

        # check the stopping criteria
        # if any of the below are true we won't grow the tree instaed we create a new node(leaf node) and return it
        # 1. current depth is greater than or equal to the max depth
        # 2. n_labels == 1 means, all samples have one lable - you dont have to split anymore - its a leaf node
        # 3. number of samples at this point  is smaller than min_sample_split - measn when the current n_samples is 1 or 0
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        #choosing a random set of features to find the best split among them
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False) 

        # find the best split (we get here if we did not get caught in the stopping critieria)
        best_feature, best_thresh, best_gain = self._best_split(X, y, feat_idxs, depth)

        # create child nodes/trees
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right, best_gain)

    def _best_split(self, X, y, feature_idxs, depth):
        ''' find the feature and the threshold which reults the best splt among given features
            - X is a 2D-array with predictor variables
            - y is a 2D-array with target variable
            - feature_idxs is the list of features considered for the split'''
        best_gain = -1
        split_idx, split_threshold = None, None
        gain_results_feat = []

        for feat_idx in feature_idxs:
            gain_results_thresh = []
           
            X_column = X[:, feat_idx]
            #thresholds means unique values in this column to be considers for splitting - remember X columns are considered integer in DT
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
                    
                gain_results_thresh.append({thr:gain})    
                
            gain_results_feat.append({feat_idx:gain_results_thresh})
        
        self.gain_dict[depth] = gain_results_feat

        return split_idx, split_threshold, best_gain

    # threshold is the X value being considered for splitting - remember X columns are considered integer in DT
    def _information_gain(self, y, X_column, threshold):
        ''' find the information gain for a split by the considered feature
            - y is a 2D-array with target variables
            - X_column is a 2D-array with values for the considered feature
            - threshold is the considered dividing value/threshold'''
        # parent entropy = entropy before split
        entropy_before_split = self._entropy(y)

        # get the tuple indexes that goes to left and right branch
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # calculate the Remainder for the feature = weighted average entropy of the left ad right branch
        n = len(y)
        n_samples_left, n_samples_right = len(left_idxs), len(right_idxs)
        entropy_left, entropy_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        # remainder for this feature = entropy after split 
        remainder = (n_samples_left / n) * entropy_left + (n_samples_right / n) * entropy_right

        # calculate the Inforation Gain
        information_gain = entropy_before_split - remainder
        return information_gain

    def _split(self, X_column, split_thresh):
        ''' do the splitting after knowing the best split, return the indexes of the tuples that go into left and righ tree
             tuples where X-column value is less than or equal to split_thresh value goes to left and others go to right
            - X_column is a 2D-array with values for the considered feature
            - split_thresh is the value/threshold used to do the split'''
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        ''' calculate the entropy
            entropy = -sum[P(x).logp(x)]
            - y is a 2D-array with target variables'''
        hist = np.bincount(y)
        all_p = hist / len(y)
        return -np.sum([p * np.log(p) for p in all_p if p > 0])

    def _most_common_label(self, y):
        ''' get the most probable label
            - y is a 2D-array with target variables'''
        counter = Counter(y)
        # getting the most common first value (most common tuple and the first information)
        value = counter.most_common(1)[0][0]
        return value;

    # sending the X of test set
    def predict(self, X):
        ''' predict teh traget value for a given set of unseen records
            - X is a 2D-array with predictor variabes of unseen data'''
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        ''' travers the tree
            - x is a tuple with predictor variable values of unseen data
            - node is the starting node'''
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)
    
    def print_information_gain_results(self, feature_names):
        ''' public method to print the list of resuls from Information Gain 
            calculataions the tree for visualization
            - feature_names is a list of feature names in the same order'''
        for depth, feat_list in self.gain_dict.items():
            print('Depth: ' + str(depth) + ' - Split: ' + str(depth+1))
            for item in feat_list:
                for feat, thr_list in item.items():
                    print('\tFeature: ' + feature_names[feat])
                    for thr in thr_list:
                        for key, value in thr.items():
                            print ('\t\tThreshold: ' + str(key) + ' Gain: ' + str(round(value, 3)))
                            
    
    def _draw_tree(self, parent, parent_node_name, edge_name, node, feature_names, graph):
        ''' private method to draw the tree for visualization
            - parent is the parent node of the current node
            - parent_node_name is the name of the parent node of the current node
            - edge_name is the name of the edge draw between the current Node and its parent
            - node is the current Node
            - feature_names is a list of feature names in the same order
            - graph is a Digraph form graphviz library'''
        if (node != None):
            if node.is_leaf_node():
                node_name = feature_names[parent.feature] + '_' + str(node.value)
                graph.node(node_name, 'No' if node.value == 0 else 'Yes' )
                graph.edge(parent_node_name, node_name, label=edge_name)
            else:
                node_name = feature_names[node.feature] + '\nGain = ' + str(round(node.best_gain,3))
                graph.node(node_name)

                if(parent != None):
                    graph.edge(parent_node_name, node_name, label=edge_name)

            self._draw_tree(node, node_name, 'No', node.left, feature_names, graph)

            self._draw_tree(node, node_name, 'Yes', node.right, feature_names, graph)
    
    def visualize_tree(self, feature_names):
        ''' public method to draw the tree for visualization
            - feature_names is a list of feature names in the same order'''
        graph = graphviz.Digraph('finite_state_machine', filename='dt_visualization.gv')
        graph.attr(rankdir='LR', size='8,5')
        graph.attr('node', shape='square')
    
        self._draw_tree(None, None, None, self.root, feature_names, graph)
        return graph
    



