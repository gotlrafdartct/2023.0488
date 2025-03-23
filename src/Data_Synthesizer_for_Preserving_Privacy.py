import numpy as np
import pandas as pd
import math
import os
import json
import datetime
from itertools import combinations, product
from multiprocessing.pool import Pool
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression


# Functions for (DataLearner)GreedyBayes
def mutual_information(data_child, data_parents: pd.DataFrame):
    if data_parents.shape[1] == 1:
        if data_child.dtype == 'float64':
            return mutual_info_regression(data_parents, data_child)[0]
        else:
            data_parents = data_parents.iloc[:, 0]
    else:
        data_parents = data_parents.apply(lambda x: ' '.join(str(i) for i in x.values), axis=1)
        if data_child.dtype == 'float64':
            data_parents, _ = pd.factorize(data_parents)
            data_parents = data_parents.reshape(-1, 1)
            return mutual_info_regression(data_parents, data_child)[0]

    return mutual_info_score(data_child, data_parents)


def worker(paras):
    child, v_set, num_parents, split, dataset = paras
    parents_pair_list = []
    mutual_info_list = []

    if split + num_parents - 1 < len(v_set):
        for other_parents in combinations(v_set[split + 1:], num_parents - 1):
            parents = list(other_parents)
            parents.append(v_set[split])
            parents_pair_list.append((child, parents))
            mi = mutual_information(dataset[child], dataset[parents])
            mutual_info_list.append(mi)

    return parents_pair_list, mutual_info_list


def greedy_bayes(dataset: pd.DataFrame, network_degree: int, root_attribute: str):
    print('================ Constructing Bayesian Network (BN) ================')
    v_set = [root_attribute]
    rest_attributes = set(dataset.columns)
    rest_attributes.remove(root_attribute)
    print('Adding ROOT', str(root_attribute))
    n_set = []

    while rest_attributes:
        parents_pair_list = []
        mutual_info_list = []

        num_parents = min(len(v_set), network_degree)
        tasks = [(child, v_set, num_parents, split, dataset) for child, split in
                 product(rest_attributes, range(len(v_set) - num_parents + 1))]
        with Pool() as pool:
            res_list = pool.map(worker, tasks)

        for res in res_list:
            parents_pair_list += res[0]
            mutual_info_list += res[1]

        idx = mutual_info_list.index(max(mutual_info_list))

        n_set.append(parents_pair_list[idx])
        adding_attribute = parents_pair_list[idx][0]
        v_set.append(adding_attribute)
        rest_attributes.remove(adding_attribute)
        print('Adding attribute', adding_attribute)

    print('========================== BN constructed ==========================')

    return n_set


# Functions for loading/saving list from/into json file
def save_structure(filepath, filename, structure_to_json):
    with open(filepath + filename, 'w') as json_file:
        json.dump(structure_to_json, json_file, ensure_ascii=False)


def load_structure(filepath, filename):
    with open(filepath + filename, "r") as json_file:
        structure_from_json = json.load(json_file)
    return structure_from_json


# Functions for (DataLearner)NoisyCondProb
def laplace_noise_parameter(k, n_attributes, n_tuples, private_param):
    return 2 * (n_attributes - k) / (n_tuples * private_param)


def fit_in_full_space(value, threshold):
    decimal_part, integer_part = math.modf(float(value) / threshold)
    if decimal_part < 0.5:
        return integer_part * threshold
    else:
        return (integer_part + 1) * threshold


def construct_attribute_value_matrix(attributes, data):
    iterables = []
    for attr in attributes:
        iterables.append(data[attr].unique().tolist())

    return iterables


def get_noisy_distribution_of_attributes(attributes, input_original, private_param):
    data = input_original.copy().loc[:, attributes]
    data['count'] = 1
    # Treat the values of selected attributes in each sample row as a whole,
    # and count the total number of rows with the same attribute value combination across the sample
    stats = data.groupby(attributes).sum()
    stats.reset_index(inplace=True)

    # Storing all possible values for each attribute
    iterables = construct_attribute_value_matrix(attributes, data)

    # Creating a dataframe, each row representing a possible combination of values of attributes
    full_space = pd.DataFrame(columns=attributes, data=list(product(*iterables)))
    stats = pd.merge(full_space, stats, how='left')
    stats.fillna(0, inplace=True)

    if private_param:
        k = len(attributes) - 1
        n_rows, n_attributes = input_original.shape
        noise_param = laplace_noise_parameter(k, n_attributes, n_rows, private_param)
        laplace_noises = np.random.laplace(0, scale=noise_param, size=stats.index.size)
        stats['count'] += laplace_noises
        stats.loc[stats['count'] < 0, 'count'] = 0

    return stats


def normalize_given_distribution(frequencies):
    distribution = np.array(frequencies, dtype=float)
    distribution = distribution.clip(0)  # replace negative values with 0
    summation = distribution.sum()
    if summation > 0:
        if np.isinf(summation):
            # Return a normalized version of the infinite components
            # normalize_given_distribution([0, np.inf, 1])
            # → [0., 1.0, 0.]
            return normalize_given_distribution(np.isinf(distribution))
        else:
            return distribution / summation
    else:
        # Return uniform distribution
        # normalize_given_distribution([0, 0, 0])
        # → [0.333, 0.333, 0.333]
        return np.full_like(distribution, 1 / distribution.size)


def construct_noisy_conditional_distributions(bayesian_network, input_original, private_param,
                                              full_space_num_tuples_for_numerical):
    print('================ Constructing Noisy Conditional Distributions (NCD) ================')
    conditional_distributions = {}

    # k (i.e.,BN_degree) representing the number of parent nodes for each feature
    k = len(bayesian_network[-1][1])

    # Classifying features as categorical or numerical
    categorical_attr_list = input_original.columns.tolist()
    numerical_attr_list = []
    for i in input_original.columns:
        if input_original[i].dtype == 'float64':
            numerical_attr_list.append(i)
            categorical_attr_list.remove(i)

    # Preprocessing numerical attributes with too many unique values
    for attr in numerical_attr_list:
        if len(input_original[attr].unique()) > full_space_num_tuples_for_numerical:
            interval = (input_original[attr].max() - input_original[attr].min()) / full_space_num_tuples_for_numerical
            input_original[attr] = input_original[attr].apply(lambda x: fit_in_full_space(x, interval))

    full_attribute_value_matrix = construct_attribute_value_matrix(input_original.columns.values, input_original)

    # Getting the first k+1 attributes for root attribute
    root = bayesian_network[0][1][0]
    k_plus_1_attributes = [root]
    for child, _ in bayesian_network[:k]:
        k_plus_1_attributes.append(child)

    noisy_dist_of_k_plus_1_attributes = get_noisy_distribution_of_attributes(k_plus_1_attributes, input_original,
                                                                             private_param)

    # Generating noisy distribution of root attribute
    root_stats = noisy_dist_of_k_plus_1_attributes.loc[:, [root, 'count']].groupby(root).sum()['count']
    conditional_distributions[root] = normalize_given_distribution(root_stats).tolist()
    print('Completing ROOT ' + root)

    for idx, (child, parents) in enumerate(bayesian_network):
        conditional_distributions[child] = {}

        if idx < k:
            stats = noisy_dist_of_k_plus_1_attributes.copy().loc[:, parents + [child, 'count']]
        else:
            stats = get_noisy_distribution_of_attributes(parents + [child], input_original, private_param)

        stats = pd.DataFrame(stats.loc[:, parents + [child, 'count']].groupby(parents + [child]).sum())

        # Use .loc to match the parent node value combinations
        # and obtain the conditional probability distribution of the child node
        if len(parents) == 1:
            for parent_instance in stats.index.levels[0]:
                dist = normalize_given_distribution(stats.loc[parent_instance]['count']).tolist()
                conditional_distributions[child][str([parent_instance])] = dist
        else:
            for parents_instance in product(*stats.index.levels[:-1]):
                dist = normalize_given_distribution(stats.loc[parents_instance]['count']).tolist()
                conditional_distributions[child][str(list(parents_instance))] = dist
        print('Completing attribute ' + str(idx + 1) + ': ' + child)
    print('========================== NCD constructed ==========================')

    return conditional_distributions, full_attribute_value_matrix


def transform_range_to_list(value_list, dict_key_list):
    value_dict = {}
    for i in range(len(value_list)):
        value_dict[dict_key_list[i]] = [x for x in value_list[i]]
    return value_dict


# Functions for loading/saving network from/into json file
def save_network(filepath, filename, dictionary):
    with open(filepath + filename, 'w') as json_file:
        json.dump(dictionary, json_file, ensure_ascii=False)


def load_network(filepath, filename):
    with open(filepath + filename, "r") as json_file:
        dictionary = json.load(json_file)
    return dictionary


# Functions for DataGenerator
def get_sampling_order(bn):
    order = [bn[0][1][0]]
    for child, _ in bn:
        order.append(child)
    return order


def generate_privacy_preserving_dataset(n, bn_structure, noisy_conditional_contribution, attr_value_matrix):
    print('================ Generating Privacy-Preserving Data ================')
    bn = bn_structure.copy()
    bn_root_attr = bn[0][1][0]
    root_attr_dist = noisy_conditional_contribution.get(bn_root_attr)
    bn_attr_order = get_sampling_order(bn)
    encoded_df = pd.DataFrame(columns=bn_attr_order)
    encoded_df[bn_root_attr] = np.random.choice(attr_value_matrix.get(bn_root_attr), size=n, p=root_attr_dist)
    print('Completing ROOT ' + bn_root_attr)

    for child, parents in bn:
        child_conditional_distributions = noisy_conditional_contribution.get(child)
        for parents_instance in child_conditional_distributions.keys():
            dist = child_conditional_distributions.get(parents_instance)
            parents_instance = list(eval(parents_instance))

            filter_condition = pd.Series(True, index=encoded_df.index)
            # Apply conditions iteratively
            for parent, value in zip(parents, parents_instance):
                filter_condition &= (encoded_df[parent] == value)

            size = encoded_df[filter_condition].shape[0]
            if size:
                encoded_df.loc[filter_condition, child] = np.random.choice(attr_value_matrix.get(child), size=size,
                                                                           p=dist)
        print('Completing attribute ' + child)
    print('========================== Data Generated ==========================')
    return encoded_df


# Functions for Information Loss
def compute_AABM(original, synthetic):
    mean_original = original.apply(lambda x: x.mean())
    mean_synthetic = synthetic.apply(lambda x: x.mean())
    mean_value = pd.concat([mean_original, mean_synthetic], axis=1).round(3)
    mean_value.columns = ["Original", "Synthetic"]
    mean_difference = mean_value.apply(lambda x: abs((x['Synthetic'] - x['Original']) / x['Original']) if x['Original'] != 0.0 else 0.0, axis=1)
    return sum(mean_difference) / len(mean_difference)


def compute_AABSD(original, synthetic):
    std_original = original.apply(lambda x: x.std())
    std_synthetic = synthetic.apply(lambda x: x.std())
    std_value = pd.concat([std_original, std_synthetic], axis=1)
    std_value.columns = ["Original", "Synthetic"]
    std_difference = std_value.apply(lambda x: abs((x['Synthetic'] - x['Original']) / x['Original']), axis=1)
    return sum(std_difference) / len(std_difference)


def compute_AABCO(original, synthetic):
    corr_difference = []
    corr_original = original.corr()
    corr_synthetic = synthetic.corr()
    feature_set = original.columns.values.tolist()
    for a_feature in original.columns.values:
        feature_set.remove(a_feature)
        for b_feature in feature_set:
            r_synthetic = corr_synthetic.loc[a_feature, b_feature].round(3)
            r_original = corr_original.loc[a_feature, b_feature].round(3)
            corr_difference.append(abs((r_synthetic - r_original) / r_original) if r_original != 0.0 else 0.0)
    return sum(corr_difference) / len(corr_difference)


if __name__ == '__main__':

    # # #

    # Print the start time of the execution
    start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Start: ' + str(start))

    # Specify the filename to retrieve data and save the results
    data_filepath = '../data/'
    data_filename = 'sample.csv'

    # Load data from file
    data_original = pd.read_csv(data_filepath + data_filename, header=0)
    loading_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Data Loaded: ' + str(loading_time))

    # # #

    # Algorithm 1 (DataLearner)GreedyBayes
    # Setting parameters
    # k: the maximum number of parents in Bayesian network, i.e., the degree of Bayesian network
    k = 4
    # X_1: the root attribute for constructing the Bayesian network
    X_1 = 'label'
    # Specify the filename to save results
    BN_structure_filename = 'BNStructure'
    BN_filepath = '../results/BNStructure/'
    BN_filename = BN_structure_filename + '_root_' + X_1 + '_k_' + str(k) + '.json'

    # Constructing the network
    N = greedy_bayes(data_original, k, X_1)
    print(N)
    
    # Saving the constructed Bayesian network as '.json' file
    if os.path.isfile(BN_filepath + BN_filename) is False:
        save_structure(BN_filepath, BN_filename, N)

    # Print the end time of the BN construction
    BN_constructed_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Bayesian network construction completed: ' + str(BN_constructed_time))

    # # #

    # Algorithm 2 (DataLearner)NoisyCondProb
    # Setting parameters
    # epsilon: a parameter determines the level of Differential Privacy
    epsilon = 1
    # Number of values in full space for numerical feature
    max_feature_value_num = 2
    # Specify the filename to save results
    NoisyCondProb_filepath = '../results/NoisyConditionalProbability/'
    NoisyCondProb_filename = 'NoisyCondProb_root_' + X_1 + '_epsilon_' + str(epsilon) + '.json'
    AttrUniqueValues_filepath = '../results/AttributeUniqueValues/'
    AttrUniqueValues_filename = 'AttrUniqueValues_' + data_filename.removesuffix(".csv") + '.json'

    # Computing conditional distribution
    try:
        N = load_structure(BN_filepath, BN_filename)
    except FileNotFoundError:
        N = greedy_bayes(data_original, k, X_1)

    conditional_probabilities, attribute_unique_values = construct_noisy_conditional_distributions(N,
                                                                                                   data_original,
                                                                                                   epsilon,
                                                                                                   max_feature_value_num)

    # Saving the constructed noisy conditional distributions as '.json' file
    if os.path.isfile(NoisyCondProb_filepath + NoisyCondProb_filename) is False:
        save_network(NoisyCondProb_filepath, NoisyCondProb_filename, conditional_probabilities)

    # AttributeValueMatrix
    attribute_values_dict = transform_range_to_list(attribute_unique_values, data_original.columns.tolist())
    if os.path.isfile(AttrUniqueValues_filepath + AttrUniqueValues_filename) is False:
        save_network(AttrUniqueValues_filepath, AttrUniqueValues_filename, attribute_values_dict)

    # Print the end time of the noisy distribution construction
    NoisyCondProb_constructed_time = datetime.datetime.now()
    print('Noisy Conditional Distribution Constructed: ' + str(NoisyCondProb_constructed_time))

    # # #

    # DataGenerator
    # Setting parameters
    # n_records_generate: the number of records to generate in the synthetic dataset
    n_records_generate = 400
    # Specify the filename to save results
    synthetic_data_filepath = '../results/PrivatePreservingDataset/'
    synthetic_data_filename = 'sample_root_' + X_1 + '_k_' + str(k) + '_epsilon_' + str(epsilon) \
                              + '_n_' + str(n_records_generate) + '.csv'

    # Loading data learner
    try:
        N = load_structure(BN_filepath, BN_filename)
    except FileNotFoundError:
        N = greedy_bayes(data_original, k, X_1)
    try:
        conditional_probabilities = load_network(NoisyCondProb_filepath, NoisyCondProb_filename)
        attribute_unique_values = load_network(AttrUniqueValues_filepath, AttrUniqueValues_filename)
    except FileNotFoundError:
        conditional_probabilities, attribute_unique_values = construct_noisy_conditional_distributions(N,
                                                                                                       data_original,
                                                                                                       epsilon,
                                                                                                       max_feature_value_num)
    loading_time = datetime.datetime.now()
    print('BN Loaded: ' + str(loading_time))

    # Generating synthetic dataset
    synthetic_dataset = generate_privacy_preserving_dataset(n_records_generate, N,
                                                            conditional_probabilities, attribute_unique_values)
    
    # Saving data to file
    if os.path.isfile(synthetic_data_filepath + synthetic_data_filename) is False:
        synthetic_dataset.to_csv(synthetic_data_filepath + synthetic_data_filename, index=0)

    # Print the end time of the data generation
    Data_synthesized_time = datetime.datetime.now()
    print('Data Generated: ' + str(Data_synthesized_time))

    # # #

    # Compute Information Loss
    # For demonstration purposes only; formal calculations should be based on the full dataset
    data_synthetic = pd.read_csv(synthetic_data_filepath + synthetic_data_filename, header=0)
    print("### For demonstration purposes only; formal comparisons should be based on the full dataset ###")

    # Compute AABM (Average Absolute Bias of Mean)
    AABM = compute_AABM(data_original, data_synthetic)
    print('AABM:', AABM)

    # Compute AABSD (Average Absolute Bias of Standard Deviation)
    AABSD = compute_AABSD(data_original, data_synthetic)
    print('AABSD:', AABSD)

    # Compute AABCO (Average Absolute Bias of Correlation)
    AABCO = compute_AABCO(data_original, data_synthetic)
    print('AABCO:', AABCO)

    # # #
