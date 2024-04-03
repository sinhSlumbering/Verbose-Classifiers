import pandas as pd
from math import log2

data = [
    ["Machamp", "Fighting","None", 264,139,207, "Yes", "No", "Cloudy", "No"],
    ["Snorlax", "Normal", "None", 190, 169, 330, "No", "No", "Sunny", "No"],
    ["Tyranitar", "Rock", "Dark", 309, 276, 225, "No", "Yes", "Sunny", "Yes"],
    ["Gengar", "Ghost", "Poison", 349, 199, 155, "No", "Yes", "Cloudy", "Yes"],
    ["Alakazam", "Psychic", "None", 367, 207, 146, "No", "Yes", "Sunny", "No"],
    ["Rayquaza", "Dragon", "Flying", 377, 210, 227, "No", "Yes", "Windy", "Yes"],
    ["Venusaur", "Grass", "Poison", 198, 189, 190, "Yes", "No", "Rainy", "No"],
    ["Charizard", "Fire", "Flying", 273, 213, 186, "Yes", "No", "Sunny", "No"],
    ["Gardevoir", "Psychic", "Fairy", 326, 229, 169, "No", "Yes", "Rainy", "Yes"],
    ["Krookodile", "Ground", "Dark", 289, 138, 216, "Yes", "No", "Sunny", "Yes"],
    ["Scizor", "Bug", "Steel", 279, 250, 172, "No", "Yes", "Rainy", "Yes"],
    ["Weavile", "Dark", "Ice", 243, 171, 172, "Yes", "No", "Rainy", "Yes"],
    ["Honchkrow", "Dark", "Flying", 243, 103, 225, "Yes", "No", "Windy", "Yes"],
    ["Blaziken", "Fire", "Fighting", 329, 168, 190, "No", "Yes", "Sunny", "No"],
    ["Swampert", "Water", "Ground", 283, 218, 225, "No", "Yes", "Rainy", "No"],
    ["Salamence", "Dragon", "Flying", 310, 251, 216, "No", "Yes", "Windy", "No"],
    ["Hydreigon", "Dragon", "Dark", 256, 188, 211, "No", "No", "Windy", "Yes"],
    ["Metagross", "Steel", "Psychic", 310, 251, 216, "No", "Yes", "Cloudy", "No"],
    ["Lucario","Fighting","Steel",310,175,172,"Yes","No","Cloudy","No"],
    ["Banette", "Dark","Ghost",312,160,162,"No","Yes", "Cloudy","Yes"]																	
]
attributes = ["Name", "Type1", "Type2", "Attack", "Defense", "Stamina",  "Is Shadow", "Has Mega Evo", "Enhanced By Weather", "Good For Mewtwo Raid"]  


# Discretize attack, defense, and stamina


def discretize_data(data, attributes):
  """
  Discretizes Attack, Defense, and Stamina based on mean and standard deviation.
  """
  discretized_data = data.copy()
  for attribute in ["Attack", "Defense", "Stamina"]:
    # Calculate mean and standard deviation

    mean = data[attribute].mean()
    std = data[attribute].std()

    # Define thresholds for low, mid, and high categories (one standard deviation from mean)
    low_threshold = mean - std
    high_threshold = mean + std
    print(attribute, mean, std, low_threshold, high_threshold)
    # Discretize based on thresholds
    discretized_data.loc[data[attribute] <= low_threshold, attribute] = "Low"
    discretized_data.loc[(data[attribute] > low_threshold) & (data[attribute] <= high_threshold), attribute] = "Mid"
    discretized_data.loc[data[attribute] > high_threshold, attribute] = "High"
  return discretized_data


def calculate_entropy(data, target_column):
  """
  Calculates the entropy of a dataset for a given target column.
  """
  class_counts = data[target_column].value_counts()
  entropy = 0
  for count in class_counts:
    proportion = count / len(data)
    entropy -= proportion * log2(proportion + 1e-10)
    print(f"count {count} / length {len(data)}")  # Add a small epsilon to avoid log(0)
    print(f"subset entropy {entropy}=-log({proportion})")
  return entropy


def information_gain(data, split_attribute, target_column):

  parent_entropy = calculate_entropy(data, target_column)

  # Get unique values of the split attribute
  values = data[split_attribute].unique()
#   print(values)

  # Calculate weighted entropy after split
  weighted_entropy = 0
  for value in values:
    # Filter data for this value of the split attribute
    filtered_data = data[data[split_attribute] == value]
    # print(filtered_data)
    # Calculate entropy for the filtered data subset
    print("\n"+value)
    print(f"proportion -> {len(filtered_data)/len(data)} = {len(filtered_data)}/{len(data)} * ")
    subset_entropy = calculate_entropy(filtered_data, target_column)
    # Weight by proportion of data in this subset
    proportion = len(filtered_data) / len(data)
    weighted_entropy += proportion * subset_entropy
  print(f"parent entropy: {parent_entropy}, weighted entropy: {weighted_entropy}\n")
  # Return information gain
  return parent_entropy - weighted_entropy

def intrinsic_info(data, attribute):
  """
  Calculates the intrinsic information (split info) for a given attribute.
  """
  values = data[attribute].unique()
  intrinsic_info = 0
  for value in values:
    proportion = len(data[data[attribute] == value]) / len(data)
    intrinsic_info -= proportion * log2(proportion + 1e-10)
  return intrinsic_info


def gain_ratio(data, split_attribute, target_column):
  """
  Calculates the gain ratio of a specific attribute split.
  """
  information_gain_value = information_gain(data.copy(), split_attribute, target_column)
  intrinsic_info_value = intrinsic_info(data, split_attribute)
  if intrinsic_info_value == 0:
    return 0  # Avoid division by zero
  return information_gain_value / intrinsic_info_value

def build_decision_tree(data, target_column, attributes):
  """
  Builds a decision tree recursively using information gain.
  """
  # Base case: All data belongs to one class or no more attributes left
  if len(data[target_column].unique()) == 1 or len(attributes) == 0:
    return data[target_column].iloc[0]  # Return the majority class

  # Find attribute with highest information gain
  best_attribute = None
  max_gain = float("-inf")  # Set to negative infinity
  for attribute in attributes:
    gain = information_gain(data.copy(), attribute, target_column)
    print(f"information gain for attribute {attribute}: {gain}\n\n")
    if gain > max_gain:
      max_gain = gain
      best_attribute = attribute
  # print(attributes)
  # Build subtrees for each value of the best attribute
  tree = {best_attribute: {}}
  for value in data[best_attribute].unique():
    filtered_data = data[data[best_attribute] == value]
    remaining_attributes = attributes.copy()
    # print(remaining_attributes)
    remaining_attributes.remove(best_attribute)
    print(best_attribute, value)
    print(filtered_data.to_string())
    subtree = build_decision_tree(filtered_data, target_column, remaining_attributes)
    tree[best_attribute][value] = subtree

  return tree


def classify(tree, data):
  """
  Classifies a new data point using the built decision tree.
  """
  attribute = next(iter(tree))  # Get the root node attribute
  value = data[attribute]

  if isinstance(tree[attribute][value], str):  # Leaf node (class label)
    return tree[attribute][value]
  else:
    # Recursively classify based on the subtree
    return classify(tree[attribute][value], data)


# Prepare data
data = pd.DataFrame(data, columns=attributes)
data = data.drop("Name", axis=1)  # Assuming name is not relevant
print(attributes)
discretized_data = discretize_data(data.copy(), attributes.copy())
print(discretized_data.to_string())

target_column = "Good For Mewtwo Raid"
attributes = attributes[1:-1]  # Exclude target column
print(attributes)
# Build the decision tree
decision_tree = build_decision_tree(discretized_data.copy(), target_column, attributes.copy())

# Print the decision tree (optional)
# You can use libraries like `pprint` for a more readable representation
print(decision_tree)

# Prepare data
# data = pd.DataFrame(data, columns=attributes)
# data = data.drop("Name", axis=1)  # Assuming name is not relevant
# discretized_data = discretize_data(data.copy(), attributes.copy())

target_column = "Good For Mewtwo Raid"

# Information Gain for Entire Dataset
# entire_data_entropy = calculate_entropy(discretized_data, target_column)
# print("Entropy of Entire Dataset:", entire_data_entropy)

# Information Gain for Each Attribute
# for attribute in discretized_data.columns[:-1]:
#   gain = information_gain(discretized_data.copy(), attribute, target_column)
#   print(f"Information Gain for '{attribute}': {gain:.4f}")


# for attribute in discretized_data.columns[:-1]:
#   gain_ratio_value = gain_ratio(discretized_data.copy(), attribute, target_column)
#   print(f"Gain Ratio for '{attribute}': {gain_ratio_value:.4f}")
