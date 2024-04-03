import numpy as np
import pandas as pd
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

df = pd.DataFrame(data, columns=attributes)

print(df)

def calculate_gaussian_probability(x, mean, std):
  print(f"{x}, mean ->{mean}, stddev->{std}")
  import math
  return (1 / (math.sqrt(2 * math.pi) * std)) * math.exp(-((x - mean) ** 2) / (2 * std**2))


def calculate_class_probability(data, attributes, target_attribute, target_class, new_instance):
  class_counts = {}
  for item in data:
    class_value = item[attributes.index(target_attribute)]
    class_counts[class_value] = class_counts.get(class_value, 0) + 1

  print(class_counts)

  # Calculate total number of data points
  total_data_points = sum(class_counts.values())
  print(total_data_points)

  # Calculate class prior probability
  class_prior = class_counts.get(target_class, 0) / total_data_points
  print(f"Class prior for '{target_class}': {class_prior}")

  # Calculate likelihood for each attribute
  likelihood = 1.0
  for i, attribute in enumerate(attributes):
    if attribute == target_attribute or attribute == "Name":
      continue  # Skip target attribute for likelihood calculation
    if pd.api.types.is_numeric_dtype(df[attribute]):

      # Assuming Gaussian distribution (normal distribution) for numeric attributes
      target_class_data = df.loc[df[target_attribute] == target_class, attribute]
      mean = target_class_data.mean()
      std = target_class_data.std()
      likelihood *= calculate_gaussian_probability(new_instance[i], mean, std)
      print(f"Likelihood for '{attribute}' (Gaussian): {calculate_gaussian_probability(new_instance[i], mean, std)}")
    else:
        attribute_value = new_instance[i]
        matching_data_points = 0
        for item in data:
            if item[i] == attribute_value and item[attributes.index(target_attribute)] == target_class:
                matching_data_points += 1
        # Add smoothing (avoid zero probabilities)
        print(class_counts[target_class]+len(set(df[attribute])))
        print(set(df[attribute]))
        likelihood *= (matching_data_points + 1) / (class_counts[target_class] + len(set(df[attribute])))
        print(f"Likelihood for '{attribute}' being '{attribute_value}':  matches->{matching_data_points} classcounts->{class_counts[target_class]} +{len(set(df[attribute]))} {(matching_data_points + 1) / (class_counts[target_class] + len(set(df[attribute])))}")

  # Calculate posterior probability (considering class prior and likelihood)
  posterior = class_prior * likelihood
  print(f"Posterior probability for '{target_class}': {posterior}")

  return posterior

# Sample data (assuming 'data' and 'attributes' are defined elsewhere)
new_data = [
  ["Garchomp", "Dragon", "Ground", 600, 250, 272, "Yes", "No", "Rainy"],  
  ["Beedril", "Bug", "Poison", 367, 185, 140, "Yes", "No", "Sunny"], 
  ["Azumaril", "Water", "Fairy", 361, 200, 180, "No", "Yes", "Rainy"],
  ["Aggron", "Steel", "Rock", 366, 210, 228, "Yes", "No", "Sunny"],
  ["Rapidash", "Psychic", "Fairy", 207, 162, 163, "No", "No", "Windy"],
  ["Hoopa", "Dark", "Psychic", 311, 191, 173, "No", "No", "Windy"]
]

# Specify target attribute and class
target_attribute = "Good For Mewtwo Raid"
target_class = "Yes"

# Print calculations for each new data point
for new_item in new_data:
  print(f"\n--- Predicting for Pokemon: {new_item[0]} ---")
  infavor = calculate_class_probability(data, attributes, target_attribute, target_class, new_item)
  against = calculate_class_probability(data, attributes, target_attribute, "No", new_item)
  print(new_item[0])
  print(f"infavor ->{infavor}, against -> {against}")
  if infavor > against:
    print("Good Pick")
  else:
    print("Bad Pick")