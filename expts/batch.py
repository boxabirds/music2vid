import json
import argparse

class Img:
    def __init__(self, H=None, W=None, sampler=None):
        self.H = H
        self.W = W
        self.sampler = sampler

def do_render(filename, img):
    print(f"Filename: {filename}, Img properties: {vars(img)}")

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='Path to the input JSON file')
args = parser.parse_args()
input_file = args.input

with open(input_file, "r") as file:
    data = json.load(file)

compositions = data["compositions"]
combinations = data["combinations"]

def generate_combinations(properties, current_combination, index):
    if index == len(properties):
        yield current_combination
    else:
        prop_names, prop_values = properties[index]
        for value_set in prop_values:
            new_combination = current_combination.copy()
            if len(prop_names) == 1:  # Single property
                new_combination[prop_names[0]] = value_set
            else:  # Multiple properties
                for prop_name, value in zip(prop_names, value_set):
                    new_combination[prop_name] = value
            yield from generate_combinations(properties, new_combination, index + 1)

def extract_properties(combinations):
    extracted_properties = []
    for keys, values in combinations.items():
        key_list = keys.split(",")
        prop_names = [key.strip().split(".")[-1] for key in key_list]
        extracted_properties.append((prop_names, values))
    return extracted_properties

properties = extract_properties(combinations)
print(f"Properties: {properties}")

for composition in compositions:
    for combination in generate_combinations(properties, {}, 0):
        img_instance = Img()
        for prop_name, value in combination.items():
            setattr(img_instance, prop_name, value)
        do_render(composition, img_instance)
