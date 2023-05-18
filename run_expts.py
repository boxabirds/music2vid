import json
import argparse
from lib.parameters import DeforumAnimArgs, Root, DeforumArgs

class Img:
    def __init__(self, H=None, W=None, sampler=None):
        self.H = H
        self.W = W
        self.sampler = sampler

class Motion:
    def __init__(self, sampler=None):
        self.sampler = sampler

def do_render(filename, img, motion):
    print(f"Filename: {filename}, Img properties: {vars(img)}, Motion properties: {vars(motion)}")

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
                prop_name = prop_names[0]
                if prop_name.startswith("img."):
                    img_prop_name = prop_name.split(".")[1]
                    if "img" not in new_combination:
                        new_combination["img"] = {}
                    new_combination["img"][img_prop_name] = value_set
                elif prop_name.startswith("motion."):
                    motion_prop_name = prop_name.split(".")[1]
                    if "motion" not in new_combination:
                        new_combination["motion"] = {}
                    new_combination["motion"][motion_prop_name] = value_set
            else:  # Multiple properties
                img_dict = {}
                motion_dict = {}
                for prop_name, value in zip(prop_names, value_set):
                    if prop_name.startswith("img."):
                        img_prop_name = prop_name.split(".")[1]
                        img_dict[img_prop_name] = value
                    elif prop_name.startswith("motion."):
                        motion_prop_name = prop_name.split(".")[1]
                        if "motion" not in new_combination:
                            new_combination["motion"] = {}
                        new_combination["motion"][motion_prop_name] = value
                if img_dict:
                    if "img" not in new_combination:
                        new_combination["img"] = {}
                    new_combination["img"].update(img_dict)
            try:
                for prop_name, value in new_combination["motion"].items():
                    new_combination["motion"][prop_name] = value
            except KeyError:
                pass
            yield from generate_combinations(properties, new_combination, index + 1)


def extract_properties(combinations, dictionary_name):
    extracted_properties = []
    for keys, values in combinations.items():
        key_list = keys.split(",")
        prop_names = [key.strip() for key in key_list]
        if dictionary_name == "img":
            prop_names = [prop_name for prop_name in prop_names if prop_name.startswith("img.")]
        elif dictionary_name == "motion":
            prop_names = [prop_name for prop_name in prop_names if prop_name.startswith("motion.")]
        if prop_names:
            extracted_properties.append((prop_names, values))
    if not extracted_properties:
        print(f"No {dictionary_name} properties found")
    else:
        print(f"Extracted {dictionary_name} properties: {extracted_properties}")
    return extracted_properties

img_properties = extract_properties(combinations, "img")
motion_properties = extract_properties(combinations, "motion")

img_combinations = [{} for _ in range(len(img_properties))]
motion_combinations = [{} for _ in range(len(motion_properties))]

for composition in compositions:
    for img_combination in generate_combinations(img_properties, {}, 0):
        for motion_combination in generate_combinations(motion_properties, {}, 0):
            img_instance = Img()
            for prop_name, value in img_combination["img"].items():
                setattr(img_instance, prop_name, value)
            motion_instance = Motion()
            if "motion" in motion_combination:
                for prop_name, value in motion_combination["motion"].items():
                    setattr(motion_instance, prop_name, value)
            do_render(composition, img_instance, motion_instance)
