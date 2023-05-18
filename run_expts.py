import json
import argparse
from lib.parameters import DeforumAnimArgs, Root, DeforumArgs
from pathlib import Path

def do_render(filename, img:DeforumArgs, motion:DeforumAnimArgs, root:Root):
    #print(f"\n== Filename: {filename},\n\n= Img properties: {vars(img)},\n\n= Motion properties: {vars(motion)}, \n\n=Root properties: {vars(Root)}")
    print(f"== Filename: {filename}: model_checkpoint = '{root.model_checkpoint}', img.W, img.H = {img.W}, {img.H}")

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='Path to the input JSON file')
args = parser.parse_args()
input_file = args.input

with open(input_file, "r") as file:
    data = json.load(file)

compositions = data["compositions"]
combinations = data["combinations"]

def generate_combinations(properties, current_combination, index:int):
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
                elif prop_name.startswith("root."):
                    root_prop_name = prop_name.split(".")[1]
                    if "root" not in new_combination:
                        new_combination["root"] = {}
                    new_combination["root"][root_prop_name] = value_set
            else:  # Multiple properties
                img_dict = {}
                motion_dict = {}
                root_dict = {}
                for prop_name, value in zip(prop_names, value_set):
                    if prop_name.startswith("img."):
                        img_prop_name = prop_name.split(".")[1]
                        img_dict[img_prop_name] = value
                    elif prop_name.startswith("motion."):
                        motion_prop_name = prop_name.split(".")[1]
                        motion_dict[motion_prop_name] = value
                    elif prop_name.startswith("root."):
                        root_prop_name = prop_name.split(".")[1]
                        root_dict[root_prop_name] = value
                if img_dict:
                    if "img" not in new_combination:
                        new_combination["img"] = {}
                    new_combination["img"].update(img_dict)
                if motion_dict:
                    if "motion" not in new_combination:
                        new_combination["motion"] = {}
                    new_combination["motion"].update(motion_dict)
                if root_dict:
                    if "root" not in new_combination:
                        new_combination["root"] = {}
                    new_combination["root"].update(root_dict)
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
        elif dictionary_name == "root":
            prop_names = [prop_name for prop_name in prop_names if prop_name.startswith("root.")]
        if prop_names:
            extracted_properties.append((prop_names, values))
    if not extracted_properties:
        print(f"No {dictionary_name} properties found")
    else:
        print(f"Extracted {dictionary_name} properties: {extracted_properties}")
    return extracted_properties

print(f"Combinations: {combinations}")

img_properties = extract_properties(combinations, "img")
motion_properties = extract_properties(combinations, "motion")
root_properties = extract_properties(combinations, "root")

img_combinations = [{} for _ in range(len(img_properties))]
motion_combinations = [{} for _ in range(len(motion_properties))]
root_combinations = [{} for _ in range(len(root_properties))]

count = 0
for composition in compositions:
    for img_combination in generate_combinations(img_properties, {}, 0):
        for motion_combination in generate_combinations(motion_properties, {}, 0):
            for root_combination in generate_combinations(root_properties, {}, 0):
                #print(f"### composition: {Path(composition).stem} img: {img_combination} / motion: {motion_combination} / root: {root_combination}")
                img_instance = DeforumArgs()
                if "img" in img_combination:
                    print(f"### img_combination: {img_combination}")
                    for prop_name, value in img_combination["img"].items():
                        setattr(img_instance, prop_name, value)
                motion_instance = DeforumAnimArgs()
                if "motion" in motion_combination:
                    for prop_name, value in motion_combination["motion"].items():
                        setattr(motion_instance, prop_name, value)
                root_instance = Root()
                if "root" in root_combination:
                    for prop_name, value in root_combination["root"].items():
                        setattr(root_instance, prop_name, value)
                count += 1
                do_render(composition, img_instance, motion_instance, root_instance)

print(f"{count} combinations generated")
