import json
import random
from config import CAPTIONS_JSON_PATH, CAPTIONS_TEXT_PATH, TRAIN_JSON_PATH, TEST_JSON_PATH

def create_caption_json(context_file_path, output_json_path):
    """
    Convert a text file of image-caption pairs into a JSON dictionary.

    Parameters:
        context_file_path (str): Path to the input text file (format: image_name, caption).
        output_json_path (str): Path to save the resulting JSON file.
    """
    image_caption_dict = {}

    with open(context_file_path, 'r') as f:
        for line in f:
            img, caption = line.strip().split(',', 1)
            if img not in image_caption_dict:
                image_caption_dict[img] = []
            image_caption_dict[img].append(caption)

    with open(output_json_path, 'w') as out_file:
        json.dump(image_caption_dict, out_file, indent=4)

def split_captions_json(json_path, train_path, test_path, test_ratio=0.2, seed=42):
    # Load full dataset
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Set random seed for reproducibility
    random.seed(seed)

    # Get all image filenames and shuffle them
    all_images = list(data.keys())
    random.shuffle(all_images)

    # Compute split index
    split_idx = int(len(all_images) * (1 - test_ratio))
    train_imgs = all_images[:split_idx]
    test_imgs = all_images[split_idx:]

    # Create split dicts
    train_data = {img: data[img] for img in train_imgs}
    test_data = {img: data[img] for img in test_imgs}

    # Save to files
    with open(train_path, 'w') as f_train:
        json.dump(train_data, f_train, indent=4)

    with open(test_path, 'w') as f_test:
        json.dump(test_data, f_test, indent=4)

    print(f"Train images: {len(train_data)}, Test images: {len(test_data)}")

# Example usage
if __name__ == "__main__":
    create_caption_json(
        context_file_path=CAPTIONS_TEXT_PATH,
        output_json_path=CAPTIONS_JSON_PATH
    )
    split_captions_json(
        json_path=CAPTIONS_JSON_PATH,
        train_path=TRAIN_JSON_PATH,
        test_path=TEST_JSON_PATH,
        test_ratio=0.2
    )
