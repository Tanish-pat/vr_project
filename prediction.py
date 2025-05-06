from transformers import BlipProcessor, BlipForQuestionAnswering
import os, json, csv
from PIL import Image
import torch
from tqdm import tqdm
from pathlib import Path

# Load model and processor
BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").save_pretrained("./Model/blip-saved-model")

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model_path = Path("./Model/blip-saved-model").resolve()
if not model_path.exists():
    raise FileNotFoundError(f"Model path does not exist: {model_path}")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")
# model = BlipForQuestionAnswering.from_pretrained(str(model_path), local_files_only=True).to("cuda")


# Test data directory
test_data_dir = "vqa_dataset"

# Output results
results = []

# Iterate over files in test_data_dir
for file in tqdm(os.listdir(test_data_dir), desc="Processing"):
    json_path = os.path.join(test_data_dir, file)

    with open(json_path, "r") as f:
        sample = json.load(f)

    image_path = sample[0]["path"]
    full_image_path = os.path.join("Data", image_path)
    image = Image.open(full_image_path).convert("RGB")

    for qa in sample[1]["questions"]:
        question = qa["question"]

        inputs = processor(image, question, return_tensors="pt").to("cuda:0", torch.float16)
        out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True)

        results.append((os.path.basename(image_path), question, answer))

# Write results to CSV
with open("Results/results.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Question", "Predicted Answer"])
    writer.writerows(results)

print("Results saved to Results/results.csv")
