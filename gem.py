import csv
import os
import json
import argparse
import google.generativeai as genai
from PIL import Image
import time
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from datetime import datetime

# ========== CONFIGURATION ==========
API_FILE = "API.json"
csv_file_path = '../abo-images-small/images/metadata/images.csv'
json_directory = 'abo-listings/listings/metadata/'
images_folder = '../abo-images-small/images/small/'
output_folder = './vqa_dataset/'

# ========== ARGUMENT PARSING ==========
def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate Visual Question Answering dataset')
    parser.add_argument('--partition', type=int, choices=[1, 2, 3], required=True,
                        help='Partition number (1, 2, or 3) to process')
    parser.add_argument('--check-main', type=int, choices=[0, 1], default=0,
                        help='Whether to check for main_image_id (0=no, 1=yes)')
    parser.add_argument('--limit', type=int, default=1000,
                        help='Maximum number of images to process')
    parser.add_argument('--reset', action='store_true',
                        help='Reset progress and start from the beginning of the partition')
    return parser.parse_args()

# ========== STATE MANAGEMENT ==========
def get_state_file_path(partition_number):
    """Get the path to the state file for a specific partition."""
    return os.path.join(output_folder, f"partition_{partition_number}_state.json")

def save_state(partition_number, current_index, successful, failed, total_in_partition):
    """Save the current state to a file."""
    state_data = {
        "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "current_index": current_index,
        "successful_generations": successful,
        "failed_generations": failed,
        "total_in_partition": total_in_partition,
        "completed_percentage": (current_index / total_in_partition * 100) if total_in_partition > 0 else 0
    }
    
    state_file = get_state_file_path(partition_number)
    with open(state_file, 'w') as f:
        json.dump(state_data, f, indent=2)
    
def load_state(partition_number):
    """Load the state from a file if it exists."""
    state_file = get_state_file_path(partition_number)
    
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading state file: {e}")
    
    return None

# ========== API KEY MANAGEMENT ==========
def load_api_keys():
    with open(API_FILE, "r") as f:
        return json.load(f)

def save_api_keys(api_keys):
    with open(API_FILE, "w") as f:
        json.dump(api_keys, f, indent=2)

def rotate_api_key(api_keys):
    failed_key = api_keys.pop(0)
    api_keys.append(failed_key)
    save_api_keys(api_keys)
    print(f"🔁 Rotated API key. Using new key: {api_keys[0]}")
    return api_keys

api_keys = load_api_keys()
if not api_keys:
    print("❌ No API keys found in API.json.")
    exit(1)

genai.configure(api_key=api_keys[0])
safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
}

model = genai.GenerativeModel(
    model_name="models/gemini-1.5-flash",
    safety_settings=safety_settings,
)

# ========== TIMING FUNCTIONS ==========
def format_time(seconds):
    """Format time in seconds to a readable string."""
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {seconds:.2f}s"

# ========== VQA GENERATION ==========
def generate_vqa(image_path, metadata, output_json_path):
    global api_keys, model
    
    # Start timing
    generation_start_time = time.time()
    api_call_time = 0
    
    with Image.open(image_path) as img:
        # prompt2
        prompt = (
            "You are provided with an image of a product and its associated metadata from an e-commerce listing.\n\n"
            "Follow these instructions precisely:\n\n"
            "1. **First**, perform a detailed visual analysis of the image. Extract observable product attributes such as color, shape, material, design elements, and functional features.\n"
            "2. **Then**, refer to the metadata only as a secondary source to confirm or supplement visual evidence. Do not rely solely on metadata.\n"
            "3. Based solely on what can be visually confirmed (and optionally supported by metadata), generate **15 one-word Visual Question Answer (VQA) pairs**.\n"
            "4. Only include **relevant, non-trivial** questions. If the product is simple or minimal, produce fewer but high-quality pairs.\n"
            "5. List the questions in **decreasing order of importance**, prioritizing the most distinctive or defining characteristics.\n"
            "6. Each question must be detailed (approx. 15 words) and clearly refer to something visually evident in the product.\n"
            "7. Each answer must be **a single word**, strictly grounded in the image or metadata.\n"
            "8. Avoid any hallucinations, guesses, generic filler, or duplicate questions.\n"
            "9. Cover diverse aspects such as color, material, texture, design, components, shape, or markings.\n"
            "10. Format your response as a JSON array:\n"
            "    [\n"
            "      {\"question\": \"...\", \"answer\": \"...\"},\n"
            "      ...\n"
            "    ]\n"
            "11. **Return only the JSON array above — no prose, no comments, no formatting tags.**\n"
        )

        attempt_count = 0
        while True:
            attempt_count += 1
            api_start_time = time.time()
            try:
                response = model.generate_content([prompt, img, json.dumps(metadata or {})], stream=False)
                result = response.text
                api_end_time = time.time()
                api_call_time = api_end_time - api_start_time
                break
            except Exception as e:
                api_end_time = time.time()
                api_call_time = api_end_time - api_start_time
                if "429" in str(e):
                    print(f"⚠️ Rate limit hit (attempt {attempt_count}). Rotating API key...")
                    api_keys = rotate_api_key(api_keys)
                    genai.configure(api_key=api_keys[0])
                    model = genai.GenerativeModel(
                        model_name="models/gemini-1.5-flash",
                        safety_settings=safety_settings,
                    )
                else:
                    print(f"❌ Error during generation (attempt {attempt_count}): {str(e)}")
                    generation_end_time = time.time()
                    total_time = generation_end_time - generation_start_time
                    print(f"⏱️ Failed after {format_time(total_time)} (API call: {format_time(api_call_time)})")
                    return False

        try:
            cleaned_result = result.strip().lstrip('```json').rstrip('```').strip()
            qa_pairs = json.loads(cleaned_result)

            # Create the final structure as a list with two elements
            final_output = [
                {"path": image_path},  # 0th element contains the path
                {"questions": qa_pairs}  # 1st element contains the questions
            ]

            with open(output_json_path, "w", encoding="utf-8") as out_f:
                json.dump(final_output, out_f, indent=2)
                
            # Calculate total time
            generation_end_time = time.time()
            total_time = generation_end_time - generation_start_time
            
            print(f"✅ Image path is {image_path}")
            print(f"✅ Saved VQA to {output_json_path}")
            print(f"⏱️ Generation time: {format_time(total_time)} (API call: {format_time(api_call_time)})")
            return True
            
        except json.JSONDecodeError:
            print("❌ GPT output was not valid JSON.")
            print(result)
            
            # Calculate total time even for failures
            generation_end_time = time.time()
            total_time = generation_end_time - generation_start_time
            print(f"⏱️ Failed after {format_time(total_time)} (API call: {format_time(api_call_time)})")
            return False

# ========== JSON LOOKUP ==========
def search_json_entry(directory, image_id, check_main_only):
    start_time = time.time()
    
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    for line_num, line in enumerate(lines, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            if not isinstance(entry, dict):
                                continue

                            main_match = entry.get("main_image_id") == image_id
                            other_match = not check_main_only and image_id in entry.get("other_image_ids", [])

                            if main_match or other_match:
                                end_time = time.time()
                                search_time = end_time - start_time
                                print(f"✅ Found {image_id} in {filename} (line {line_num})")
                                print(f"⏱️ Search time: {format_time(search_time)}")
                                return entry
                        except json.JSONDecodeError:
                            print(f"⚠️ Skipping malformed JSON at {filename} line {line_num}")
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
    
    end_time = time.time()
    search_time = end_time - start_time
    print(f"⏱️ Search completed in {format_time(search_time)} - No match found")
    return None

# ========== GET PARTITION ==========
def get_partition(all_rows, partition_number, total_partitions=3):
    """Split rows into partitions and return the specified partition."""
    # Calculate partition size and boundaries
    total_images = len(all_rows)
    partition_size = total_images // total_partitions
    
    start_idx = (partition_number - 1) * partition_size
    
    # For the last partition, include any remaining images
    if partition_number == total_partitions:
        end_idx = total_images
    else:
        end_idx = partition_number * partition_size
    
    partition_rows = all_rows[start_idx:end_idx]
    
    print(f"📊 Partition {partition_number}/{total_partitions}: " 
          f"Processing images {start_idx+1}-{end_idx} out of {total_images} " 
          f"({len(partition_rows)} images)")
    
    return partition_rows

# ========== MAIN FUNCTION ==========
def main():
    args = parse_arguments()
    partition_number = args.partition
    check_main_image_id = args.check_main
    max_iterations = args.limit
    reset_state = args.reset
    
    # Create output folder with partition number
    partition_output_folder = os.path.join(output_folder, f"partition_{partition_number}")
    os.makedirs(partition_output_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)  # Ensure main output folder exists too

    start_time = time.time()
    
    # Load all images from CSV at once
    with open(csv_file_path, mode='r') as file:
        all_rows = list(csv.DictReader(file))
    
    # Reverse the list to start from the bottom
    all_rows.reverse()
    
    # Get the specific partition for this run
    partition_rows = get_partition(all_rows, partition_number)
    total_in_partition = len(partition_rows)
    
    # Check if we need to resume or start fresh
    start_index = 0
    successful_generations = 0
    failed_generations = 0
    
    if not reset_state:
        existing_state = load_state(partition_number)
        if existing_state:
            start_index = existing_state.get("current_index", 0)
            successful_generations = existing_state.get("successful_generations", 0)
            failed_generations = existing_state.get("failed_generations", 0)
            
            # Don't start beyond the end of the partition
            if start_index >= total_in_partition:
                start_index = 0
            
            print(f"🔄 Resuming from index {start_index} ({start_index/total_in_partition*100:.1f}% completed)")
            print(f"📊 Previous stats: {successful_generations} successful, {failed_generations} failed")
    else:
        print(f"🔄 Starting fresh (--reset flag used)")
    
    # Calculate how many iterations to process
    remaining_in_partition = total_in_partition - start_index
    max_iterations = min(max_iterations, remaining_in_partition)
    
    print(f"🚀 Starting VQA generation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Target: Processing {max_iterations} of {remaining_in_partition} remaining images (Partition {partition_number})")
    print(f"📅 Current Date/Time: 2025-04-30 16:35:18")
    print(f"👤 User: Abhinav-gh")
    print(f"🔍 Check main_image_id: {'Yes' if check_main_image_id else 'No'}")
    
    loop_counter = 0
    
    try:
        while loop_counter < max_iterations:
            current_index = start_index + loop_counter
            loop_counter += 1
            iteration_start_time = time.time()
            
            # Save state after every 10 iterations or if it's the first one
            if loop_counter == 1 or loop_counter % 10 == 0:
                save_state(partition_number, current_index, successful_generations, failed_generations, total_in_partition)
            
            print(f"\n{'='*50}")
            print(f"⭐ PARTITION {partition_number} - ITERATION {loop_counter}/{max_iterations}")
            print(f"  (Index {current_index}/{total_in_partition}, {successful_generations} successful, {failed_generations} failed)")
            print(f"{'='*50}")
            
            # Get image metadata from the partition at the current index
            image_metadata = partition_rows[current_index]
            
            image_id = image_metadata['image_id']
            image_path = os.path.join(images_folder, image_metadata['path'])
            print(f"🔍 Processing image_id: {image_id}")

            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}")
                failed_generations += 1
                continue

            metadata = None
            if check_main_image_id == 1:
                print(f"🔎 Looking for metadata with image_id {image_id} as main_image_id...")
                metadata = search_json_entry(json_directory, image_id, check_main_only=True)
                if not metadata:
                    print(f"❌ image_id {image_id} not found as main_image_id.")
                    failed_generations += 1
                    continue
            elif check_main_image_id == 0:
                metadata = None  # skip lookup entirely

            print(f"✅ Image {'and metadata ' if metadata else ''}found. Proceeding to VQA generation.")
            output_json_path = os.path.join(partition_output_folder, f"{image_id}.json")
            
            # Skip if already processed
            if os.path.exists(output_json_path):
                print(f"⏩ Skipping {image_id} - already processed")
                continue
                
            success = generate_vqa(image_path, metadata, output_json_path)
            if success:
                successful_generations += 1
            else:
                failed_generations += 1
            
            # Calculate and display iteration timing
            iteration_end_time = time.time()
            iteration_time = iteration_end_time - iteration_start_time
            total_time_so_far = iteration_end_time - start_time
            avg_time_per_iteration = total_time_so_far / loop_counter
            
            # Estimate remaining time
            remaining_iterations = max_iterations - loop_counter
            estimated_remaining_time = remaining_iterations * avg_time_per_iteration
            
            # Save state more frequently for long-running iterations
            if iteration_time > 60:  # If an iteration takes more than a minute
                save_state(partition_number, current_index, successful_generations, failed_generations, total_in_partition)
            
            print(f"⏱️ Iteration {loop_counter} completed in {format_time(iteration_time)}")
            print(f"📊 Progress: {loop_counter}/{max_iterations} iterations ({(loop_counter/max_iterations*100):.1f}%)")
            print(f"📈 Partition progress: {(current_index+1)/total_in_partition*100:.1f}% of partition {partition_number}")
            print(f"⏳ Estimated time remaining: {format_time(estimated_remaining_time)}")
            
            if successful_generations + failed_generations > 0:
                success_rate = (successful_generations/(successful_generations+failed_generations)*100)
                print(f"📈 Success rate: {success_rate:.1f}%")
            
            # Sleep between iterations
            if loop_counter < max_iterations:
                print(f"😴 Sleeping for 5 seconds...")
                time.sleep(5)
    
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
    finally:
        # Always save state at the end
        current_index = start_index + loop_counter - 1
        save_state(partition_number, current_index, successful_generations, failed_generations, total_in_partition)
        
        # Final statistics
        end_time = time.time()
        total_runtime = end_time - start_time
        
        print(f"\n{'='*50}")
        print(f"🏁 PARTITION {partition_number} SESSION ENDED AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")
        print(f"📊 Total runtime: {format_time(total_runtime)}")
        print(f"✅ Successful generations: {successful_generations}/{loop_counter} ({(successful_generations/max(1, loop_counter)*100):.1f}%)")
        print(f"❌ Failed generations: {failed_generations}/{loop_counter} ({(failed_generations/max(1, loop_counter)*100):.1f}%)")
        print(f"⏱️ Average time per generation: {format_time(total_runtime/max(1, loop_counter))}")
        print(f"📈 Overall partition progress: {(current_index+1)/total_in_partition*100:.1f}% completed")
        print(f"🔄 To resume, run the same command again.")
        print(f"🔄 To start over, add the --reset flag.")

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    main()