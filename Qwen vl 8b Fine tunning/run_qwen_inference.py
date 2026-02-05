from unsloth import FastVisionModel
from transformers import TextStreamer
import torch
from PIL import Image, ImageDraw
import json
import os
import glob

# --- Configuration ---
# Path to your fine-tuned model (e.g., "lora_model" or "outputs/checkpoint-130")
MODEL_PATH = "lora_model" 
# Path to a test image
TEST_IMAGE_PATH = "floorplans_dataset/test/images/13_jpg.rf.ecf5fa2028e2c49913ee40e051eb4adf.jpg" 
# The same instruction used during training
INSTRUCTION = "Detect all walls in this floor plan and output the result in JSON format: {'walls': [{'bbox': [x_min, y_min, x_max, y_max]}, ...]}"

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = model_path,
        load_in_4bit = True,
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer

def run_inference(model, tokenizer, image_path):
    print(f"Running inference on {image_path}...")
    image = Image.open(image_path).convert("RGB")
    
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": INSTRUCTION}
        ]}
    ]
    
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    # Use TextStreamer to see output in real-time
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=512,
        use_cache=True,
        temperature=0.1, # Low temperature for more deterministic JSON
        streamer=streamer
    )
    
    # Decode the full response
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    return output_text

def extract_json(text):
    """Extracts JSON object from a string potentially containing other text."""
    try:
        # Heuristic: split by "assistant" if present (common in chat templates)
        if "assistant" in text:
            text = text.split("assistant")[-1]
            
        # Strip markdown code blocks if present
        text = text.replace("```json", "").replace("```", "").strip()
        
        # Find the first '{' and the last '}'
        start = text.find('{')
        end = text.rfind('}') + 1
        
        if start != -1 and end != -1:
            json_str = text[start:end]
            return json.loads(json_str)
        else:
            print("No JSON brackets found in output.")
            # print(f"Raw output: {text}") # Debug
            return None
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        try:
            # Fallback: simple fix for single quotes
            fixed_str = json_str.replace("'", '"')
            return json.loads(fixed_str)
        except:
            pass
        return None

def bbox_to_yolo(bbox, img_width=1000, img_height=1000, class_id=0):
    """Converts [xmin, ymin, xmax, ymax] (0-1000 scale) to [class_id, x_center, y_center, width, height] (normalized 0-1)."""
    x_min, y_min, x_max, y_max = bbox
    
    # Box dimensions in 0-1000 scale
    box_w = x_max - x_min
    box_h = y_max - y_min
    
    # Box center in 0-1000 scale
    x_c = x_min + box_w / 2
    y_c = y_min + box_h / 2
    
    # Normalize (model output is always 0-1000)
    x_c_n = x_c / 1000
    y_c_n = y_c / 1000
    w_n = box_w / 1000
    h_n = box_h / 1000
    
    return [class_id, x_c_n, y_c_n, w_n, h_n]

def save_yolo_labels(json_data, image_path, output_dir="inference_labels"):
    if not json_data or "walls" not in json_data:
        print("No valid 'walls' data found to convert.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Output filename
    base_name = os.path.basename(image_path)
    txt_name = os.path.splitext(base_name)[0] + ".txt"
    output_path = os.path.join(output_dir, txt_name)
    
    lines = []
    for item in json_data["walls"]:
        bbox = item["bbox"]
        # Model predicted coords are 0-1000. Convert to normalized YOLO.
        yolo_vals = bbox_to_yolo(bbox)
        # Format: class x_c y_c w h
        line = f"{yolo_vals[0]} {yolo_vals[1]:.6f} {yolo_vals[2]:.6f} {yolo_vals[3]:.6f} {yolo_vals[4]:.6f}"
        lines.append(line)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Saved YOLO labels to {output_path}")

def visualize(image_path, json_data, output_path="prediction_vis.jpg"):
    if not json_data or "walls" not in json_data:
        return

    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    print(f"Visualizing on image size: {width}x{height}")
    
    for item in json_data["walls"]:
        bbox = item["bbox"] # [xmin, ymin, xmax, ymax] in 0-1000 coords output by model
        
        # Scale to actual image size for visualization
        x_min = bbox[0] * (width / 1000)
        y_min = bbox[1] * (height / 1000)
        x_max = bbox[2] * (width / 1000)
        y_max = bbox[3] * (height / 1000)
        
        scaled_bbox = [x_min, y_min, x_max, y_max]
        
        draw.rectangle(scaled_bbox, outline="red", width=3)
    
    img.save(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Model path {MODEL_PATH} does not exist. Please check your training output.")
        
    try:
        model, tokenizer = load_model(MODEL_PATH)
        
        if not os.path.exists(TEST_IMAGE_PATH):
            search_path = "floorplans_dataset/valid/images/*.jpg"
            files = glob.glob(search_path)
            if files:
                TEST_IMAGE_PATH = files[0]
            else:
                print("No test images found.")
                exit()

        raw_output = run_inference(model, tokenizer, TEST_IMAGE_PATH)
        parsed_data = extract_json(raw_output)
        
        if parsed_data:
            print("\nParsed JSON:", parsed_data)
            save_yolo_labels(parsed_data, TEST_IMAGE_PATH)
            visualize(TEST_IMAGE_PATH, parsed_data)
        else:
            print("Failed to extract valid JSON from model output.")

    except Exception as e:
        print(f"An error occurred: {e}")
