import json
import argparse
import os

def filter_noise(video_data, k=3):
    """
    Implements the Consistency Filter.
    Rule: An object must appear in at least 'k' frames to be kept.
    """
    print(f"Analyzing {video_data['video_id']} for noise...")
    
    # 1. Count how many frames each object appears in
    object_counts = {}
    for frame in video_data['frames']:
        # Use a set to avoid counting the same object twice in one frame
        unique_objs = set(frame['objects'])
        for obj in unique_objs:
            object_counts[obj] = object_counts.get(obj, 0) + 1
            
    # 2. Identify "Real" objects (Assume appearing >= k times is real)
    valid_objects = {obj for obj, count in object_counts.items() if count >= k}
    
    print(f"  > Found Objects: {list(object_counts.keys())}")
    print(f"  > Keeping only persistent objects: {list(valid_objects)}")
    
    # 3. Create clean frames with only valid objects
    filtered_frames = []
    for frame in video_data['frames']:
        clean_objs = [obj for obj in frame['objects'] if obj in valid_objects]
        
        new_frame = frame.copy()
        new_frame['objects'] = clean_objs
        filtered_frames.append(new_frame)
        
    return {
        "video_id": video_data['video_id'],
        "frames": filtered_frames,
        "valid_objects": list(valid_objects)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mock', action='store_true', help="Use mock data")
    args = parser.parse_args()

    # Define paths
    input_file = 'data/mock_data/video_mock.json'
    output_dir = 'data/processed_json'
    os.makedirs(output_dir, exist_ok=True)

    if args.mock:
        print(f"--- RUNNING MODULE 1: ROBUST PERCEPTION (MOCK MODE) ---")
        if not os.path.exists(input_file):
            print(f"Error: {input_file} not found!")
            return

        with open(input_file, 'r') as f:
            raw_data = json.load(f)
            
        for video in raw_data:
            clean_data = filter_noise(video, k=3)
            
            # Save the result
            out_path = os.path.join(output_dir, f"{video['video_id']}_clean.json")
            with open(out_path, 'w') as out_f:
                json.dump(clean_data, out_f, indent=2)
            print(f"  > Success! Clean data saved to {out_path}")

if __name__ == "__main__":
    main()