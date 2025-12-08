import os
from PIL import Image

def augment_data(base_dir):
    # The subfolders to process
    subfolders = ['train', 'val', 'test']
    
    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    for folder in subfolders:
        folder_path = os.path.join(base_dir, folder)
        
        # Check if folder exists to avoid errors
        if not os.path.exists(folder_path):
            print(f"Warning: Folder '{folder_path}' not found. Skipping...")
            continue
            
        print(f"--- Processing folder: {folder} ---")
        
        # Get a list of files first so we don't loop over the new images we create
        # (os.listdir gets a snapshot of the directory)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        
        count = 0
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            
            try:
                with Image.open(file_path) as img:
                    # Separate filename and extension
                    name, ext = os.path.splitext(filename)
                    
                    # 1. Rotate 90 degrees
                    img_90 = img.rotate(90, expand=True)
                    img_90.save(os.path.join(folder_path, f"{name}_rot90{ext}"))
                    
                    # 2. Rotate 180 degrees
                    img_180 = img.rotate(180, expand=True)
                    img_180.save(os.path.join(folder_path, f"{name}_rot180{ext}"))
                    
                    # 3. Rotate 270 degrees
                    img_270 = img.rotate(270, expand=True)
                    img_270.save(os.path.join(folder_path, f"{name}_rot270{ext}"))
                    
                    count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        print(f"Successfully augmented {count} images in '{folder}'.\n")

if __name__ == "__main__":
    # Assuming the script is outside the 'dataset' folder
    dataset_path = 'dataset' 
    augment_data(dataset_path)
    print("Data augmentation complete.")