import os

def delete_jpg_files(directory='.'):
    """
    Deletes all .jpg and .jpeg files (case-insensitive) from the specified directory
    and its subdirectories.

    Args:
        directory (str): The path to the directory to process. Defaults to the
                         current directory ('.').
    """
    deleted_count = 0

    print(f"**WARNING: This script will delete .jpg and .jpeg files.**")
    print(f"**Target directory: {os.path.abspath(directory)}**")
    input("Press Enter to continue, or Ctrl+C to cancel...") # Give the user a chance to confirm

    print("-" * 40)
    print("Starting deletion process...")

    for root, _, files in os.walk(directory):
        for filename in files:
            # Check if the file ends with .jpg or .jpeg (case-insensitive)
            if filename.lower().endswith(('.jpg', '.jpeg')):
                file_path = os.path.join(root, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    deleted_count += 1
                except OSError as e:
                    print(f"Error: Could not delete '{file_path}': {e}")

    print("-" * 40)
    print(f"Deletion complete.")
    print(f"Total files deleted: {deleted_count}")

if __name__ == "__main__":
    # --- Configuration ---
    # Set the directory you want to clean.
    # '.' means the script's current directory.
    # Example: folder_to_clean = 'C:/Users/YourUser/Pictures' (Windows)
    # Example: folder_to_clean = '/home/youruser/downloads' (Linux/macOS)
    folder_to_clean = r''
    # --- End Configuration ---

    delete_jpg_files(folder_to_clean)
