import os
import re
import sys
import time
import requests
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Progress bar for downloads

def recursive_gdown(folder_id, current_path='', max_workers=4, quiet_gdown=False): 
    url = f"https://drive.google.com/embeddedfolderview?id={folder_id}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.content}")
        return
    
    data = response.text

    # Extracting .npz filenames from the HTML
    npz_pattern = r'<div class="flip-entry-title">([^<]*?\.npz)</div>'
    npz_files = re.findall(npz_pattern, data)

    folder_title_match = re.search(r"<title>(.*?)</title>", data)
    folder_title = folder_title_match.group(1) if folder_title_match else "Unknown"

    # Optimized regex patterns to find file links and subfolders
    file_pattern = r"https://drive\.google\.com/file/d/([-\w]{25,})/view"
    folder_pattern = r"https://drive.google.com/drive/folders/([-\w]{25,})"
    
    files = re.findall(file_pattern, data)
    folders = re.findall(folder_pattern, data)
    if len(files) > 0:
        print(f"Found {len(files)} files and {len(folders)} folders in '{folder_title}'")
        print(f"Found {len(npz_files)} .npz files")

    # Create directory for current folder
    path = os.path.join(current_path, folder_title)
    os.makedirs(path, exist_ok=True)

    # Multi-threaded file downloads for .npz files
    def download_file(npz_filename):
        file_path = os.path.join(path, npz_filename)
        
        # Check if the file already exists
        if os.path.exists(file_path):
            print(f"File '{npz_filename}' already exists, skipping...")
            return npz_filename  # Skip downloading
        
        # Construct the gdown download command
        file_url = f"https://drive.google.com/uc?id={files[npz_files.index(npz_filename)]}"
        
        output_redirect = " > nul 2>&1" if os.name == "nt" else " > /dev/null 2>&1" if quiet_gdown else ""
        command = f"gdown {file_url} -O \"{file_path}\"{output_redirect}"
        
        exit_code = os.system(command)

        num_tries = 0
        while exit_code != 0:
            if num_tries > 3:
                print(f'Tried downloading {npz_filename} already {num_tries} times unsuccessfully. Please re-download your cookies.txt and put them in ~/.cache/gdown/')
                exit(1)
            #print(f"Retrying {npz_filename} in 30 seconds...")
            time.sleep(30)
            exit_code = os.system(command)  # Retry downloading
            num_tries += 1

        return npz_filename  # Return filename to update progress bar

    # Progress bar setup
    progress_bar = tqdm(total=len(npz_files), desc="Downloading .npz Files", unit="file")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(download_file, npz_filename): npz_filename for npz_filename in npz_files}
        for future in as_completed(future_to_file):
            future.result()  # Wait for each file download to complete
            progress_bar.update(1)

    progress_bar.close()

    # Recursively process each sub-folder
    for folder_id in folders:
        recursive_gdown(folder_id, path, max_workers, quiet_gdown)

if __name__ == "__main__": 
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <gdrive-folder-id> [max_workers] [--quiet-gdown]")
        exit(1)

    folder_id = sys.argv[1]
    if not re.match(r"^[-\w]{25,}$", folder_id):
        print(f"Invalid ID: {folder_id}")
        exit(1)

    # Get max_workers from user input or default to 4
    max_workers = 4
    quiet_gdown = False

    if len(sys.argv) > 2:
        for arg in sys.argv[2:]:
            if arg.isdigit():
                max_workers = int(arg)
            elif arg == "--quiet-gdown":
                quiet_gdown = True
                
    if max_workers == 0:# Use all cores
        import multiprocessing  # Detect CPU cores
        max_workers = multiprocessing.cpu_count() 

    recursive_gdown(folder_id, "./SegFM3D", max_workers, quiet_gdown)
