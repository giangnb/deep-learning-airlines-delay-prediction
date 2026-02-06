import os
import zipfile
import urllib.request
from pathlib import Path


class DataDownloader:
    """A class to handle downloading and extracting ZIP files."""

    def __init__(self, download_dir="data"):
        """Initialize downloader with a specified download directory.
        
        Args:
            download_dir (str): Directory to save downloads. Defaults to 'data'.
        """
        self.download_dir = download_dir
        # Create directory if it doesn't exist
        os.makedirs(self.download_dir, exist_ok=True)

    def download_zip(self, url):
        """
        Download a ZIP file from URL, extract it, and clean up.

        Args:
            url (str): The URL of the ZIP file to download.
        """
        # Extract filename from URL
       # 1. Prepare file paths
        filename = url.split('/')[-1]
        filepath = os.path.join(self.download_dir, filename)

        # 2. ADD THIS: Create an opener with a User-Agent header
        # This tells the server you are a browser, not a script
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
        urllib.request.install_opener(opener)

        # 3. Download the ZIP file
        print(f"Downloading {filename} from {url}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded to {filepath}")

            # 4. Extract the ZIP file
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(self.download_dir)
            print(f"Extracted successfully")

            # 5. Clean up the ZIP
            os.remove(filepath)
            print(f"ZIP file deleted: {filepath}")
            
        except Exception as e:
            print(f"Error during download/extraction: {e}")
            
        # Download the ZIP file
        print(f"Downloading {filename} from {url}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded to {filepath}")

        # Extract the ZIP file
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(self.download_dir)
        print(f"Extracted successfully")

        # Check if extraction was successful and clean up
        extracted_files = zipfile.ZipFile(filepath, 'r').namelist()
        if extracted_files:
            print(f"Verifying extracted files...")
            # Check if at least one file was extracted
            first_extracted = os.path.join(self.download_dir, extracted_files[0].split('/')[0])
            if os.path.exists(first_extracted):
                print(f"Extraction verified. Deleting ZIP file...")
                os.remove(filepath)
                print(f"ZIP file deleted: {filepath}")
            else:
                print(f"Warning: Could not verify extracted files")
        else:
            print(f"Warning: No files found in ZIP archive")

    def download_sample_2023(self):
        """Download and extract the sample dataset for 2023."""
        url = "https://giangnguyen.id.vn/gbc-data/aasd4010/DelayFlights_2023.zip"
        self.download_zip(url)

    def download_raw(self):
        """Download and extract the raw dataset for all years."""
        url = "https://giangnguyen.id.vn/gbc-data/aasd4010/DelayFlights-raw.zip"
        self.download_zip(url)

    def download_cleaned_handpick(self):
        """Download and extract the raw dataset for all years."""
        url = "https://giangnguyen.id.vn/gbc-data/aasd4010/DelayFlights-cleaned-handpick.zip"
        self.download_zip(url)
        
    def download_cleaned_final(self):
        """Download and extract the raw dataset for all years."""
        url = "https://giangnguyen.id.vn/gbc-data/aasd4010/DelayFlights-cleaned-final.zip"
        self.download_zip(url)

if __name__ == "__main__":
    downloader = DataDownloader()
    downloader.download_cleaned_handpick()
