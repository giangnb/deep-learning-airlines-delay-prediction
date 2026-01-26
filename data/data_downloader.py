import os
import zipfile
import urllib.request
from pathlib import Path


class DataDownloader:
    """A class to handle downloading and extracting ZIP files."""

    def download_zip(self, url):
        """
        Download a ZIP file from URL, extract it, and clean up.

        Args:
            url (str): The URL of the ZIP file to download.
        """
        # Extract filename from URL
        filename = url.split('/')[-1]
        filepath = os.path.join(os.getcwd(), filename)

        # Download the ZIP file
        print(f"Downloading {filename} from {url}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded to {filepath}")

        # Extract the ZIP file
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(os.getcwd())
        print(f"Extracted successfully")

        # Check if extraction was successful and clean up
        extracted_files = zipfile.ZipFile(filepath, 'r').namelist()
        if extracted_files:
            print(f"Verifying extracted files...")
            # Check if at least one file was extracted
            first_extracted = os.path.join(os.getcwd(), extracted_files[0].split('/')[0])
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

if __name__ == "__main__":
    downloader = DataDownloader()
    downloader.download_sample_2023()