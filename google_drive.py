"""
Google Drive file downloader for GraphML files
"""
import os
import json
import logging
from pathlib import Path
from googleapiclient.discovery import build
from google.oauth2 import service_account

logger = logging.getLogger(__name__)

# File ID mapping from environment variables
FILE_IDS = {
    'Ashanti_Region_Ghana.graphml': os.environ.get('GOOGLE_DRIVE_ASHANTI_FILE_ID'),
    'North_East_Region_Ghana.graphml': os.environ.get('GOOGLE_DRIVE_NORTH_EAST_FILE_ID'),
    'Northern_Region_Ghana.graphml': os.environ.get('GOOGLE_DRIVE_NORTHERN_FILE_ID'),
    'Savannah_Region_Ghana.graphml': os.environ.get('GOOGLE_DRIVE_SAVANNAH_FILE_ID'),
    'Upper_East_Region_Ghana.graphml': os.environ.get('GOOGLE_DRIVE_UPPER_EAST_FILE_ID'),
    'Upper_West_Region_Ghana.graphml': os.environ.get('GOOGLE_DRIVE_UPPER_WEST_FILE_ID'),
}

def get_drive_service():
    """Initialize Google Drive API service"""
    try:
        credentials_json = os.environ.get('GOOGLE_DRIVE_CREDENTIALS')
        if not credentials_json:
            logger.error("GOOGLE_DRIVE_CREDENTIALS not set")
            return None
        
        creds_info = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(
            creds_info,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=credentials, cache_discovery=False)
        return service
    except Exception as e:
        logger.error(f"Failed to initialize Drive service: {e}")
        return None

def download_file(service, file_id, filepath):
    """Download a file from Google Drive"""
    try:
        request = service.files().get_media(fileId=file_id)
        with open(filepath, 'wb') as f:
            downloader = request.execute()
            f.write(downloader)
        return True
    except Exception as e:
        logger.error(f"Failed to download {file_id}: {e}")
        return False

def download_all_graphml_files():
    """Download all GraphML files from Google Drive"""
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    service = get_drive_service()
    if not service:
        logger.error("Cannot download files - Drive service not available")
        return False
    
    downloaded = 0
    for filename, file_id in FILE_IDS.items():
        if not file_id:
            logger.warning(f"No file ID for {filename}")
            continue
        
        filepath = data_dir / filename
        if filepath.exists():
            logger.info(f"File already exists: {filename}, skipping")
            downloaded += 1
            continue
        
        logger.info(f"Downloading {filename}...")
        if download_file(service, file_id, filepath):
            logger.info(f"Downloaded {filename}")
            downloaded += 1
        else:
            logger.error(f"Failed to download {filename}")
    
    logger.info(f"Downloaded {downloaded}/{len(FILE_IDS)} files")
    return downloaded > 0
