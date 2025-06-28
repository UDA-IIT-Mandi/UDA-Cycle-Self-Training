#!/usr/bin/env python3
"""
DCASE TAU 2020 Dataset Processor
Automates the extraction and splitting of DCASE TAU 2020 dataset for unsupervised domain adaptation.

This script:
1. Extracts the main dataset zip file
2. Extracts individual audio and meta zip files
3. Organizes the dataset according to the specified directory structure
4. Splits data into source (Device A) and target (Devices B,C,S1-S6) categories
"""

import os
import zipfile
import pandas as pd
import shutil
from pathlib import Path
import argparse
import logging
from typing import List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DCASETAUProcessor:
    def __init__(self, zip_path: str, output_dir: str = "dcase"):
        """
        Initialize the DCASE TAU 2020 dataset processor.
        
        Args:
            zip_path (str): Path to the main dataset zip file
            output_dir (str): Output directory for processed dataset
        """
        self.zip_path = Path(zip_path)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path("temp_extraction")
        
        # Device mapping for domain adaptation
        self.source_devices = ["a"]  # Device A as source domain
        self.train_target_devices = ["b", "c", "s1", "s2", "s3"]  # Target devices in training set
        self.test_target_devices = ["b", "c", "s1", "s2", "s3", "s4", "s5", "s6"]  # Target devices in test set
        
    def setup_directories(self):
        """Create the required directory structure."""
        directories = [
            self.output_dir,
            self.output_dir / "evaluation_setup",
            self.output_dir / "train" / "source",
            self.output_dir / "train" / "target",
            self.output_dir / "test" / "source",
            self.output_dir / "test" / "target",
            self.temp_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Created directory structure in {self.output_dir}")
    
    def extract_main_zip(self):
        """Extract the main dataset zip file."""
        logger.info(f"Extracting main zip file: {self.zip_path}")
        
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.temp_dir)
            
        logger.info("Main zip file extracted successfully")
    
    def extract_sub_zips(self):
        """Extract all sub zip files (audio.1-16, meta, doc)."""
        logger.info("Extracting sub zip files...")
        
        # Find all zip files in temp directory
        zip_files = list(self.temp_dir.glob("*.zip"))
        
        for zip_file in zip_files:
            logger.info(f"Extracting: {zip_file.name}")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)
                
        logger.info(f"Extracted {len(zip_files)} sub zip files")
    
    def organize_audio_files(self):
        """Collect all audio files from the 16 audio subfolders."""
        logger.info("Organizing audio files...")
        
        audio_files = []
        
        # Look for the main TAU directory
        tau_dirs = list(self.temp_dir.glob("TAU-urban-acoustic-scenes-2020-mobile-development*"))
        
        for tau_dir in tau_dirs:
            audio_dir = tau_dir / "audio"
            if audio_dir.exists():
                wav_files = list(audio_dir.glob("*.wav"))
                audio_files.extend(wav_files)
                logger.info(f"Found {len(wav_files)} audio files in {tau_dir.name}")
        
        logger.info(f"Total audio files found: {len(audio_files)}")
        return audio_files
    
    def copy_documentation(self):
        """Copy README and LICENSE files to output directory."""
        logger.info("Copying documentation files...")
        
        tau_dirs = list(self.temp_dir.glob("TAU-urban-acoustic-scenes-2020-mobile-development*"))
        
        for tau_dir in tau_dirs:
            # Copy README files
            for readme_file in ["README.md", "README.html"]:
                src = tau_dir / readme_file
                if src.exists():
                    shutil.copy2(src, self.output_dir / readme_file)
                    logger.info(f"Copied {readme_file}")
            
            # Copy LICENSE
            license_file = tau_dir / "LICENSE"
            if license_file.exists():
                shutil.copy2(license_file, self.output_dir / "LICENSE")
                logger.info("Copied LICENSE")
    
    def copy_meta_and_evaluation_setup(self):
        """Copy meta.csv and evaluation setup files (fold CSVs)."""
        logger.info("Copying meta.csv and evaluation setup files...")
        
        tau_dirs = list(self.temp_dir.glob("TAU-urban-acoustic-scenes-2020-mobile-development*"))
        
        # Track if we found any evaluation setup directory
        eval_setup_found = False
        
        for tau_dir in tau_dirs:
            # Copy meta.csv to root directory
            meta_file = tau_dir / "meta.csv"
            if meta_file.exists():
                shutil.copy2(meta_file, self.output_dir / "meta.csv")
                logger.info("Copied meta.csv")
            
            # Copy evaluation setup files
            eval_setup_dir = tau_dir / "evaluation_setup"
            if eval_setup_dir.exists():
                eval_setup_found = True
                for csv_file in eval_setup_dir.glob("*.csv"):
                    shutil.copy2(csv_file, self.output_dir / "evaluation_setup" / csv_file.name)
                    logger.info(f"Copied {csv_file.name}")
        
        # Only warn once if no evaluation setup directory was found
        if not eval_setup_found:
            logger.warning("No evaluation_setup directory found in any TAU directory")
    
    def extract_device_from_filename(self, filename: str) -> str:
        """
        Extract device identifier from filename.
        Expected format: scene-location-number1-number2-device.wav
        Example: tram-stockholm-198-5977-a.wav -> 'a'
        """
        # Remove .wav extension and split by '-'
        name_without_ext = filename.replace('.wav', '')
        parts = name_without_ext.split('-')
        
        if len(parts) >= 5:
            # Last part should be the device identifier
            device = parts[-1].lower()
            return device
        
        logger.warning(f"Unexpected filename format: {filename}")
        return None
    
    def split_dataset(self):
        """Split dataset into train/test and source/target based on fold1 files."""
        logger.info("Splitting dataset...")
        
        # Read fold files
        fold_train_path = self.output_dir / "evaluation_setup" / "fold1_train.csv"
        fold_test_path = self.output_dir / "evaluation_setup" / "fold1_test.csv"
        
        if not fold_train_path.exists() or not fold_test_path.exists():
            logger.error("Fold CSV files not found!")
            return
        
        # Read CSV files with proper header handling
        try:
            # First, check if the files have headers by reading a few lines
            with open(fold_train_path, 'r') as f:
                first_line = f.readline().strip()
            
            # If first line looks like a header (contains 'filename' or 'scene_label'), skip it
            has_header = 'filename' in first_line.lower() or 'scene_label' in first_line.lower()
            
            if has_header:
                train_df = pd.read_csv(fold_train_path, sep='\t', header=0)
                test_df = pd.read_csv(fold_test_path, sep='\t', header=0)
                # Ensure column names are consistent
                if len(train_df.columns) >= 2:
                    train_df.columns = ['filename', 'scene_label']
                    test_df.columns = ['filename', 'scene_label']
            else:
                train_df = pd.read_csv(fold_train_path, sep='\t', header=None, names=['filename', 'scene_label'])
                test_df = pd.read_csv(fold_test_path, sep='\t', header=None, names=['filename', 'scene_label'])
            
        except Exception as e:
            logger.warning(f"Error reading CSV with headers, trying without headers: {e}")
            train_df = pd.read_csv(fold_train_path, sep='\t', header=None, names=['filename', 'scene_label'])
            test_df = pd.read_csv(fold_test_path, sep='\t', header=None, names=['filename', 'scene_label'])
        
        # Strip 'audio/' prefix from filenames in CSV if present
        train_df['filename'] = train_df['filename'].str.replace('audio/', '', regex=False)
        test_df['filename'] = test_df['filename'].str.replace('audio/', '', regex=False)
        
        # Filter out any rows where filename is literally 'filename' (header remnants)
        train_df = train_df[train_df['filename'] != 'filename']
        test_df = test_df[test_df['filename'] != 'filename']
        
        # Filter out any rows with NaN filenames
        train_df = train_df.dropna(subset=['filename'])
        test_df = test_df.dropna(subset=['filename'])
        
        logger.info(f"Train files: {len(train_df)}, Test files: {len(test_df)}")
        
        # Get all audio files
        audio_files = self.organize_audio_files()
        audio_dict = {f.name: f for f in audio_files}
        
        logger.info(f"Audio files available: {len(audio_dict)}")
        
        # Process training files
        self._process_split(train_df, audio_dict, "train")
        
        # Process test files
        self._process_split(test_df, audio_dict, "test")
    
    def _process_split(self, df: pd.DataFrame, audio_dict: Dict, split_type: str):
        """Process a specific split (train/test) and organize by source/target."""
        logger.info(f"Processing {split_type} split...")
        
        # Select appropriate target devices based on split type
        if split_type == "train":
            target_devices = self.train_target_devices
        else:  # test
            target_devices = self.test_target_devices
        
        source_count = 0
        target_count = 0
        missing_files = 0
        pattern_mismatch_files = 0
        unknown_device_files = 0
        
        for _, row in df.iterrows():
            filename = row['filename']
            
            # Skip if filename is empty or invalid
            if pd.isna(filename) or filename == '' or filename == 'filename':
                continue
            
            if filename not in audio_dict:
                missing_files += 1
                logger.warning(f"Audio file not found: {filename}")
                continue
            
            # Extract device from filename
            device = self.extract_device_from_filename(filename)
            
            if device is None:
                pattern_mismatch_files += 1
                logger.warning(f"Could not extract device from filename (pattern mismatch): {filename}")
                continue
            
            # Determine source or target
            if device in self.source_devices:
                dest_dir = self.output_dir / split_type / "source"
                source_count += 1
            elif device in target_devices:
                dest_dir = self.output_dir / split_type / "target"
                target_count += 1
            else:
                unknown_device_files += 1
                logger.warning(f"Unknown device '{device}' for file: {filename} in {split_type} set")
                logger.warning(f"Expected devices for {split_type}: source={self.source_devices}, target={target_devices}")
                continue
            
            # Copy file
            src_file = audio_dict[filename]
            dest_file = dest_dir / filename
            shutil.copy2(src_file, dest_file)
        
        logger.info(f"{split_type.capitalize()} split completed:")
        logger.info(f"  Source files: {source_count}")
        logger.info(f"  Target files: {target_count}")
        if missing_files > 0:
            logger.warning(f"  Missing audio files: {missing_files}")
        if pattern_mismatch_files > 0:
            logger.warning(f"  Files with pattern mismatch: {pattern_mismatch_files}")
        if unknown_device_files > 0:
            logger.warning(f"  Files with unknown devices: {unknown_device_files}")
    
    def cleanup_temp(self):
        """Remove temporary extraction directory."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temporary files")
    
    def process(self, cleanup: bool = True):
        """Run the complete processing pipeline."""
        try:
            logger.info("Starting DCASE TAU 2020 dataset processing...")
            
            # Step 1: Setup directories
            self.setup_directories()
            
            # Step 2: Extract main zip
            self.extract_main_zip()
            
            # Step 3: Extract sub zips
            self.extract_sub_zips()
            
            # Step 5: Copy documentation
            self.copy_documentation()
            
            # Step 6: Copy meta.csv and evaluation setup
            self.copy_meta_and_evaluation_setup()
            
            # Step 7: Split dataset
            self.split_dataset()
            
            # Step 8: Cleanup
            if cleanup:
                self.cleanup_temp()
            
            logger.info(f"Dataset processing completed successfully!")
            logger.info(f"Output directory: {self.output_dir.absolute()}")
            
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            raise
    
    def print_dataset_stats(self):
        """Print statistics about the processed dataset."""
        if not self.output_dir.exists():
            logger.error("Output directory does not exist. Run process() first.")
            return
        
        stats = {}
        
        for split in ["train", "test"]:
            stats[split] = {}
            for domain in ["source", "target"]:
                path = self.output_dir / split / domain
                if path.exists():
                    count = len(list(path.glob("*.wav")))
                    stats[split][domain] = count
                else:
                    stats[split][domain] = 0
        
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"{'Split':<10} {'Source':<10} {'Target':<10} {'Total':<10}")
        print("-"*50)
        
        for split in ["train", "test"]:
            source_count = stats[split]["source"]
            target_count = stats[split]["target"]
            total = source_count + target_count
            print(f"{split.capitalize():<10} {source_count:<10} {target_count:<10} {total:<10}")
        
        total_source = stats["train"]["source"] + stats["test"]["source"]
        total_target = stats["train"]["target"] + stats["test"]["target"]
        grand_total = total_source + total_target
        
        print("-"*50)
        print(f"{'Total':<10} {total_source:<10} {total_target:<10} {grand_total:<10}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Process DCASE TAU 2020 dataset for unsupervised domain adaptation")
    parser.add_argument("zip_path", help="Path to the main dataset zip file")
    parser.add_argument("--output_dir", default="dcase", help="Output directory (default: dcase)")
    parser.add_argument("--no_cleanup", action="store_true", help="Keep temporary extraction files")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics after processing")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.zip_path):
        logger.error(f"Zip file not found: {args.zip_path}")
        return
    
    # Process dataset
    processor = DCASETAUProcessor(args.zip_path, args.output_dir)
    processor.process(cleanup=not args.no_cleanup)
    
    # Print statistics if requested
    if args.stats:
        processor.print_dataset_stats()


if __name__ == "__main__":
    main()