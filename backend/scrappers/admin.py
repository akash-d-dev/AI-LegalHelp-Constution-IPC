import json
import os
import logging
import re
from pathlib import Path

# Get the script's directory
SCRIPT_DIR = Path(__file__).parent.absolute()

# Create the logs directory in the script's directory
os.makedirs(SCRIPT_DIR / 'logs', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(SCRIPT_DIR / 'logs' / 'scraper.log'),
        logging.StreamHandler()
    ]
)

class JudgmentIndexer:
    def __init__(self):
        self.data_dir = SCRIPT_DIR / 'data'
        self.index_tracker_file = self.data_dir / 'index_tracker.json'
        self.highest_index = self._load_highest_index()
        logging.info(f"Initialized JudgmentIndexer with highest index: {self.highest_index}")

    def _load_highest_index(self):
        """Load the highest index used from tracker file or start from 0"""
        if self.index_tracker_file.exists():
            try:
                with open(self.index_tracker_file, 'r') as f:
                    data = json.load(f)
                    return data.get('highest_index', 0)
            except json.JSONDecodeError:
                logging.error("Error reading index tracker file")
                return 0
        return 0

    def _save_highest_index(self):
        """Save the current highest index to tracker file"""
        try:
            with open(self.index_tracker_file, 'w') as f:
                json.dump({'highest_index': self.highest_index}, f)
            logging.info(f"Saved highest index: {self.highest_index}")
        except Exception as e:
            logging.error(f"Error saving index tracker: {str(e)}")

    def process_files(self):
        """Process all JSON files in the data directory"""
        json_files = [f for f in self.data_dir.glob('*.json') if f.name != 'index_tracker.json' and f.name != 'progress_tracker.json']
        
        for file_path in json_files:
            try:
                logging.info(f"Processing file: {file_path}")
                self._process_single_file(file_path)
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {str(e)}")

        self._save_highest_index()

    def _process_single_file(self, file_path):
        """Process a single JSON file to add indices to judgments"""
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                judgments = json.load(f)

            if not isinstance(judgments, list):
                logging.warning(f"File {file_path} does not contain a list of judgments")
                return

            modified = False
            for judgment in judgments:
                if not isinstance(judgment, dict):
                    continue

                if 'index' not in judgment:
                    self.highest_index += 1
                    judgment['index'] = self.highest_index
                    modified = True
                else:
                    # Update highest_index if we find a higher one
                    self.highest_index = max(self.highest_index, judgment['index'])

            if modified:
                # Save the updated judgments back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(judgments, f, ensure_ascii=False, indent=2)
                logging.info(f"Updated file {file_path} with new indices")

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            raise

class JudgmentFilter:
    def __init__(self):
        self.data_dir = SCRIPT_DIR / 'data'
        self.min_text_length = 250  # Minimum characters for meaningful judgment
        self.min_sentences = 3      # Minimum sentences for meaningful judgment
        
    def filter_judgments(self, input_file, output_file=None):
        """Filter judgments based on quality criteria - updates file in place"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                judgments = json.load(f)
                
            original_count = len(judgments)
            filtered_judgments = []
            
            for judgment in judgments:
                if self._is_quality_judgment(judgment):
                    filtered_judgments.append(judgment)
                    
            filtered_count = len(filtered_judgments)
            removed_count = original_count - filtered_count
            
            # Save filtered judgments back to the same file
            with open(input_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_judgments, f, ensure_ascii=False, indent=2)
                
            logging.info(f"Filtered {original_count} judgments to {filtered_count} quality judgments")
            logging.info(f"Removed {removed_count} poor quality judgments")
            logging.info(f"Updated file in-place: {input_file}")
            
            return input_file
            
        except Exception as e:
            logging.error(f"Error filtering judgments: {str(e)}")
            return None
    
    def _is_quality_judgment(self, judgment):
        """Check if judgment meets quality criteria"""
        if not isinstance(judgment, dict):
            return False
            
        text = judgment.get('text', '')
        if not text:
            return False
            
        # Check minimum length
        if len(text.strip()) < self.min_text_length:
            return False
            
        # Check minimum sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < self.min_sentences:
            return False
            
        # Check if it contains meaningful legal content
        legal_keywords = ['section', 'ipc', 'convicted', 'sentenced', 'punishment', 'court', 'judgment']
        text_lower = text.lower()
        if not any(keyword in text_lower for keyword in legal_keywords):
            return False
            
        # Check if it's not just a title or header
        if len(text.split()) < 50:  # Less than 50 words
            return False
            
        return True

class TextFormatter:
    def __init__(self):
        self.data_dir = SCRIPT_DIR / 'data'
        
    def format_judgments(self, input_file, output_file=None):
        """Format judgment text with proper spacing - updates file in place"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                judgments = json.load(f)
                
            for judgment in judgments:
                if 'text' in judgment:
                    judgment['text'] = self._format_text(judgment['text'])
                if 'title' in judgment:
                    judgment['title'] = self._format_text(judgment['title'])
                    
            # Save formatted judgments back to the same file
            with open(input_file, 'w', encoding='utf-8') as f:
                json.dump(judgments, f, ensure_ascii=False, indent=2)
                
            logging.info(f"Formatted judgments updated in-place: {input_file}")
            return input_file
            
        except Exception as e:
            logging.error(f"Error formatting judgments: {str(e)}")
            return None
    
    def _format_text(self, text):
        """Format text with proper spacing"""
        if not text:
            return text
            
        # Add space before capital letters (but not at start of sentence)
        text = re.sub(r'(?<!^)(?<!\.\s)(?<!!\s)(?<!\?\s)([A-Z])', r' \1', text)
        
        # Add space before and after numbers when followed/preceded by letters
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
        
        # Fix IPC section references with better spacing
        text = re.sub(r'Section(\d+)', r'Section \1', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+)IPC', r'\1 IPC', text, flags=re.IGNORECASE)
        text = re.sub(r'IPC(\d+)', r'IPC \1', text, flags=re.IGNORECASE)
        
        # Fix common legal abbreviations
        text = re.sub(r'(\d+)CrPC', r'\1 CrPC', text, flags=re.IGNORECASE)
        text = re.sub(r'CrPC(\d+)', r'CrPC \1', text, flags=re.IGNORECASE)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

class IPCExtractor:
    def __init__(self):
        self.data_dir = SCRIPT_DIR / 'data'
        self.tracking_data = {
            'total_judgments': 0,
            'judgments_with_ipc': 0,
            'total_ipc_sections_found': 0,
            'missed_sections': [],
            'found_sections': [],
            'extraction_log': []
        }
        
    def extract_ipc_sections(self, input_file, output_file=None):
        """Extract and update IPC sections in structured_data - updates file in place"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                judgments = json.load(f)
                
            self.tracking_data['total_judgments'] = len(judgments)
            updated_count = 0
            
            print(f"\nProcessing {len(judgments)} judgments for IPC extraction...")
            
            for i, judgment in enumerate(judgments):
                if 'text' in judgment and 'structured_data' in judgment:
                    # Log the judgment being processed
                    judgment_title = judgment.get('title', f'Judgment {i+1}')
                    self.tracking_data['extraction_log'].append(f"Processing: {judgment_title}")
                    
                    ipc_sections = self._extract_ipc_from_text(judgment['text'], judgment_title)
                    if ipc_sections:
                        judgment['structured_data']['ipc_sections'] = ipc_sections
                        updated_count += 1
                        self.tracking_data['judgments_with_ipc'] += 1
                        self.tracking_data['total_ipc_sections_found'] += len(ipc_sections)
                        self.tracking_data['found_sections'].extend(ipc_sections)
                        print(f"  ✓ Judgment {i+1}: Found IPC sections {ipc_sections}")
                    else:
                        # Check if we missed any obvious IPC references
                        missed_sections = self._check_for_missed_sections(judgment['text'], judgment_title)
                        if missed_sections:
                            self.tracking_data['missed_sections'].extend(missed_sections)
                            self.tracking_data['extraction_log'].append(f"  ✗ Missed potential sections: {missed_sections}")
                            print(f"  ✗ Judgment {i+1}: Missed sections {missed_sections}")
                        else:
                            self.tracking_data['extraction_log'].append(f"  - No IPC sections found")
                            print(f"  - Judgment {i+1}: No IPC sections found")
                        
            # Save updated judgments back to the same file
            with open(input_file, 'w', encoding='utf-8') as f:
                json.dump(judgments, f, ensure_ascii=False, indent=2)
                
            # Save tracking report to a separate file
            tracking_file = str(input_file).replace('.json', '_tracking_report.json')
            with open(tracking_file, 'w', encoding='utf-8') as f:
                json.dump(self.tracking_data, f, ensure_ascii=False, indent=2)
                
            logging.info(f"Updated IPC sections for {updated_count} judgments")
            logging.info(f"Total IPC sections found: {self.tracking_data['total_ipc_sections_found']}")
            logging.info(f"Judgments with IPC: {self.tracking_data['judgments_with_ipc']}/{self.tracking_data['total_judgments']}")
            logging.info(f"Updated file in-place: {input_file}")
            logging.info(f"Tracking report saved to: {tracking_file}")
            
            # Print summary to console
            self._print_extraction_summary()
            
            return input_file
            
        except Exception as e:
            logging.error(f"Error extracting IPC sections: {str(e)}")
            print(f"Error: {str(e)}")
            return None
    
    def _extract_ipc_from_text(self, text, judgment_title):
        """Extract IPC sections from text with improved patterns"""
        ipc_sections = set()
        
        # Multiple patterns to catch different formats - updated based on debug results
        patterns = [
            # Basic patterns that work
            r'Section\s+(\d+)',  # Section X (basic) - this catches most cases
            r'Sections\s+(\d+)\s*and\s*(\d+)',  # Sections X and Y
            r'Section\s+(\d+),\s*Indian\s+Penal\s+Code',  # Section X, Indian Penal Code
            r'Section\s+(\d+)\s*Indian\s+Penal\s+Code',  # Section X Indian Penal Code
            r'(\d+)\s*Indian\s+Penal\s+Code',  # X Indian Penal Code
            
            # Standard patterns
            r'Section\s+(\d+)\s*(?:of\s+)?(?:the\s+)?(?:Indian\s+Penal\s+Code|IPC)',
            r'Section\s+(\d+)\s*IPC',
            r'IPC\s+Section\s+(\d+)',
            r'(\d+)\s*IPC\s+Section',
            r'under\s+Section\s+(\d+)\s*IPC',
            r'(\d+)\s*of\s*IPC',
            r'IPC\s+(\d+)',
            r'(\d+)\s*IPC',
            r'Section\s+(\d+)\s*of\s*the\s*Indian\s*Penal\s*Code',
            
            # Additional patterns for better coverage
            r'Sections\s+(\d+),\s*(\d+)',      # Multiple sections with comma
            r'Section\s+(\d+)\s*and\s*(\d+)',  # Section and another
            r'Section\s+(\d+),\s*(\d+)',       # Section, another
            r'(\d+)\s*and\s*(\d+)\s*IPC',      # Numbers and IPC
            r'(\d+),\s*(\d+)\s*IPC',           # Numbers, IPC
            
            # Patterns for specific legal language
            r'convicted\s+under\s+Section\s+(\d+)',
            r'punished\s+under\s+Section\s+(\d+)',
            r'charged\s+under\s+Section\s+(\d+)',
            r'offence\s+under\s+Section\s+(\d+)',
            r'crime\s+under\s+Section\s+(\d+)',
            
            # Patterns for abbreviated forms
            r'Sec\.\s*(\d+)',
            r'Sec\s+(\d+)',
            r'(\d+)\s*Sec\.',
            r'(\d+)\s*Sec\s',
            
            # Patterns for specific IPC references
            r'Indian\s+Penal\s+Code\s+Section\s+(\d+)',
            r'Penal\s+Code\s+Section\s+(\d+)',
            r'Code\s+Section\s+(\d+)',
            
            # Patterns for spaced text (I P C instead of IPC)
            r'(\d+)\s*I\s*P\s*C',  # X I P C (spaced)
            r'I\s*P\s*C\s*(\d+)',  # I P C X (spaced)
            r'Section\s+(\d+)\s*I\s*P\s*C',  # Section X I P C (spaced)
            r'I\s*P\s*C\s*Section\s+(\d+)',  # I P C Section X (spaced)
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle multiple captures (like in "Sections 302 and 307")
                    for section_num in match:
                        section_num = str(section_num).strip()
                        if section_num.isdigit() and 1 <= int(section_num) <= 511:
                            ipc_sections.add(section_num)
                else:
                    # Handle single capture
                    section_num = str(match).strip()
                    if section_num.isdigit() and 1 <= int(section_num) <= 511:
                        ipc_sections.add(section_num)
        
        # Debug logging
        if ipc_sections:
            self.tracking_data['extraction_log'].append(f"  ✓ Found IPC sections: {list(ipc_sections)}")
        else:
            self.tracking_data['extraction_log'].append(f"  - No IPC sections found in text")
            
        return list(ipc_sections)
    
    def _check_for_missed_sections(self, text, judgment_title):
        """Check for obvious IPC references that might have been missed"""
        missed = []
        
        # Look for obvious "Section X" patterns that might have been missed
        obvious_patterns = [
            r'Section\s+(\d+)',
            r'(\d+)\s*IPC',
            r'IPC\s+(\d+)',
        ]
        
        for pattern in obvious_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                section_num = str(match).strip()
                if section_num.isdigit() and 1 <= int(section_num) <= 511:
                    missed.append(section_num)
        
        return list(set(missed))  # Remove duplicates
    
    def _print_extraction_summary(self):
        """Print a summary of the extraction results"""
        print("\n" + "="*60)
        print("IPC EXTRACTION SUMMARY")
        print("="*60)
        print(f"Total judgments processed: {self.tracking_data['total_judgments']}")
        print(f"Judgments with IPC sections: {self.tracking_data['judgments_with_ipc']}")
        print(f"Total IPC sections found: {self.tracking_data['total_ipc_sections_found']}")
        print(f"Success rate: {(self.tracking_data['judgments_with_ipc']/self.tracking_data['total_judgments']*100):.1f}%")
        
        if self.tracking_data['found_sections']:
            print(f"\nMost common IPC sections found:")
            from collections import Counter
            section_counts = Counter(self.tracking_data['found_sections'])
            for section, count in section_counts.most_common(10):
                print(f"  Section {section}: {count} times")
        
        if self.tracking_data['missed_sections']:
            print(f"\nPotentially missed sections:")
            missed_counts = Counter(self.tracking_data['missed_sections'])
            for section, count in missed_counts.most_common(5):
                print(f"  Section {section}: {count} times")
        
        print("="*60)

class AdminManager:
    def __init__(self):
        self.data_dir = SCRIPT_DIR / 'data'
        self.indexer = JudgmentIndexer()
        self.filter = JudgmentFilter()
        self.formatter = TextFormatter()
        self.ipc_extractor = IPCExtractor()
        
    def show_menu(self):
        """Show admin menu and handle user choices"""
        while True:
            print("\n" + "="*50)
            print("JUDGMENT DATA ADMIN PANEL")
            print("="*50)
            print("1. Index all judgment files")
            print("2. Filter poor quality judgments")
            print("3. Format text with proper spacing")
            print("4. Extract and update IPC sections")
            print("5. Run complete data processing pipeline")
            print("6. List all JSON files")
            print("7. Exit")
            print("="*50)
            
            choice = input("Enter your choice (1-7): ").strip()
            
            if choice == '1':
                self.indexer.process_files()
                print("Indexing completed!")
                
            elif choice == '2':
                self._handle_filtering()
                
            elif choice == '3':
                self._handle_formatting()
                
            elif choice == '4':
                self._handle_ipc_extraction()
                
            elif choice == '5':
                self._run_complete_pipeline()
                
            elif choice == '6':
                self._list_files()
                
            elif choice == '7':
                print("Exiting admin panel...")
                break
                
            else:
                print("Invalid choice. Please enter a number between 1-7.")
    
    def _handle_filtering(self):
        """Handle judgment filtering"""
        files = self._get_json_files()
        if not files:
            print("No JSON files found!")
            return
            
        print("\nAvailable files:")
        for i, file in enumerate(files, 1):
            print(f"{i}. {file.name}")
        print(f"{len(files) + 1}. Process ALL files")
            
        try:
            choice = input(f"\nSelect file to filter (1-{len(files) + 1}): ").strip()
            
            if choice == str(len(files) + 1):
                # Process all files
                print(f"\nProcessing all {len(files)} files...")
                processed_files = []
                for file in files:
                    print(f"Processing: {file.name}")
                    output_file = self.filter.filter_judgments(file)
                    if output_file:
                        processed_files.append(output_file)
                        print(f"✓ Completed: {output_file}")
                    else:
                        print(f"✗ Failed: {file.name}")
                
                print(f"\nBatch filtering completed! Processed {len(processed_files)} files.")
                return
                
            choice_num = int(choice) - 1
            if 0 <= choice_num < len(files):
                output_file = self.filter.filter_judgments(files[choice_num])
                if output_file:
                    print(f"Filtering completed! Output: {output_file}")
            else:
                print("Invalid choice!")
        except ValueError:
            print("Please enter a valid number!")
    
    def _handle_formatting(self):
        """Handle text formatting"""
        files = self._get_json_files()
        if not files:
            print("No JSON files found!")
            return
            
        print("\nAvailable files:")
        for i, file in enumerate(files, 1):
            print(f"{i}. {file.name}")
        print(f"{len(files) + 1}. Process ALL files")
            
        try:
            choice = input(f"\nSelect file to format (1-{len(files) + 1}): ").strip()
            
            if choice == str(len(files) + 1):
                # Process all files
                print(f"\nProcessing all {len(files)} files...")
                processed_files = []
                for file in files:
                    print(f"Processing: {file.name}")
                    output_file = self.formatter.format_judgments(file)
                    if output_file:
                        processed_files.append(output_file)
                        print(f"✓ Completed: {output_file}")
                    else:
                        print(f"✗ Failed: {file.name}")
                
                print(f"\nBatch formatting completed! Processed {len(processed_files)} files.")
                return
                
            choice_num = int(choice) - 1
            if 0 <= choice_num < len(files):
                output_file = self.formatter.format_judgments(files[choice_num])
                if output_file:
                    print(f"Formatting completed! Output: {output_file}")
            else:
                print("Invalid choice!")
        except ValueError:
            print("Please enter a valid number!")
    
    def _handle_ipc_extraction(self):
        """Handle IPC section extraction"""
        files = self._get_json_files()
        if not files:
            print("No JSON files found!")
            return
            
        print("\nAvailable files:")
        for i, file in enumerate(files, 1):
            print(f"{i}. {file.name}")
        print(f"{len(files) + 1}. Process ALL files")
            
        try:
            choice = input(f"\nSelect file to extract IPC sections (1-{len(files) + 1}): ").strip()
            
            if choice == str(len(files) + 1):
                # Process all files
                print(f"\nProcessing all {len(files)} files...")
                processed_files = []
                for file in files:
                    print(f"Processing: {file.name}")
                    output_file = self.ipc_extractor.extract_ipc_sections(file)
                    if output_file:
                        processed_files.append(output_file)
                        print(f"✓ Completed: {output_file}")
                    else:
                        print(f"✗ Failed: {file.name}")
                
                print(f"\nBatch IPC extraction completed! Processed {len(processed_files)} files.")
                return
                
            choice_num = int(choice) - 1
            if 0 <= choice_num < len(files):
                output_file = self.ipc_extractor.extract_ipc_sections(files[choice_num])
                if output_file:
                    print(f"IPC extraction completed! Output: {output_file}")
            else:
                print("Invalid choice!")
        except ValueError:
            print("Please enter a valid number!")
    
    def _run_complete_pipeline(self):
        """Run complete data processing pipeline"""
        files = self._get_json_files()
        if not files:
            print("No JSON files found!")
            return
            
        print("\nRunning complete pipeline on all files...")
        
        for file in files:
            print(f"\nProcessing: {file.name}")
            
            # Step 1: Filter (updates file in-place)
            result = self.filter.filter_judgments(file)
            if not result:
                print(f"  ✗ Failed to filter: {file.name}")
                continue
            print(f"  ✓ Filtered: {file.name}")
                
            # Step 2: Format (updates file in-place)
            result = self.formatter.format_judgments(file)
            if not result:
                print(f"  ✗ Failed to format: {file.name}")
                continue
            print(f"  ✓ Formatted: {file.name}")
                
            # Step 3: Extract IPC (updates file in-place)
            result = self.ipc_extractor.extract_ipc_sections(file)
            if result:
                print(f"  ✓ IPC extracted: {file.name}")
            else:
                print(f"  ✗ Failed IPC extraction: {file.name}")
        
        # Step 4: Index all files
        self.indexer.process_files()
        print("\nComplete pipeline finished!")
    
    def _list_files(self):
        """List all JSON files in data directory"""
        files = self._get_json_files()
        if not files:
            print("No JSON files found in data directory!")
            return
            
        print(f"\nFound {len(files)} JSON files:")
        for i, file in enumerate(files, 1):
            size = file.stat().st_size / 1024  # Size in KB
            print(f"{i}. {file.name} ({size:.1f} KB)")
    
    def _get_json_files(self):
        """Get all JSON files in data directory"""
        return [f for f in self.data_dir.glob('*.json') 
                if f.name not in ['index_tracker.json', 'progress_tracker.json']]

def main():
    admin = AdminManager()
    admin.show_menu()

if __name__ == "__main__":
    main()
