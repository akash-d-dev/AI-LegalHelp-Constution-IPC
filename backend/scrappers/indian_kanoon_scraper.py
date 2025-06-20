import requests
from bs4 import BeautifulSoup
import time
import json
import os
from datetime import datetime
import logging
from urllib.parse import urljoin, quote
import re

# Create necessary directories
os.makedirs('scrappers/data', exist_ok=True)

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('scrappers/scraper.log'),
        logging.StreamHandler()
    ]
)

class ProgressTracker:
    def __init__(self):
        self.tracker_file = 'scrappers/data/progress_tracker.json'
        self.progress = self._load_progress()
        logging.debug(f"Initialized ProgressTracker with state: {self.progress}")
    
    def _load_progress(self):
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r') as f:
                    progress = json.load(f)
                logging.debug(f"Loaded progress from {self.tracker_file}: {progress}")
                return progress
            except json.JSONDecodeError as e:
                logging.error(f"Error reading progress tracker: {str(e)}")
                return self._get_default_progress()
        logging.debug("No progress file found, starting fresh")
        return self._get_default_progress()
    
    def _get_default_progress(self):
        progress = {
            'current_query_index': 0,
            'current_page': 1,
            'completed_queries': [],
            'total_judgments': 0,
            'last_saved_file': None,
            'start_time': datetime.now().isoformat()
        }
        logging.debug(f"Created default progress: {progress}")
        return progress
    
    def save_progress(self):
        try:
            with open(self.tracker_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
            logging.debug(f"Saved progress to {self.tracker_file}: {self.progress}")
        except Exception as e:
            logging.error(f"Error saving progress: {str(e)}")
    
    def reset_progress(self):
        self.progress = self._get_default_progress()
        self.save_progress()
        logging.info("Progress tracker has been reset")
    
    def update_progress(self, query_index, page, judgments_count, saved_file):
        old_progress = self.progress.copy()
        self.progress['current_query_index'] = query_index
        self.progress['current_page'] = page
        self.progress['total_judgments'] = judgments_count
        self.progress['last_saved_file'] = saved_file
        self.save_progress()
        logging.debug(f"Updated progress: {old_progress} -> {self.progress}")
    
    def mark_query_completed(self, query):
        if query not in self.progress['completed_queries']:
            self.progress['completed_queries'].append(query)
            self.save_progress()
            logging.debug(f"Marked query as completed: {query}")

class IndianKanoonScraper:
    def __init__(self):
        self.base_url = "https://indiankanoon.org"
        self.search_url = "https://indiankanoon.org/search/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.progress_tracker = ProgressTracker()
        logging.debug(f"Initialized IndianKanoonScraper with headers: {self.headers}")
        
    def search_judgments(self, query, max_pages=10, start_page=1):
        """
        Search for judgments based on query
        """
        judgments = []
        page = start_page
        
        while page <= max_pages:
            try:
                # URL encode the query
                encoded_query = quote(query)
                url = f"{self.search_url}?formInput={encoded_query}&pagenum={page}"
                logging.debug(f"Fetching URL: {url}")
                
                response = self.session.get(url)
                response.raise_for_status()
                
                # Debug response
                logging.debug(f"Response status code: {response.status_code}")
                logging.debug(f"Response headers: {dict(response.headers)}")
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Debug HTML content
                logging.debug(f"Page title: {soup.title.string if soup.title else 'No title found'}")
                
                # Find all result divs
                results = soup.find_all('div', class_='result')
                logging.debug(f"Found {len(results)} result divs on page {page}")
                
                if not results:
                    logging.warning(f"No results found on page {page}. HTML content: {soup.prettify()[:500]}...")
                    break
                    
                for i, result in enumerate(results):
                    logging.debug(f"Processing result {i+1}/{len(results)} on page {page}")
                    judgment_data = self._extract_judgment_data(result)
                    if judgment_data:
                        # Only include judgments that contain sentencing information
                        if self._contains_sentencing_info(judgment_data['text']):
                            judgments.append(judgment_data)
                            logging.info(f"Found judgment with sentencing info: {judgment_data['title']}")
                        else:
                            logging.debug(f"Skipping judgment without sentencing info: {judgment_data['title']}")
                    else:
                        logging.warning(f"Failed to extract judgment data from result {i+1}")
                
                logging.info(f"Processed page {page}, found {len(results)} results, extracted {len(judgments)} judgments with sentencing info")
                page += 1
                time.sleep(3)  # Increased delay to be more respectful
                
            except Exception as e:
                logging.error(f"Error processing page {page}: {str(e)}", exc_info=True)
                break
                
        return judgments
    
    def _contains_sentencing_info(self, text):
        """
        Check if the judgment text contains sentencing information
        """
        sentencing_keywords = [
            'sentenced to', 'imprisonment', 'rigorous imprisonment', 'simple imprisonment',
            'life imprisonment', 'death penalty', 'hanged', 'fine of', 'rupees',
            'years', 'months', 'days', 'convicted', 'punishment', 'sentence',
            'Section 302', 'Section 304', 'Section 307', 'Section 376', 'Section 420',
            'murder', 'attempt to murder', 'rape', 'cheating', 'theft', 'robbery',
            'dacoity', 'kidnapping', 'assault', 'hurt', 'grievous hurt'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in sentencing_keywords)
    
    def _extract_judgment_data(self, result_div):
        """
        Extract judgment data from a result div
        """
        try:
            # Log the HTML structure of the result div
            logging.debug(f"Result div HTML structure:")
            logging.debug(f"Classes: {result_div.get('class', [])}")
            logging.debug(f"Direct children: {[child.name for child in result_div.children if hasattr(child, 'name')]}")
            
            # Find all links in the result div
            all_links = result_div.find_all('a')
            logging.debug(f"Found {len(all_links)} links in result div")
            for link in all_links:
                logging.debug(f"Link: {link.get('href')} - Text: {link.get_text(strip=True)} - Classes: {link.get('class', [])}")
            
            # Find the title link - it's in a div with class 'result_title'
            title_div = result_div.find('div', class_='result_title')
            if title_div:
                logging.debug(f"Found title div with classes: {title_div.get('class', [])}")
                title_link = title_div.find('a')
                if title_link:
                    logging.debug(f"Found title link: {title_link.get('href')} - Text: {title_link.get_text(strip=True)}")
                else:
                    logging.warning("No title link found in title div")
            else:
                logging.warning("No title div found in result")
                return None
            
            if not title_link:
                logging.warning("No title link found in result div")
                return None
            
            title = title_link.get_text(strip=True)
            url = urljoin(self.base_url, title_link['href'])
            
            # Get the judgment text
            text = self._get_judgment_text(url)
            if not text:
                return None
            
            # Extract date if available
            date = None
            date_div = result_div.find('div', class_='result_date')
            if date_div:
                date_text = date_div.get_text(strip=True)
                date = self._extract_date(date_text)
            
            # Extract additional structured data
            structured_data = self._extract_structured_data(text, title)
            
            return {
                'title': title,
                'text': text,
                'date': date,
                'url': url,
                'court': 'Supreme Court',
                'timestamp': datetime.now().isoformat(),
                'structured_data': structured_data
            }
            
        except Exception as e:
            logging.error(f"Error extracting judgment data: {str(e)}", exc_info=True)
            return None
    
    def _extract_structured_data(self, text, title):
        """
        Extract structured data from judgment text for punishment prediction
        """
        structured_data = {
            'ipc_sections': [],
            'charges': [],
            'sentence': {},
            'case_facts': '',
            'aggravating_factors': [],
            'mitigating_factors': [],
            'court_level': 'Supreme Court'
        }
        
        # Extract IPC sections
        ipc_pattern = r'Section\s+(\d+)\s*(?:of\s+)?(?:the\s+)?(?:Indian\s+Penal\s+Code|IPC)'
        ipc_matches = re.findall(ipc_pattern, text, re.IGNORECASE)
        structured_data['ipc_sections'] = list(set(ipc_matches))
        
        # Extract sentencing information
        sentence_patterns = [
            r'sentenced\s+to\s+(\d+)\s*(?:years?|months?|days?)\s*(?:rigorous|simple)?\s*imprisonment',
            r'(\d+)\s*(?:years?|months?|days?)\s*(?:rigorous|simple)?\s*imprisonment',
            r'life\s+imprisonment',
            r'death\s+penalty',
            r'fine\s+of\s+[Rr]s\.?\s*(\d+(?:,\d+)*)',
            r'[Rr]s\.?\s*(\d+(?:,\d+)*)\s*as\s+fine'
        ]
        
        for pattern in sentence_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if 'life imprisonment' in pattern.lower():
                    structured_data['sentence']['type'] = 'life imprisonment'
                elif 'death penalty' in pattern.lower():
                    structured_data['sentence']['type'] = 'death penalty'
                elif 'fine' in pattern.lower():
                    structured_data['sentence']['fine'] = matches[0]
                else:
                    structured_data['sentence']['duration'] = matches[0]
        
        # Extract case facts (first few sentences)
        sentences = text.split('.')
        if len(sentences) > 3:
            structured_data['case_facts'] = '. '.join(sentences[:3]) + '.'
        
        # Extract aggravating factors
        aggravating_keywords = ['weapon', 'deadly weapon', 'firearm', 'knife', 'lathi', 'premeditated', 
                              'brutal', 'cruel', 'inhuman', 'heinous', 'serious injury', 'fatal']
        for keyword in aggravating_keywords:
            if keyword.lower() in text.lower():
                structured_data['aggravating_factors'].append(keyword)
        
        # Extract mitigating factors
        mitigating_keywords = ['sudden quarrel', 'heat of passion', 'provocation', 'no premeditation',
                             'first offender', 'remorse', 'cooperation', 'family circumstances']
        for keyword in mitigating_keywords:
            if keyword.lower() in text.lower():
                structured_data['mitigating_factors'].append(keyword)
        
        return structured_data
    
    def _get_judgment_text(self, url):
        """
        Get the full text of a judgment
        """
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Log the HTML structure for debugging
            logging.debug(f"HTML structure for {url}:")
            logging.debug(f"Title: {soup.title.string if soup.title else 'No title'}")
            
            # Log all divs with their classes
            all_divs = soup.find_all('div', class_=True)
            logging.debug(f"All divs with classes:")
            for div in all_divs:
                logging.debug(f"Div class: {div.get('class')} - First 100 chars: {div.get_text(strip=True)[:100]}")
            
            # Get all text content, preserving structure
            text_parts = []
            
            # Get the title from the document title
            if soup.title:
                text_parts.append(soup.title.string.strip())
            
            # Find the expanded headline and fragments
            expanded_headline = soup.find('div', class_='expanded_headline')
            if expanded_headline:
                text_parts.append(expanded_headline.get_text(strip=True))
            
            # Get all fragments
            fragments = soup.find_all('div', class_='fragment')
            for fragment in fragments:
                text = fragment.get_text(strip=True)
                if text:  # Only add non-empty fragments
                    text_parts.append(text)
            
            if not text_parts:
                logging.warning(f"No text content found in judgment at URL: {url}")
                return None
            
            text = '\n\n'.join(text_parts)
            logging.debug(f"Extracted judgment text length: {len(text)} characters")
            return text
            
        except Exception as e:
            logging.error(f"Error getting judgment text from {url}: {str(e)}", exc_info=True)
            return None
    
    def _extract_date(self, date_text):
        """
        Extract date from date text
        """
        try:
            return date_text.strip()
        except Exception as e:
            logging.error(f"Error extracting date: {str(e)}", exc_info=True)
            return None
    
    def save_judgments(self, judgments, filename):
        """
        Save judgments to a JSON file
        """
        try:
            filepath = os.path.join('scrappers/data', filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(judgments, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved {len(judgments)} judgments to {filepath}")
            return filepath
        except Exception as e:
            logging.error(f"Error saving judgments: {str(e)}", exc_info=True)
            return None

def main():
    scraper = IndianKanoonScraper()
    
    # Check if we should resume or restart
    if os.path.exists(scraper.progress_tracker.tracker_file):
        print("\nPrevious scraping session found!")
        print(f"Total judgments collected: {scraper.progress_tracker.progress['total_judgments']}")
        print(f"Last saved file: {scraper.progress_tracker.progress['last_saved_file']}")
        
        while True:
            choice = input("\nDo you want to:\n1. Resume from last checkpoint\n2. Start fresh\nEnter choice (1/2): ").strip()
            if choice in ['1', '2']:
                break
            print("Invalid choice. Please enter 1 or 2.")
        
        if choice == '2':
            scraper.progress_tracker.reset_progress()
    
    # Targeted queries for punishment prediction - focusing on specific IPC sections and sentencing
    queries = [
        # Major criminal offenses with clear sentencing patterns
        "Supreme Court Section 302 IPC murder sentence",
        "Supreme Court Section 304 IPC culpable homicide sentence", 
        "Supreme Court Section 307 IPC attempt to murder sentence",
        "Supreme Court Section 376 IPC rape sentence",
        "Supreme Court Section 420 IPC cheating sentence",
        "Supreme Court Section 379 IPC theft sentence",
        "Supreme Court Section 392 IPC robbery sentence",
        "Supreme Court Section 395 IPC dacoity sentence",
        "Supreme Court Section 363 IPC kidnapping sentence",
        "Supreme Court Section 323 IPC hurt sentence",
        "Supreme Court Section 325 IPC grievous hurt sentence",
        
        # Specific sentencing keywords
        "Supreme Court life imprisonment sentence",
        "Supreme Court death penalty sentence",
        "Supreme Court rigorous imprisonment sentence",
        "Supreme Court fine punishment sentence",
        
        # Case outcome focused queries
        "Supreme Court convicted sentenced imprisonment",
        "Supreme Court punishment awarded years",
        "Supreme Court sentence reduced appeal",
        "Supreme Court sentence enhanced appeal",
        
        # Specific crime types with sentencing
        "Supreme Court murder case sentence punishment",
        "Supreme Court rape case sentence punishment", 
        "Supreme Court robbery case sentence punishment",
        "Supreme Court kidnapping case sentence punishment",
        "Supreme Court assault case sentence punishment",
        "Supreme Court theft case sentence punishment",
        
        # Aggravating and mitigating factors
        "Supreme Court weapon used sentence enhanced",
        "Supreme Court sudden quarrel sentence reduced",
        "Supreme Court premeditated murder sentence",
        "Supreme Court first offender sentence lenient"
    ]
    
    all_judgments = []
    start_index = scraper.progress_tracker.progress['current_query_index']
    start_page = scraper.progress_tracker.progress['current_page']
    
    try:
        for i, query in enumerate(queries[start_index:], start=start_index):
            if query in scraper.progress_tracker.progress['completed_queries']:
                logging.info(f"Skipping completed query: {query}")
                continue
                
            logging.info(f"Starting search for: {query}")
            judgments = scraper.search_judgments(query, max_pages=20, start_page=start_page)
            logging.info(f"Found {len(judgments)} judgments with sentencing info for query: {query}")
            
            all_judgments.extend(judgments)
            
            # Save after each query in case of interruption
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            saved_file = f"supreme_court_sentencing_judgments_{timestamp}.json"
            filepath = scraper.save_judgments(all_judgments, saved_file)
            
            if filepath:
                scraper.progress_tracker.update_progress(
                    query_index=i + 1,
                    page=1,  # Reset page for next query
                    judgments_count=len(all_judgments),
                    saved_file=filepath
                )
                scraper.progress_tracker.mark_query_completed(query)
            
            start_page = 1  # Reset page counter for next query
            time.sleep(5)  # Be nice to the server between queries
            
    except KeyboardInterrupt:
        print("\nScraping interrupted by user. Progress has been saved.")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
    finally:
        print(f"\nScraping completed. Total judgments with sentencing info collected: {len(all_judgments)}")
        print(f"Last saved file: {scraper.progress_tracker.progress['last_saved_file']}")

if __name__ == "__main__":
    main() 