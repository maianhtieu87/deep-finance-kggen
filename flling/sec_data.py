
# -*- coding: utf-8 -*-
"""
SEC Filing Scraper with Content Extraction for HTML and XBRL
Based on Document class from secedgartext
@author: AdamGetbags
Modified by: Grok
Further developed by: Phong Ngo
Improved by: Grok (based on secedgartext Document class)
"""

import requests
import pandas as pd
import logging
import os
import time
from tqdm import tqdm
from bs4 import BeautifulSoup, NavigableString, Tag, Comment
import re
import polars as pl
from datetime import datetime
import pytz
from concurrent.futures import ThreadPoolExecutor
from lxml import etree
from abc import ABCMeta
from statistics import median
import warnings
import copy
import json
import argparse

warnings.filterwarnings("ignore", category=FutureWarning)

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(os.path.join("data", "03_primary", "filing_fails.log"), mode="w")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("Program starts")

# Create request header
headers = {'User-Agent': "phongngo4060@gmail.com"}

# Create output directory
os.makedirs(os.path.join("data", "03_primary"), exist_ok=True)

# Define date range for filtering
START_DATE = "2023-01-15"
END_DATE = "2025-04-25"

# Thiết lập argparse
parser = argparse.ArgumentParser(description='SEC Filing Scraper')
parser.add_argument('--documents', type=str, help='Comma-separated list of document types (e.g., 10-K,10-Q)')
args = parser.parse_args()

# Đọc và xử lý search terms từ JSON
try:
    project_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    project_dir = os.getcwd()
json_path = os.path.join(project_dir, 'document_group_section_search.json')
try:
    with open(json_path, 'r', encoding='utf-8') as f:
        json_text = f.read()
        try:
            search_terms = json.loads(json_text)
            if not search_terms or not isinstance(search_terms, dict):
                logger.error(f'Search terms file is missing or invalid: {json_path}')
                raise SystemExit(1)
        except json.JSONDecodeError as e:
            logger.error(f'Invalid JSON format in file: {json_path}, Error: {e}')
            raise SystemExit(1)
except FileNotFoundError:
    logger.error(f'Search terms file not found: {json_path}')
    raise SystemExit(1)

# Xử lý regex trong search_terms chỉ cho HTML
search_terms_regex = copy.deepcopy(search_terms)
for filing in search_terms:
    for idx, section in enumerate(search_terms[filing]):
        if not isinstance(section, dict) or 'html' not in section:
            logger.warning(f"Invalid section structure for filing {filing}, index {idx}")
            continue
        for idx2, pattern in enumerate(section['html']):
            if not isinstance(pattern, dict) or not all(k in pattern for k in ['start', 'end']):
                logger.warning(f"Invalid pattern structure for filing {filing}, section {idx}, pattern {idx2}")
                continue
            for startend in ['start', 'end']:
                regex_string = pattern[startend]
                if not isinstance(regex_string, str):
                    logger.warning(f"Invalid regex string for {startend} in filing {filing}, section {idx}, pattern {idx2}")
                    continue
                regex_string = regex_string.replace('_', '\\s{,5}')
                regex_string = regex_string.replace('\n', '\\n')
                search_terms_regex[filing][idx]['html'][idx2][startend] = regex_string

logger.info(f"SEARCH_PATTERNS: {json.dumps(search_terms_regex, indent=2)}")
SEARCH_PATTERNS = search_terms_regex

# Xác định loại tài liệu
args.documents = args.documents or ','.join(list(search_terms.keys()))
args.documents = re.split(',', args.documents)
logger.info(f"Loaded search patterns for document types: {', '.join(args.documents)}")

# Define CIK list
ciks = pd.DataFrame({
    'cik': [1318605, 1065280, 789019, 1018724],  # TSLA, NFLX, MSFT, AMZN
    'ticker': ['TSLA', 'NFLX', 'MSFT', 'AMZN']
})

# Convert 'cik' to string and pad with zeros to 10 digits
ciks['cik'] = ciks['cik'].astype(str).str.zfill(10)

# Print CIKs
print("CIK DataFrame:")
print(ciks)

# Metadata class to store filing metadata
class Metadata:
    def __init__(self, cik, ticker, form_type, filed_at, accession_number, document_url):
        self.metadata_file_name = f"{cik}_{accession_number.replace('-', '')}"
        self.cik = cik
        self.ticker = ticker
        self.form_type = form_type
        self.filed_at = filed_at
        self.accession_number = accession_number
        self.sec_index_url = document_url
        self.section_name = None
        self.extraction_method = None
        self.endpoints = [None, None]
        self.warnings = []
        self.time_elapsed = 0
        self.section_end_time = None
        self.section_n_characters = 0
        self.output_file = None

    def save_to_json(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vars(self), f, indent=2)

    def save_to_db(self):
        logger.info(f"Would save metadata to DB for {self.metadata_file_name}")

# Abstract Document class
class Document(metaclass=ABCMeta):
    def __init__(self, file_path, doc_text, extraction_method, metadata):
        self._file_path = file_path
        self.doc_text = doc_text
        self.extraction_method = extraction_method
        self.metadata = metadata
        self.log_cache = []
        self.extracted_content = {}  # Store content by section_name

    def get_excerpt(self, form_type, skip_existing_excerpts=False):
        start_time = time.process_time()
        self.prepare_text()
        prep_time = time.process_time() - start_time
        file_name_root = self.metadata.metadata_file_name
        for section_search_terms in SEARCH_PATTERNS.get(form_type, []):
            start_time = time.process_time()
            metadata = copy.copy(self.metadata)
            warnings = []
            section_name = section_search_terms['itemname']
            section_output_path = os.path.join("data", "03_primary","text_files",f"{file_name_root}_{section_name}")
            txt_output_path = f"{section_output_path}_excerpt.txt"
            metadata_path = f"{section_output_path}_metadata.json"
            failure_metadata_output_path = f"{section_output_path}_failure.json"

            search_pairs = section_search_terms['html']
            text_extract, extraction_summary, start_text, end_text, warnings = self.extract_section(search_pairs)
            time_elapsed = time.process_time() - start_time
            metadata.section_name = section_name
            metadata.extraction_method = self.extraction_method
            if start_text:
                start_text = start_text.replace('"', "'")
            if end_text:
                end_text = end_text.replace('"', "'")
            metadata.endpoints = [start_text, end_text]
            metadata.warnings = warnings
            metadata.time_elapsed = round(prep_time + time_elapsed, 1)
            metadata.section_end_time = str(datetime.utcnow())
            
            if text_extract:
                metadata.section_n_characters = len(text_extract)
                with open(txt_output_path, 'w', encoding='utf-8', newline='\n') as txt_output:
                    txt_output.write(text_extract)
                self.log_cache.append(('DEBUG', f"SUCCESS: Saved file for {section_name} at {txt_output_path}"))
                self.extracted_content[section_name] = text_extract
                try:
                    os.remove(failure_metadata_output_path)
                except:
                    pass
                metadata.output_file = txt_output_path
                metadata.save_to_json(metadata_path)
            else:
                self.log_cache.append(('WARNING', f"No excerpt located for {section_name} at {metadata.sec_index_url}"))
                try:
                    os.remove(metadata_path)
                except:
                    pass
                metadata.save_to_json(failure_metadata_output_path)
            metadata.save_to_db()
        return self.log_cache

    def prepare_text(self):
        pass

    def extract_section(self, search_pairs):
        pass

# HTML Document class
class HtmlDocument(Document):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.soup = None
        self.plaintext = None

    def search_terms_type(self):
        return "html"

    def prepare_text(self):
        html_text = self.doc_text
        html_text = re.sub(r'<\s', '<', html_text)
        html_text = re.sub(r'(<small>|</small>)', '', html_text, flags=re.IGNORECASE)
        html_text = re.sub(r'(\nITEM\s{1,10}[1-9])', r'<br>\1', html_text, flags=re.IGNORECASE)

        start_time = time.process_time()
        try:
            self.soup = BeautifulSoup(html_text, 'lxml')
        except:
            self.soup = BeautifulSoup(html_text, 'html.parser')
        parsing_time = time.process_time() - start_time
        self.log_cache.append(('DEBUG', f"Parsing time: {parsing_time:.2f}s; {len(html_text):,} characters; {len(self.soup.find_all()):,} HTML elements"))

        if len(html_text) / len(self.soup.find_all()) > 500:
            html_text = re.sub(r'\n\n', r'<br>', html_text, flags=re.IGNORECASE)
            self.soup = BeautifulSoup(html_text, 'html.parser')

        for table in self.soup.find_all('table'):
            if self.should_remove_table(table):
                json_str = self.table_to_json(table)
                table.replace_with(json_str)

        document_string = ''
        all_paras = []
        is_in_a_paragraph = True
        ec = self.soup.find()
        while ec:
            if self.is_line_break(ec) or ec.next_element is None:
                if is_in_a_paragraph:
                    is_in_a_paragraph = False
                    all_paras.append(document_string.strip())
                    document_string = document_string + '\n\n'
            else:
                if isinstance(ec, NavigableString) and not isinstance(ec, Comment):
                    ecs = re.sub(r'\s+', ' ', ec.string.strip())
                    if len(ecs) > 0:
                        if not is_in_a_paragraph:
                            is_in_a_paragraph = True
                            document_string += ecs
                        else:
                            document_string += ' ' + ecs
            ec = ec.next_element

        document_string = re.sub(r'\n\s+\n', '\n\n', document_string)
        document_string = re.sub(r'\n{3,}', '\n\n', document_string)
        self.plaintext = document_string.strip()

    def extract_section(self, search_pairs):
        start_text = 'na'
        end_text = 'na'
        warnings = []
        text_extract = None
        longest_text_length = 0
        for st in search_pairs:
            pattern = st['start'] + r'[\s\S]*?' + st['end']
            logger.debug(f"Trying regex pattern: {pattern}")
            item_search = re.findall(pattern, self.plaintext, re.DOTALL | re.IGNORECASE)
            if item_search:
                logger.debug(f"Found {len(item_search)} matches for pattern: {pattern}")
                for s in item_search:
                    if isinstance(s, tuple):
                        self.log_cache.append(('ERROR', "Groups found in Regex, please correct"))
                        logger.error(f"Regex groups found in pattern: {pattern}")
                    if len(s) > longest_text_length:
                        text_extract = s.strip()
                        longest_text_length = len(s)
                if text_extract:
                    final_text_lines = text_extract.split('\n')
                    start_text = final_text_lines[0][:100]
                    end_text = final_text_lines[-1][:100]
                    logger.debug(f"Extracted text starts with: {start_text}")
                    logger.debug(f"Extracted text ends with: {end_text}")
                    break
            else:
                logger.debug(f"No matches for pattern: {pattern}")
        extraction_summary = self.extraction_method + '_document'
        if not text_extract:
            warnings.append(f'Extraction did not work for HTML file with patterns: {search_pairs}')
            extraction_summary = self.extraction_method + '_document: failed'
            logger.warning(f"Extraction failed for patterns: {search_pairs}")
        else:
            text_extract = re.sub(r'\n\s{,5}Table of Contents\n', '', text_extract, flags=re.IGNORECASE)
        return text_extract, extraction_summary, start_text, end_text, warnings

    def should_remove_table(self, html):
        char_counts = [len(t) for t in html.stripped_strings if len(t) > 0]
        contains_item = any(re.search(r'ITEM\s*\d+[A-Z]?', t, re.IGNORECASE) for t in html.stripped_strings)
        return len(char_counts) > 5 and median(char_counts) < 30 and not contains_item if char_counts else False
    
    def table_to_json(self, table, use_headers=True):
        rows = []
        headers = []
        json_list = []
        
        if use_headers:
            header_row = table.find('tr')
            if header_row:
                headers = [header.get_text().strip() for header in header_row.find_all('th') if header.get_text().strip()]
        
        for row in table.find_all('tr'):
            cells = [cell.get_text().strip() for cell in row.find_all(['td', 'th']) if cell.get_text().strip()]
            if cells:
                if use_headers and headers and len(cells) == len(headers):
                    row_data = dict(zip(headers, cells))
                    rows.append(row_data)
                else:
                    rows.append(cells)
        
        for idx, row in enumerate(rows):
            if isinstance(row, dict):
                json_list.append({"row": idx, "data": row})
            else:
                json_list.append({"row": idx, "data": row})
        
        return json.dumps(json_list, ensure_ascii=False, indent=4) if json_list else json.dumps([])

    def is_line_break(self, e):
        is_block_tag = e.name in ['p', 'div', 'br', 'hr', 'tr', 'table', 'form', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        if is_block_tag and e.parent.name == 'td':
            if len(e.parent.findChildren(name=e.name)) == 1:
                is_block_tag = False
        is_block_style = False
        if hasattr(e, 'attrs') and 'style' in e.attrs:
            is_block_style = bool(re.search(r'margin-(top|bottom)', e['style']))
        return is_block_tag or is_block_style

# Function to handle rate limiting
def safe_get(url, retries=3, backoff=1, timeout=30):
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 429:
                logger.warning(f"Rate limit hit for {url}. Retrying after {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"Attempt {attempt+1} failed for {url}: {e}")
            time.sleep(backoff)
            backoff *= 2
    logger.error(f"Failed to fetch {url} after {retries} attempts")
    return None

# Fetch filing metadata for each CIK
all_filings = []
for cik in tqdm(ciks['cik'], desc="Fetching filings"):
    try:
        response = safe_get(f'https://data.sec.gov/submissions/CIK{cik}.json')
        if response is None:
            continue
        
        data = response.json()
        filings = data.get('filings', {}).get('recent', {})
        if not filings:
            logger.info(f"No filings found for CIK {cik}")
            continue
        
        filings_df = pd.DataFrame(filings)
        filings_df = filings_df[filings_df['form'].isin(['10-K', '10-Q'])]
        if filings_df.empty:
            logger.info(f"No 10-K or 10-Q filings found for CIK {cik}")
            continue
        
        filings_df['filingDate'] = pd.to_datetime(filings_df['filingDate'], errors='coerce')
        filings_df = filings_df[
            (filings_df['filingDate'] >= START_DATE) & 
            (filings_df['filingDate'] <= END_DATE)
        ]
        if filings_df.empty:
            logger.info(f"No filings within date range {START_DATE} to {END_DATE} for CIK {cik}")
            continue
        
        filings_df['cik'] = cik
        filings_df['ticker'] = ciks[ciks['cik'] == cik]['ticker'].iloc[0]
        filings_df = filings_df[['cik', 'ticker', 'form', 'filingDate', 'accessionNumber', 'primaryDocument']]
        filings_df['filingUrl'] = filings_df.apply(
            lambda x: f"https://www.sec.gov/Archives/edgar/data/{cik}/{x['accessionNumber'].replace('-', '')}/{x['primaryDocument']}",
            axis=1
        )
        
        all_filings.append(filings_df)
        logger.info(f"Successfully fetched {len(filings_df)} filings for CIK {cik}")
        
    except Exception as e:
        logger.error(f"Error fetching filings for CIK {cik}: {e}")
        continue

# Combine all filings
if all_filings:
    filings_df = pd.concat(all_filings, ignore_index=True)
    filings_df = filings_df.rename(columns={
        'form': 'formType',
        'filingDate': 'filedAt',
        'filingUrl': 'document_url'
    })
    filings_df.to_parquet(os.path.join("data", "03_primary", "new_filing_metadata.parquet"))
    logger.info(f"Saved {len(filings_df)} filings to new_filing_metadata.parquet")
    print("\nSample of fetched filings:")
    print(filings_df.head())
else:
    logger.info("No filings fetched")
    print("No filings fetched")

def process_filing(url, form_type, cik, ticker, filed_at, accession_number):
    logger.debug(f"Starting to process URL: {url}, Form: {form_type}, CIK: {cik}")
    try:
        response = safe_get(url)
        if response is None:
            logger.error(f"Failed to fetch URL: {url}")
            return []
        
        content_type = response.headers.get('Content-Type', '').lower()
        metadata = Metadata(cik, ticker, form_type, filed_at, accession_number, url)
        extracted_sections = []

        if 'html' in content_type or url.endswith(('.htm', '.html')):
            extraction_method = 'html'
            doc = HtmlDocument(url, response.text, extraction_method, metadata)
            logs = doc.get_excerpt(form_type)
            for level, msg in logs:
                logger.log(getattr(logging, level), msg)
            for section_name, content in doc.extracted_content.items():
                extracted_sections.append({
                    'section_name': section_name,
                    'content': content,
                    'output_file': f"{os.path.join('data', '03_primary', f'{metadata.metadata_file_name}_{section_name}_excerpt.txt')}"
                })
            logger.debug(f"Extracted {len(extracted_sections)} sections for URL {url}")

        elif 'xml' in content_type or url.endswith('.xml'):
            extraction_method = 'xbrl'
            try:
                parser = etree.XMLParser(recover=True, remove_blank_text=True)
                tree = etree.fromstring(response.content, parser)
                content = ' '.join([elem.text for elem in tree.iter() if elem.text and len(elem.text.strip()) > 50])
                if not content:
                    logger.info(f"No narrative content found in XBRL for URL {url}")
                    return []
                doc = HtmlDocument(url, content, extraction_method, metadata)
                logs = doc.get_excerpt(form_type)
                for level, msg in logs:
                    logger.log(getattr(logging, level), msg)
                for section_name, content in doc.extracted_content.items():
                    extracted_sections.append({
                        'section_name': section_name,
                        'content': content,
                        'output_file': f"{os.path.join('data', '03_primary', f'{metadata.metadata_file_name}_{section_name}_excerpt.txt')}"
                    })
                logger.debug(f"Extracted {len(extracted_sections)} sections for URL {url}")
            except etree.XMLSyntaxError as e:
                logger.error(f"Invalid XML/XBRL for URL {url}: {e}")
                return []

        else:
            logger.info(f"Unsupported content type {content_type} for URL {url}")
            return []

        if not extracted_sections:
            logger.info(f"No sections extracted from URL {url}")
            return []
        
        logger.info(f"Successfully processed {len(extracted_sections)} sections from URL {url}")
        return extracted_sections

    except Exception as e:
        logger.error(f"Error processing filing from URL {url}: {e}", exc_info=True)
        return []

# Function to process multiple URLs concurrently
def fetch_content_batch(urls, form_types, ciks, tickers, filed_ats, accession_numbers, max_workers=8):
    content_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(lambda x: process_filing(*x), zip(urls, form_types, ciks, tickers, filed_ats, accession_numbers)),
            total=len(urls),
            desc="Processing filings"
        ))
        for url, form_type, cik, ticker, filed_at, accession_number, sections in zip(urls, form_types, ciks, tickers, filed_ats, accession_numbers, results):
            for section in sections:
                content_data.append({
                    "document_url": url,
                    "formType": form_type,
                    "cik": cik,
                    "ticker": ticker,
                    "filedAt": filed_at,
                    "accessionNumber": accession_number,
                    "section_name": section['section_name'],
                    "content": section['content'],
                    "output_file": section['output_file']
                })
    return content_data

# Convert UTC to EST
def convert_utc_to_est(utc_dt_str):
    try:
        utc_dt = datetime.strptime(utc_dt_str, '%Y-%m-%d')
        utc = pytz.UTC
        utc_dt = utc.localize(utc_dt)
        est = pytz.timezone("US/Eastern")
        est_dt = utc_dt.astimezone(est).replace(tzinfo=None)
        return est_dt.strftime('%Y-%m-%d')
    except Exception as e:
        logger.error(f"Error converting UTC to EST for {utc_dt_str}: {e}")
        return utc_dt_str

# Fetch content for all filings
if not filings_df.empty:
    urls = filings_df['document_url'].tolist()
    form_types = filings_df['formType'].tolist()
    ciks_list = filings_df['cik'].tolist()
    tickers = filings_df['ticker'].tolist()
    filed_at = [convert_utc_to_est(str(dt)) if pd.notna(dt) else "" for dt in filings_df['filedAt']]
    accession_numbers = filings_df['accessionNumber'].tolist()
    
    content_data = fetch_content_batch(urls, form_types, ciks_list, tickers, filed_at, accession_numbers)
    
    content_df = pl.DataFrame(content_data)
    content_df = content_df.filter(pl.col("content").is_not_null())
    content_df.write_parquet(os.path.join("data", "03_primary", "filing_data.parquet"))
    logger.info(f"Saved {len(content_df)} sections to filing_data.parquet")
    
    print("\nSample of fetched content:")
    print(content_df.head())
else:
    logger.info("No content to process")
    print("No content to process")

logger.info("Program ends")
