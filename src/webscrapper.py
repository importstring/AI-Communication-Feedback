import time
import re
import asyncio
import yt_dlp
from bs4 import BeautifulSoup
import requests
import random
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
from youtube_search import YoutubeSearch

class CommunicatorScraper:
    def __init__(self, run_duration_seconds=3600):
        self.base_sources = [
            "https://leadr.co/blog/famous-public-speakers/",
            "https://www.agilitypr.com/blog/post/best-practices/10-great-female-communicators-you-should-know/",
            "https://thesweeneyagency.com/blog/top-7-speakers-communication/"
        ]
        self.processed_communicators = set()
        self.video_count = 0
        self.run_duration = run_duration_seconds

        # Configure stealth browser
        options = uc.ChromeOptions()
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument(f"user-agent={self._random_user_agent()}")
        self.driver = uc.Chrome(options=options, headless=False)

    def _random_user_agent(self):
        agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        ]
        return random.choice(agents)

    def _extract_communicators(self, soup):
        patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
            r'\b(?:Dr|Mr|Mrs|Ms)\.?\s+[A-Z][a-z]+ [A-Z][a-z]+\b'  # Titles
        ]
        return list(set(re.findall('|'.join(patterns), soup.get_text())))

    def _scrape_website(self, url):
        try:
            response = requests.get(url, timeout=10, headers={'User-Agent': self._random_user_agent()})
            soup = BeautifulSoup(response.text, 'html.parser')
            return self._extract_communicators(soup)
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return []

    def _google_search_sources(self):
        print("Searching Google for expert communicators...")
        self.driver.get("https://www.google.com")
        search_box = self.driver.find_element(By.NAME, "q")
        search_box.send_keys("top communicators 2025 best public speakers")
        search_box.submit()

        # Wait for results and click through to actual pages
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.g"))
        )

        results = self.driver.find_elements(By.CSS_SELECTOR, "div.g a")[:5]
        for result in results:
            try:
                url = result.get_attribute("href")
                if url and "google.com" not in url:
                    print(f"Visiting source: {url}")
                    self.driver.execute_script("window.open('');")
                    self.driver.switch_to.window(self.driver.window_handles[1])
                    self.driver.get(url)
                    
                    # Extract names from actual page content
                    page_communicators = self._extract_communicators(
                        BeautifulSoup(self.driver.page_source, 'html.parser')
                    )
                    self.base_sources.extend(page_communicators)
                    
                    self.driver.close()
                    self.driver.switch_to.window(self.driver.window_handles[0])
                    time.sleep(random.uniform(2, 5))
            except Exception as e:
                print(f"Error processing result: {str(e)}")

    def _youtube_search(self, query):
        try:
            results = YoutubeSearch(query, max_results=10).to_dict()
            return [f"https://youtube.com/watch?v={res['id']}" for res in results]
        except Exception as e:
            print(f"YouTube search failed: {str(e)}")
            return []

    def _download_video(self, url):
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': f'videos/%(uploader)s/%(title)s.%(ext)s',
            'retries': 3,
            'merge_output_format': 'mp4',
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            self.video_count += 1
        except Exception as e:
            print(f"Download failed: {str(e)}")

    async def run(self):
        start_time = time.time()
        while time.time() - start_time < self.run_duration:
            # Scrape base sources
            communicators = []
            for url in self.base_sources:
                if url.startswith("http"):
                    communicators.extend(self._scrape_website(url))
            
            # Find new sources via Google
            self._google_search_sources()
            
            # Process communicators
            for name in set(communicators):
                if name not in self.processed_communicators:
                    print(f"Processing: {name}")
                    for video_url in self._youtube_search(f"{name} speech"):
                        self._download_video(video_url)
                    self.processed_communicators.add(name)
                    print(f"Total videos: {self.video_count}")
                    
                    # Random delay between requests
                    await asyncio.sleep(random.uniform(1, 3))
            
            # Random longer delay between iterations
            await asyncio.sleep(random.uniform(30, 60))
        
        self.driver.quit()
        print(f"Completed. Total videos downloaded: {self.video_count}")

if __name__ == "__main__":
    scraper = CommunicatorScraper(run_duration_seconds=3600)
    asyncio.run(scraper.run())
