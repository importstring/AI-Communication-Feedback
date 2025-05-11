import time
import re
import asyncio
import yt_dlp
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import undetected_chromedriver as uc

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

        # Setup undetected ChromeDriver with options
        options = uc.ChromeOptions()
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--start-maximized")
        # Add proxy or user-agent rotation here if available
        self.driver = uc.Chrome(options=options)

    def extract_names(self, soup):
        names = []
        headers = soup.find_all(['h2', 'h3', 'h4'])
        for header in headers:
            text = header.get_text().strip()
            if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+', text):
                names.append(text.split('(')[0].strip())
        return names

    def scrape_base_sources(self):
        import requests
        from bs4 import BeautifulSoup

        communicators = []
        for url in self.base_sources:
            print(f"Fetching base source: {url}")
            try:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                names = self.extract_names(soup)
                print(f"Found {len(names)} communicators in base source.")
                communicators.extend(names)
            except Exception as e:
                print(f"Error fetching {url}: {e}")
        return list(set(communicators))

    def google_search_new_sources(self):
        print("Searching Google for additional sources...")
        self.driver.get("https://www.google.com")
        search_box = self.driver.find_element(By.NAME, "q")
        search_query = "top communicators 2025 best public speakers"
        search_box.send_keys(search_query)
        search_box.submit()
        time.sleep(3)

        new_sources = []
        results = self.driver.find_elements(By.CSS_SELECTOR, "div.g")
        for result in results[:5]:
            try:
                link = result.find_element(By.TAG_NAME, "a").get_attribute("href")
                if link.startswith("http"):
                    print(f"Found new source: {link}")
                    new_sources.append(link)
            except Exception:
                continue
        self.base_sources.extend(new_sources)

    def youtube_search(self, query):
        print(f"Searching YouTube for: {query}")
        self.driver.get(f"https://www.youtube.com/results?search_query={query}")
        time.sleep(3)
        videos = self.driver.find_elements(By.CSS_SELECTOR, "ytd-video-renderer")
        links = []
        for v in videos[:3]:
            try:
                href = v.find_element(By.TAG_NAME, "a").get_attribute("href")
                if "youtube.com/watch" in href:
                    print(f"Found video link: {href}")
                    links.append(href)
            except Exception:
                continue
        return links

    def download_video(self, url):
        print(f"Downloading video: {url}")
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': 'videos/%(title)s.%(ext)s',
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print("Download successful.")
            self.video_count += 1
        except Exception as e:
            print(f"Download failed: {e}")

    async def run(self):
        start_time = time.time()
        while time.time() - start_time < self.run_duration:
            communicators = self.scrape_base_sources()
            self.google_search_new_sources()
            for communicator in communicators:
                if communicator not in self.processed_communicators:
                    print(f"Processing communicator: {communicator}")
                    video_links = self.youtube_search(communicator + " speech")
                    for link in video_links:
                        self.download_video(link)
                    self.processed_communicators.add(communicator)
                    print(f"Total videos downloaded so far: {self.video_count}")
            print("Sleeping for 10 minutes before next iteration...")
            await asyncio.sleep(600)  # Wait 10 minutes between iterations
        print(f"Run complete. Total videos downloaded: {self.video_count}")
        self.driver.quit()

if __name__ == "__main__":
    scraper = CommunicatorScraper(run_duration_seconds=3600)  # Run for 1 hour
    asyncio.run(scraper.run())
