# Cubicle Sentiment Analysis App

The following is a sentiment analysis app built using Python which, given some review text as input, tries to predict if the sentiment is positive or negative. For training, reviews for products are scraped from the Cubicle, a company specialising in speedcubing and twisty puzzles. We utilize a web scraper to gather product reviews from some of the companies listed on the <a href=https://www.thecubicle.com/pages/collections/top-brands>top brands</a> page. We then use a Streamlit app to provide an interactive way for users to specify model training parameters.

## App demonstration
https://github.com/user-attachments/assets/caebffd7-16cd-497b-8296-ec8b4d258033

## Initial setup
Start by cloning the repository then navigating into the project's root directory. Afterwords, set up a virtual environment, install the requirements file, then navigate into the `src/` directory.

### Windows
```powershell
git clone https://github.com/liam-stuart/cubicle-sentiment-analysis.git
cd ./cubicle-sentiment-analysis/
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
cd ./src/
```

### Linux
```bash
git clone https://github.com/liam-stuart/cubicle-sentiment-analysis.git
cd cubicle-sentiment-analysis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd src
```
The current installation includes PyTorch, but not CUDA support as this is OS/GPU specific. See the <a href=https://pytorch.org/get-started/locally/>Get Started</a> page on the PyTorch website if you wish to enable this. 

## Web Scraping
The project already comes with an `output.csv` file which contains over 20,000 product reviews. However, in the event that the user wishes to scrape up to date reviews, they can run the following command:
```python
scrapy crawl cubicle_scraper -O output.csv
```
Details on how to edit specific properties of the scraper can be found in comments in `web_scraper/spiders/cubicle_scraper.py`. It should also be noted that the settings on the scraper have been kept quite conservative to avoid overloading the Cubicle webpages, so the default scraper will take a while to fully run. The scraper settings can be adjusted in `web_scraper/settings.py`, but be aware that making too many requests within a short period will likely lead to the scraper being blocked.

## Running The App
Regardless of whether or not the previous step was performed, running
```python
streamlit run app.py
```
will open the Streamlit app.




