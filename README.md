# Cubicle Sentiment Analysis App

The following is a sentiment analysis app built using Python which, given some review text as input, tries to predict if the sentiment is positive or negative. For training, reviews for products are scraped from TheCubicle, a company specialising in speedcubing and twisty puzzles. We utilize a web scraper to gather product reviews from some of the companies listed on the <a href=https://www.thecubicle.com/pages/collections/top-brands>top brands</a> page. We then use a Streamlit app to provide an interactive way for users to specify model training parameters.

A link to an online version of the app can be found <a href=https://cubicle-sentiment-analysis.streamlit.app>here</a>. 

## App demonstration
https://github.com/user-attachments/assets/1cd00660-415e-43db-9356-c6b459c4359f

## Local Setup
Start by cloning the repository then navigating into the project's root directory. Afterwords, set up a virtual environment and install the `requirements.txt` file. The current installation includes CPU-only PyTorch, as CUDA support is OS/GPU specific. See the <a href=https://pytorch.org/get-started/locally/>Get Started</a> page on the PyTorch website if you wish to run on your GPU instead.

## Web Scraping
The project already comes with an `output.csv` file which contains over 20,000 product reviews. However, in the event that the user wishes to scrape up to date reviews, they can navigate to the `src/` directory and run the following command:
```python
scrapy crawl cubicle_scraper -O output.csv
```
Details on how to edit specific properties of the scraper can be found in comments in `web_scraper/spiders/cubicle_scraper.py`. It should also be noted that the settings on the scraper have been kept quite conservative to avoid overloading the Cubicle webpages, so the default scraper will take a while to fully run. The scraper settings can be adjusted in `web_scraper/settings.py`, but be aware that making too many requests within a short period will likely lead to the scraper being blocked.

If the scraper has been running for a while, pressing <kbd>CTRL</kbd> + <kbd>C</kbd> will halt the scraper but retain all reviews gathered during its execution.

## Running The App
Regardless of whether or not the previous step was performed, running
```python
streamlit run src/app.py
```
from the project root will open the Streamlit app.

## Docker
If you have Docker installed, the app can also be run using a Docker image. To do this, run the following command:
```bash
docker run -p 8501:8501 ghcr.io/liam-stuart/cubicle-sentiment-analysis:main
```




