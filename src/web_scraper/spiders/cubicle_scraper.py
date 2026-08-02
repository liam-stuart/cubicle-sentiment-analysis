import scrapy
import json
import re
import numpy as np
from scrapy.selector import Selector
from urllib.parse import urlencode
from web_scraper.items import ReviewItem


CATEGORY_FILTER = ['2x2 Speed Cubes',
                   '3x3 Speed Cubes',
                   '4x4 Speed Cubes',
                   '5x5 Speed Cubes',
                   '6x6 Speed Cubes',
                   '7x7 Speed Cubes',
                   '8x8-21x21 Cubes',
                   'Megaminx',
                   'Pyraminx',
                   'Square-1',
                   'FTO',
                   'Skewb',
                   'Clock',
                   'Cuboids',
                   'Shape Mods',
                   'Minx+',
                   'Magic Panels',
                   'Gear Cubes',
                   'Picture Cubes',
                   'Smart Cubes',
                   'Mystery Puzzles']
REGEX_FILTER = "|".join(map(re.escape, CATEGORY_FILTER))


class CubicleScraperSpider(scrapy.Spider):
    name = "cubicle_scraper"
    start_urls = ["https://www.thecubicle.com/pages/collections/top-brands"]

    def parse(self, response):
        brand_div = response.css("div.shopify-section")
        brand_links = brand_div.css("a::attr(href)").getall()[1:21]

        for brand_link in brand_links:
            yield scrapy.Request(url=response.urljoin(brand_link), callback=self.parse_products,
                                 errback=self.handle_error)

    def parse_products(self, response):
        product_cards = response.css("div[x-data='productCard']")
        for product_card in product_cards:
            review_number_raw = product_card.css("div::attr(data-number-of-reviews)").get(default="0")
            try:
                review_number = int(review_number_raw)
            except ValueError:
                review_number = 0

            if review_number != 0:
                product_link = product_card.css("a::attr(href)").get(default="")
                if product_link is not None:
                    yield scrapy.Request(url=response.urljoin(product_link), callback=self.parse_product,
                                         errback=self.handle_error,  cb_kwargs={"review_number": review_number})

        next_page = response.css("a[title*='Next']::attr(href)").get(default=None)
        if next_page is not None:
            yield scrapy.Request(url=response.urljoin(next_page), callback=self.parse_products,
                                 errback=self.handle_error)

    def parse_product(self, response, review_number):
        categories = response.xpath('//table//*[text()="Type"]/following-sibling::*[1]//text()').getall()
        categories = " ".join([item.strip() for item in categories if item.strip()])

        is_puzzle = re.search(REGEX_FILTER, categories)
        if is_puzzle is None:
            return

        else:
            pages = int(np.ceil(review_number / 10))
            product_id = response.css("div.jdgm-review-widget::attr(data-id)").get(default="")

            if product_id is not None:
                for page in range(1, pages + 1):
                    yield self.pagination_request(product_id=product_id, page=page)
            else:
                self.logger.warning("Could not locate Judge.me product ID for request.")
                return

    def pagination_request(self, product_id, page):
        params = {
            "url": "thecubicleus.myshopify.com",
            "shop_domain": "thecubicleus.myshopify.com",
            "platform": "shopify",
            "page": page,
            "per_page": 10,
            "product_id": product_id
        }
        url = f"https://api.judge.me/reviews/reviews_for_widget?{urlencode(params)}"

        return scrapy.Request(url=url, callback=self.parse_judge_me, errback=self.handle_error)

    def parse_judge_me(self, response):
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            return

        html_content = data.get("html", "")
        if not html_content:
            return

        response = Selector(text=html_content)
        reviews = response.css("div.jdgm-rev")
        if not reviews:
            return

        yield from self.parse_reviews(reviews=reviews)

    def parse_reviews(self, reviews):
        for review in reviews:
            review_item = ReviewItem()
            product_name = review.attrib.get("data-product-title", "").strip()
            body_paragraphs = review.css("div.jdgm-rev__body p::text").getall()
            body_text = " ".join(p.strip() for p in body_paragraphs if p.strip()).strip()
            title_text = review.css("b.jdgm-rev__title::text").get(default="").strip()
            score_raw = review.css("span.jdgm-rev__rating::attr(data-score)").get(default="0")
            try:
                score = int(score_raw)
            except ValueError:
                score = 0

            review_item["product_name"] = product_name
            review_item["review_title"] = title_text
            review_item["review_text"] = body_text
            review_item["score"] = score
            yield review_item

    def handle_error(self, failure):
        self.logger.error(f"Request failed: {failure}")
