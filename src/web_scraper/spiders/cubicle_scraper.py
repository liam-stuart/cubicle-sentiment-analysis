import scrapy
import json
from scrapy.selector import Selector
from urllib.parse import urlencode
from web_scraper.items import ReviewItem


class CubicleScraperSpider(scrapy.Spider):
    name = "cubicle_scraper"
    start_urls = ["https://www.thecubicle.com/pages/collections/top-brands"]

    def parse(self, response):
        product_div = response.css("div.shopify-section")
        links = product_div.css("a::attr(href)").getall()[1:11]
        for link in links:
            yield scrapy.Request(url=response.urljoin(link), callback=self.parse_products)

    def parse_products(self, response):
        product_grid = response.css("div.product-card-grid")
        product_links = product_grid.css("a::attr(href)").getall()
        for link in product_links:
            yield scrapy.Request(url=response.urljoin(link), callback=self.parse_widget)

    def parse_widget(self, response):
        product_name = response.css("h1::text").get(default="").strip()
        yield from self.parse_reviews(response, product_name, page=1)
        widget_el = response.css("div.jdgm-review-widget, div.jdgm-widget")
        product_id = widget_el.attrib.get("data-id")
        if product_id:
            yield self.pagination_request(product_name, product_id, page=2)
        else:
            self.logger.warning("Could not locate Judge.me product ID for pagination.")

    def pagination_request(self, product_name, product_id, page):
        params = {
            "url": "thecubicleus.myshopify.com",
            "shop_domain": "thecubicleus.myshopify.com",
            "platform": "shopify",
            "page": page,
            "per_page": 10,
            "product_id": product_id
        }
        url = f"https://api.judge.me/reviews/reviews_for_widget?{urlencode(params)}"

        return scrapy.Request(
            url=url,
            callback=self.parse_judge_me,
            cb_kwargs={
                "product_name": product_name,
                "product_id": product_id,
                "page": page
            }
        )

    def parse_judge_me(self, response, product_name, product_id, page):
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            return

        html_content = data.get("html", "")
        if not html_content:
            return

        selector = Selector(text=html_content)
        reviews = selector.css("div.jdgm-rev")

        if not reviews:
            return

        yield from self.parse_reviews(selector, product_name, page=page)
        yield self.pagination_request(product_name, product_id, page + 1)

    def parse_reviews(self, selector, product_name, page):
        reviews = selector.css("div.jdgm-rev")
        for review in reviews:
            review_item = ReviewItem()
            body_text = " ".join(review.css("div.jdgm-rev__body *::text").getall()).strip()
            title_text = review.css("div.jdgm-rev__title::text").get(default="").strip()
            score_raw = review.css("div.jdgm-rev__rating::attr(data-score)").get(default="0")
            try:
                score = int(float(score_raw))
            except ValueError:
                score = 0

            review_item["product_name"] = product_name
            review_item["review_title"] = title_text
            review_item["review_text"] = body_text
            review_item["score"] = score
            yield review_item
