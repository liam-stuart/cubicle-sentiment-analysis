from scrapy import Field, Item


class ReviewItem(Item):
    product_name = Field()
    review_title = Field()
    review_text = Field()
    score = Field()
