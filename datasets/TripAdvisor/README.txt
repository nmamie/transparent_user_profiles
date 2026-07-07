If you simply want to reproduce the experiments reported in our paper, please ignore OriginalReviews.json because it is meant for other purposes, e.g., teaching or future work.

OriginalReviews.json contains the original reviews crawled by ourselves from TripAdvisor (https://www.tripadvisor.com/)
In Python, you can read the data in the following way.

import json
with open('OriginalReviews.json', 'r', encoding='utf-8') as f:
    for review in json.load(f):
        if 'reviewHeading' in review:
            print(review['reviewHeading'])

An example of review is shown below.

{
    "userID": "6540C6386BF488E969263634227CBBB0",
    "hotelID": "224761",
    "reviewText": "Another very enjoyable night at this centrally located boutique hotel . Excellent service ,lovely accommodation with the best bed in London ! The bar is intimate as is the excellent restaurant and nothing is too much trouble.Will always return here ,it's quite unique .",
    "reviewURL": "https://www.tripadvisor.com/ShowUserReviews-g186338-d224761-r147393627-St_James_s_Hotel_and_Club-London_England.html",
    "userName": "Worldtraveller1953",
    "userLocation": "Chipping Ongar, United Kingdom",
    "hotelTitle": "St. James's Hotel and Club",
    "hotelCity": "London",
    "rating": 5,
    "reviewDate": "2012-12-13",
    "reviewHeading": "Great ,again",
    "travelType": "Couples",
    "travelDate": "2012-12",
    "subRatings": {
        "Value": 5,
        "Location": 5,
        "Sleep Quality": 5,
        "Rooms": 5,
        "Cleanliness": 5,
        "Service": 5
    }
}