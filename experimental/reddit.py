import praw
from praw.models import MoreComments
import json

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pprint

# This needs to be run the first time you run this script ever
# nltk.download('all')
roasters = [
    "Java Blend Coffee Roasters",
    "Blues Brew Coffee",
    "Argyle Coffee Roasters",
    "Bonfire Coffee Company",
    "Simon Hsieh’s Aroma Roast Coffees",
    "Tico Coffee Roasters",
    "Voila Coffee",
    "MJ Bear Coffee",
    "Starbucks Coffee",
    "Conscious Cup Coffee Roasters",
    "Jebena Coffees",
    "Crown & Fancy",
    "T.S.E. Custom Roastery",
    "Valverde Coffee Roasters",
    "3-19 Coffee",
    "Colibrije Specialty Coffee",
    "Jade Cafe",
    "Café Chiayi J11",
    "Baba Java Coffee",
    "Belray Coffee",
    "Deeper Roots Coffee",
    "Steadfast Coffee",
    "Mamechamame Coffee",
    "Bar Nine",
    "Cafe Kreyol",
    "Dapper & Wise",
    "Lamppost Coffee Roasters",
    "Roadmap Coffeeworks",
    "Cafe Unido",
    "Fieldheads Coffee Roasting",
    "Roast House",
    "Peet's Coffee & Tea",
    "Good Folks Coffee",
    "Big Shoulders Coffee ",
    "Water Avenue Coffee",
    "Fidalgo Coffee",
    "JBC Coffee Roasters",
    "Black Coffee in Black Jar",
    "Speckled Ax",
    "Equator Coffees & Teas",
    "Good Chance Biotechnology",
    "Roasters Note",
    "Revel Coffee",
    "Greater Goods Coffee Roasters",
    "Waka Coffee",
    "Blue Bottle Coffee",
    "Volcanica Coffee",
    "Cafe Grumpy",
    "MK Coffee Roasters",
    "Speedwell Coffee",
    "Cozy House Coffee",
    "Q Burger",
    "Handlebar Coffee",
    "Café Hubert Saint-Jean",
    "Kasasagi Coffee Roasters",
    "Coffee Hound",
    "Coffeebar",
    "Stumptown Coffee Roasters",
    "Brioso Coffee",
    "Port of Mokha",
    "Samlin Coffee",
    "Rusty's Hawaiian Coffee",
    "Vienna Coffee Company",
    "Java D'oro Gourmet Coffee",
    "Propeller Coffee",
    "Old Soul Co.",
    "Equiano Coffee",
    "Atom Coffee Roasters",
    "Battlecreek Coffee",
    "Oceana Coffee",
    "Temple Coffee Roasters",
    "Aether Coffee",
    "Fumi Coffee Company",
    "Durango Coffee Company",
    "Direct Coffee",
    "Lucky Cafe",
    "Kona View Coffee",
    "Red E Café",
    "A.R.C.",
    "Evie's Cafe",
    "Flight Coffee Co.",
    "Japanese Coffee Co.",
    "Ramshead Coffee Roasters",
    "Chocolate Fish Coffee Roasters",
    "Reanimator Coffee Roasters",
    "Coava Coffee Roasters",
    "Mostra Coffee",
    "Sightseer Coffee",
    "709 Boutique Coffee",
    "Broadsheet Coffee",
    "Satur Specialty Coffee",
    "Mellower Coffee",
    "Drink Coffee Do Stuff",
    "Kona Love Coffee Co.",
    "Glory Brew",
    "Yuban",
    "Nine Point Coffee",
    "Big Shoulders Coffee",
    "Toca Coffee",
    "Cafe Red Bean Shop",
    "De Clieu Coffee",
    "BeanFruit Coffee Co.",
    "Moore Coffee",
    "Red Bay Coffee",
    "Noelia Coffee",
    "Dávila Kafe",
    "Sunny's Coffee",
    "Voyage Coffee",
    "Ba Yang Coffee",
    "Big Creek Coffee Roasters",
    "9612 Coffee Room",
    "Interpretation Coffee",
    "OQ Coffee Co.",
    "Metropolis Coffee",
    "421 Brew House",
    "Doho Coffee Roasters",
    "Fabbrica Coffee",
    "Kirkland Signature (Costco)",
    "Mother Pigeons",
    "Red Rock Roasters",
    "Magnolia Coffee",
    "Cup to Cup Coffee Roasters",
    "Oughtred Roasting Works",
    "ILSE Coffee Roasters",
    "Mr. Espresso",
    "El Gran Café ",
    "Evie’s Cafe",
    "Home in Harmony",
    "Coffee by Design",
    "Café Grumpy",
    "Sip of Hope Community Coffee Roasters",
    "Forest Roasting",
    "Felala Coffee Lab",
    "Caffeic",
    "Quartet Kaffe",
    "Blanchard's Coffee",
    "Noble Coffee Roasting",
    "Dr. Bean's Coffee Roasters",
    "BLK & Bold",
    "Equator Coffees",
    "LECO Coffee",
    "Brain Helmet",
    "Temple Coffee",
    "Kaldi's Coffee",
    "EK Roast Studio",
    "Greybox Coffee",
    "Bona Kafo Roastery",
    "Cheer Beans",
    "Prairie Lily Coffee",
    "Black & White Coffee Roasters",
    "CofFeeling",
    "Chiming Coffee",
    "Hey Brown",
    "King's Gambit Coffee",
    "Taster's Coffee",
    "Coffee By Design",
    "Good Coffee Club",
    "Bassline Coffee",
    "Stereo Coffee Roasters",
    "4 Plus Coffee",
    "Telluride Coffee Roasters",
    "Ming's Coffee Playroom",
    "Hidden Coffee HK",
    "Counter Culture Coffee",
    "Mooju Coffee",
    "San Francisco Bay Coffee Company",
    "1980 CAFE",
    "Jiva Cubes Instant Coffee Packets",
    "Topeca Coffee",
    "Kauai Coffee Company",
    "PT’s Coffee Roasting Co.",
    "Battle Creek Coffee Roasters",
    "Allegro Coffee Roasters",
    "Dragonfly Coffee Roasters",
    "Desolate Café",
    "Per'la Specialty Roasters",
    "Joe Van Gogh",
    "Jampot Poorock Abbey",
    "Per’la Specialty Roasters",
    "Atomic Coffee Roasters",
    "Madness Roastworks",
    "Nook Bakery & Coffee Bar",
    "JYL Cafe",
    "Creation Food Co.",
    "Mystic Monk Coffee",
    "North Coast Coffee Roasting",
    "Kakalove Cafe",
    "Vascobelo",
    "Taokas Coffee",
    "Vermont Artisan Coffee & Tea",
    "Doi Chaang Coffee",
    "Big Island Coffee Roasters",
    "Beanfruit Coffee Co.",
    "Tandem Coffee",
    "Golden Ratio",
    "Klatch Coffee",
    "Hale Coffee Company",
    "Simon Hsieh Aroma Roast Coffees",
    "Cameron’s Coffee",
    "True Coffee Roasters",
    "San Francisco Bay Coffee",
    "Pop Coffee Works",
    "Allegro Coffee",
    "Maxim Coffee",
    "Fresh Roasted Coffee",
    "Black Coffee In Black Jar",
    "Peet's Coffee",
    "Wild Goose Coffee Roasters",
    "The Formosa Coffee",
    "Dr. Young Cafe",
    "Turnstile Coffee Roasters",
    "RND & Red Rooster Coffee Roaster",
    "NINETYs Roastery",
    "Poppets Coffee",
    "Detour Coffee Roasters",
    "Ho Soo Tsai",
    "Finca Las Nieves",
    "Ka'u Specialty Coffee",
    "Flower Child Coffee",
    "Jackrabbit Java",
    "Indaba Coffee Roasters",
    "The Curve Coffee Roasting",
    "Oak & Bond Coffee Company",
    "Three Keys Coffee",
    "Kona Farm Direct",
    "David's Nose",
    "Zuco Coffee Roasters",
    "Little Wolf Coffee",
    "Columbia Street Roastery",
    "Fumi Coffee",
    "Eagle Valley Coffee",
    "Notch Coffee",
    "Amor",
    "Foggy Hills Coffee Company",
    "Nostalgia Coffee Roasters",
    "Singsun Coffee",
    "Old World Coffee Lab",
    "Black Oak Coffee Roasters",
    "Soul Work Coffee",
    "Le Brewlife",
    "Emiliani Coffee",
    "Greenwell Farms",
    "Peacock Coffee Roaster",
    "Fieldheads Coffee",
    "Mr. Chao Coffee",
    "Mute Roaster",
    "Barrington Coffee Roasting",
    "RanGuo Coffee",
    "Trident Coffee",
    "Wheelys Cafe Taiwan",
    "Kona Roasted",
    "Gracenote Coffee Roasters",
    "Tipico Coffee",
    "Hill's Bros. Coffee",
    "Jaunt Coffee Roasters",
    "Dear John Coffee",
    "Plat Coffee Roastery",
    "Hsus Cafe",
    "Islamorada Coffee Roasters",
    "Novo Coffee",
    "Sudden Coffee",
    "Folly Coffee Roasters",
    "The WestBean Coffee Roasters",
    "Coffee Please",
    "Hala Tree Kona Coffee",
    "Golin Coffee Roasters",
    "St1 Cafe/Work Room",
    "Southeastern Roastery",
    "Black Medicine",
    "DoDo Kaffa",
    "GK Coffee",
    "Temple Coffee and Tea",
    "Brand 425",
    "El Gran Café",
    "Level Ground Trading",
    "94 Fresh Coffee",
    "Dou Zhai Coffee & Roast",
    "Theory Coffee Roasters",
    "Cloud City Coffee",
    "VN Beans",
    "Muka Coffee",
    "Duluth Coffee",
    "Dante Coffee",
    "Kahawa 1893",
    "Rotate Fun Club",
    "RND",
    "PT's Coffee Roasting",
    "Pro Aroma Enterprise Coffee",
    "Cadence Cold Brew Coffee",
    "Dory Coffee Roasters",
    "Warm Air Kafe",
    "South Slope Coffee Roasters",
    "Giv COFFEE",
    "Jampot Poorrock Abbey",
    "A&E Coffee & Tea",
    "Merge Coffee Company",
    "Nella's SOGOOD Coffee",
    "Lone Coffee",
    "Olympia Coffee Roasting",
    "Qima Coffee",
    "Flower Coffee Workshop",
    "Open Seas Coffee",
    "Street Bean",
    "Small Eyes Café",
    "Desert Sun Coffee Roasters",
    "Peach Coffee Roasters",
    "Treeman Coffee",
    "Caffe Luxxe",
    "Tug Hill Artisan Roasters",
    "Larry's Coffee",
    "Espresso Republic",
    "Mustard Seed Cafe",
    "Nongfu Spring Co., Ltd.",
    "Topeca Coffee Roasters",
    "The Reverse Orangutan",
    "Euphora",
    "The Angry Roaster",
    "Kafe 1804",
    "Vennel Coffee",
    "Evans Brothers Coffee Roasters",
    "Kunshan Kokei co., LTD",
    "Rusty Dog Coffee",
    "Desolate Cafe",
    "Red Rooster Coffee Roaster",
    "MoonMoon Coffee",
    "Modern Times Coffee",
    "Roadmap CoffeeWorks",
    "Bird Rock Coffee Roasters",
    "Chousin Coffee Collection",
    "Press House Coffee",
    "Barefoot Coffee Roasters",
    "Steady State Roasting Company",
    "AKA Coffee Roasters",
    "Regent Coffee",
    "One Fresh Cup",
    "Bargain Cafe",
    "Finca Tasta",
    "Chez Cafe",
    "Mudhouse Coffee Roasters",
    "Geisha Coffee Roaster",
    "Balmy Day Coffee Office",
    "Cafe Fugu Roasters",
    "Amavida Coffee Roasters",
    "Grounds for Change",
    "Badbeard’s Microroastery",
    "Normandy Coffee Thai Cuisine",
    "Rhetoric Coffee",
    "Tribo Coffee",
    "Gorilla Conservation Coffee",
    "Montana Coffee Traders",
    "Fire Ridge Coffee",
    "Chaos Coffee Company",
    "Qin Mi Coffee",
    "Eight O'Clock Coffee",
    "Hula Daddy Kona Coffee",
    "Peri Coffee",
    "States Coffee",
    "Rubasse Coffee Roaster",
    "MT49 Cafe",
    "CafeTaster",
    "Augie's Coffee Roasters",
    "Reunion Island Coffee",
    "Ilustre Specialty Coffees",
    "Lina Premium Coffee",
    "Rest Coffee Roasters",
    "Rusty’s Hawaiian",
    "Miranda Farms",
    "Novo Coffee Roasters",
    "Buon Caffe",
    "Bluebeard Coffee Roasters",
    "Press Coffee",
    "SÖT Coffee Roaster",
    "Barrington Coffee Roasting Co.",
    "1951 Coffee Company",
    "Raggiana Coffee",
    "Monarch Coffee",
    "The Curve Coffee Roasting Co.",
    "Cimarron Coffee Roasters",
    "Vivid Coffee",
    "Sucré Beans",
    "Tony's Coffee",
    "HT Traders",
    "Rusty's Hawaiian",
    "Santos Coffee",
    "PT's Coffee Roasting Co.",
    "Rooster Roastery",
    "Charlotte Coffee Company",
    "Euphora Coffee",
    "SOT Coffee Roaster",
    "El Gran Cafe",
    "Difference Coffee",
    "Wheelys Café Taiwan",
    "Luv Moshi",
    "Swelter Coffee",
    "Simon Hsieh's Aroma Roast Coffees",
    "Water Street Coffee",
    "Olympia Coffee",
    "Triple Coffee Co.",
    "Hub Coffee Roasters",
    "Pamma Coffee",
    "Genesis Coffee Lab",
    "Portrait Coffee",
    "Good Chance Biotechnology, Ltd.",
    "Kaffetre",
    "Four Barrel Coffee",
    "Nescafe",
    "Lexington Coffee Roasters",
    "Spirit Animal Coffee",
    "Branch Street Coffee Roasters",
    "Queen Coffee",
    "Wuguo Cafe",
    "Hatch Coffee",
    "Durango Coffee Companuy",
    "UCC Ueshima Coffee",
    "Guorizi Zhai Coffee",
    "Signature Reserve",
    "Corvus Coffee Roasters",
    "Kakalove Café",
    "Willoughby's Coffee & Tea",
    "Min Enjoy Cafe",
    "Cafe Arles",
    "Souvenir Coffee",
    "Omine Coffee",
    "Cafe Femenino",
    "Coffee Cycle",
    "Spix's Cafe",
    "Choosy Gourmet",
    "Dinwei Cafe",
    "Chu Bei",
    "Battlecreek Coffee Roasters",
    "Sensory House Coffee",
    "HWC Roasters",
    "Ghost Town Coffee Roasters",
    "inLove Cafe",
    "Taster’s Coffee",
    "Back Home Coffee",
    "Succulent Coffee Roasters",
    "RD Cafe",
    "Campos Coffee",
    "Three Chairs Specialty Turkish Coffee",
    "Yellow Brick Coffee",
    "Bluekoff Company",
    "Small Eyes Cafe",
    "Cafe Douceur",
    "Looking Homeward Coffee",
    "Origin Coffee Roasters",
    "Boon Boona Coffee",
    "Wonderstate Coffee",
    "VERYTIME",
    "Pedestrian Coffee",
    "Folklore Coffee",
    "Tehmag Foods Corporation",
    "Manzanita Roasting Company",
    "Ben's Beans",
    "Mi's Cafe",
    "Chromatic Coffee",
    "Auto Coffee",
    "modcup coffee",
    "Bootstrap Coffee Roasters",
    "States Coffee & Mercantile",
    "Wildgoose Coffee Roasters",
    "RamsHead Coffee Roasters",
    "Thanksgiving Coffee Company",
    "Bridge Coffee Co.",
    "Collage Coffee",
    "Joe Van Gogh Coffee",
    "Highwire Coffee Roasters",
    "Chromatic x Dripdash Collab",
    "SkyTop Coffee",
    "Crescendo Coffee Roasters",
    "Cafe Virtuoso",
    "Sweet Bloom Coffee Roasters",
    "Yum China",
    "Green Stone Coffee",
    "Lin's Home Roasters",
    "Steady State Roasting",
    "Paradise Roasters",
    "Mikava Coffee",
    "Beachcomber Coffee",
    "Red Bean Coffee",
    "Novel Coffee Roasters",
]


reddit = praw.Reddit(
    client_id="your client id",
    client_secret="your client secret",
    user_agent="macos:findyourcoffee:v1.0.0 (by u/worm-dealer)",
)
print(reddit.read_only)

# Show listings (general): top_ten_results = [s for s in reddit.subreddit("coffee").new(limit=10)]

# Search a subreddit using a query:


def find_query_in_comments(query, comment):

    def find_query_in_comments_acc(reply):
        if isinstance(reply, MoreComments):
            return []

        results = []
        lower_text = reply.body.lower()
        lines = lower_text.split(". ")
        for line in lines:
            if query in line:
                results.append(line)

        for r in reply.replies:
            results += find_query_in_comments_acc(r)

        return results

    return find_query_in_comments_acc(comment)


def preprocess_text(text):

    tokens = word_tokenize(text.lower())
    filtered_tokens = [
        token for token in tokens if token not in stopwords.words("english")
    ]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    processed_text = " ".join(lemmatized_tokens)

    return processed_text


# # query issue: 'press coffee' -> 'french press coffee'
# query = "modcup"
# query = query.lower()
# search_results = [s for s in reddit.subreddit("coffee").search(query=query)]

# # print(search_results)
# query_found = []
# for submission in search_results:
#     lower_text = submission.selftext.lower()
#     lines = ". ".join(lower_text.split("\n")).split(". ")
#     for line in lines:
#         if query in line:
#             query_found.append(line)

#     submission.comments.replace_more(limit=None)
#     for c in submission.comments:
#         query_found += find_query_in_comments(query, c)

# for i, result in enumerate(query_found):
#     print(str(i + 1) + ")", result)

# analyzer = SentimentIntensityAnalyzer()

# for text in query_found:
#     processed_text = preprocess_text(text)
#     sentiments = analyzer.polarity_scores(processed_text)
#     print(sentiments)

coffeeSub = reddit.subreddit("coffee")


def buildJson():
    allJson = {}
    for i, x in enumerate(roasters):
        query = x.lower()
        search_results = [s for s in coffeeSub.search(query=query)]
        query_found = []
        for submission in search_results:
            lower_text = submission.selftext.lower()
            lines = ". ".join(lower_text.split("\n")).split(". ")
            for line in lines:
                if query in line:
                    query_found.append(line)

            submission.comments.replace_more(limit=None)
            for c in submission.comments:
                query_found += find_query_in_comments(query, c)

        analyzer = SentimentIntensityAnalyzer()
        stored = {}
        for text in query_found:
            processed_text = preprocess_text(text)
            sentiments = analyzer.polarity_scores(processed_text)
            stored[text] = [processed_text, sentiments]
        allJson[x] = stored
    with open("sentiments.json", "w") as outfile:
        json.dump(allJson, outfile)


buildJson()
