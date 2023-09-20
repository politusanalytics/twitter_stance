import pymongo
import re

repo_path = "/home/username/twitter_stance"

# Connect to mongodb
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["politus_twitter"]
tweet_col = db["tweets"]

reg_list = []
with open("{}/data/mansur_stance/mansur_keywords.txt".format(repo_path), "r", encoding="utf-8") as f:
    for line in f:
        if line:
            reg_list.append(line[:-1])
    reg = re.compile("|".join(reg_list), flags=re.S|re.I)
    #reg = "|".join(reg_list)

http_reg = re.compile(r"^(?!.*(https:|http:|www\.))", flags=re.S|re.I)

keyword_true_query = {"has_mansur_keyword": None,
                      "$and": [{"text": {"$regex": http_reg}},
                               {"text": {"$regex": reg}}]}
tweet_col.update_many(keyword_true_query, {"$set": {"has_mansur_keyword": True}})

# keyword_false_query = {"has_mansur_keyword": None,
#                        "$or": [{"text": {"$not": {"$regex": http_reg}}},
#                                {"text": {"$not": {"$regex": reg}}}]}
keyword_false_query = {"has_mansur_keyword": None}
tweet_col.update_many(keyword_false_query, {"$set": {"has_mansur_keyword": False}})
