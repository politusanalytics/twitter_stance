"""
This script is used to predict big files with a trained model from train.py.
You can already do prediction in train.py, but this script lazy loads a file
so that we can process bigger files.
"""

from transformers import AutoModel, AutoTokenizer, AutoConfig
import sys
import numpy as np
import json
import torch
import gzip
import pymongo
from pymongo import UpdateOne
import re

# TODO: This currently does not work with an input file!

# INPUTS
# If database: input "database". If input filename: should be json or json.gz file in json line format.
database_or_input_filename = sys.argv[1]
task_name = sys.argv[2]

if database_or_input_filename == "database":
    # Connect to mongodb
    mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = mongo_client["politus_twitter"]
    tweet_col = db["tweets"]

# MUST SET THESE VALUES
output_filename = "out.json"
pretrained_transformers_model = "dbmdz/bert-base-turkish-128k-cased"
max_seq_length = 64
batch_size = 1024
repo_path = "/home/username/twitter_stance"

if task_name == "erdogan_relevant":
    label_list = ["relevant", "irrelevant"]
    idx_to_label = {i: lab for i,lab in enumerate(label_list)}
    encoder_path = "{}/models/best_models/encoder_dbmdz_bert-base-turkish-128k-cased_erdogan_relevant_46.pt".format(repo_path)
    classifier_path = "{}/models/best_models/classifier_dbmdz_bert-base-turkish-128k-cased_erdogan_relevant_46.pt".format(repo_path)

    # update has_erdogan_keyword field first
    if database_or_input_filename == "database":
        case_insensitive_reg_list = []
        normal_reg_list = []
        with open("{}/data/erdogan_stance/erdogan_keywords.txt".format(repo_path), "r", encoding="utf-8") as f:
            for line in f:
                if line:
                    if "[" in line:
                        case_insensitive_reg_list.append(line[:-1])
                    else:
                        normal_reg_list.append(line[:-1])
            case_insensitive_reg = re.compile("|".join(case_insensitive_reg_list), flags=re.S|re.I)
            #case_insensitive_reg = "|".join(case_insensitive_reg_list)
            normal_reg = re.compile("|".join(normal_reg_list), flags=re.S)
            #normal_reg = "|".join(normal_reg_list)

        keyword_true_query = {"has_erdogan_keyword": None,
                              "$or": [{"text": {"$regex": case_insensitive_reg}},
                                      {"text": {"$regex": normal_reg}}]}
        tweet_col.update_many(keyword_true_query, {"$set": {"has_erdogan_keyword": True}})
        keyword_false_query = {"has_erdogan_keyword": None,
                               "$and": [{"text": {"$not": {"$regex": case_insensitive_reg}}},
                                        {"text": {"$not": {"$regex": normal_reg}}}]}
        tweet_col.update_many(keyword_false_query, {"$set": {"has_erdogan_keyword": False}})

        # for row in tweet_col.find({"has_erdogan_keyword": None}, ["text"]): # , "date": {"$gte": dateutil.parser.parse("2023-01-01")}
        #     if re.search(case_insensitive_reg, row.get("text", "")) == None and re.search(normal_reg, row.get("text", "")) == None:
        #         tweet_col.update_one({"_id": row["_id"]}, {"$set": {"has_erdogan_keyword": False}})
        #     else:
        #         tweet_col.update_one({"_id": row["_id"]}, {"$set": {"has_erdogan_keyword": True}})


    query = {"has_erdogan_keyword": True, "erdogan_relevant": None, "text": {"$nin": ["", None]}}

elif task_name == "erdogan_stance":
    label_list = ["pro", "against", "neutral"]
    idx_to_label = {i: lab for i,lab in enumerate(label_list)}
    encoder_path = "{}/models/best_models/encoder_dbmdz_bert-base-turkish-128k-cased_erdogan_stance_45.pt".format(repo_path)
    classifier_path = "{}/models/best_models/classifier_dbmdz_bert-base-turkish-128k-cased_erdogan_stance_45.pt".format(repo_path)
    query = {"has_erdogan_keyword": True, "erdogan_relevant": "relevant", "erdogan_stance": None, "text": {"$nin": ["", None]}}

elif task_name == "kk_relevant":
    label_list = ["relevant", "irrelevant"]
    idx_to_label = {i: lab for i,lab in enumerate(label_list)}
    encoder_path = "{}/models/best_models/encoder_dbmdz_bert-base-turkish-128k-cased_kk_relevant_47.pt".format(repo_path)
    classifier_path = "{}/models/best_models/classifier_dbmdz_bert-base-turkish-128k-cased_kk_relevant_47.pt".format(repo_path)

    # update has_kk_keyword field first
    if database_or_input_filename == "database":
        reg_list = []
        with open("{}/data/kilicdar_stance/kilicdar_keywords.txt".format(repo_path), "r", encoding="utf-8") as f:
            for line in f:
                if line:
                    reg_list.append(line[:-1])
            reg = re.compile("|".join(reg_list), flags=re.S|re.I)
            #reg = "|".join(reg_list)

        http_reg = re.compile(r"^(?!.*(https:|http:|www\.))", flags=re.S|re.I)

        keyword_true_query = {"has_kk_keyword": None,
                              "$and": [{"text": {"$regex": http_reg}},
                                       {"text": {"$regex": reg}}]}
        tweet_col.update_many(keyword_true_query, {"$set": {"has_kk_keyword": True}})
        keyword_false_query = {"has_kk_keyword": None,
                               "$or": [{"text": {"$not": {"$regex": http_reg}}},
                                       {"text": {"$not": {"$regex": reg}}}]}
        tweet_col.update_many(keyword_false_query, {"$set": {"has_kk_keyword": False}})

        # url_reg = re.compile("^(?!.*(https:|http:|www\.))", flags=re.S|re.I)
        # for row in tweet_col.find({"has_kk_keyword": None}, ["text"]): # , "date": {"$gte": dateutil.parser.parse("2023-01-01")}
        #     if re.search(url_reg, row.get("text", "")) == None and re.search(reg, row.get("text", "")) != None:
        #         tweet_col.update_one({"_id": row["_id"]}, {"$set": {"has_kk_keyword": True}})
        #     else:
        #         tweet_col.update_one({"_id": row["_id"]}, {"$set": {"has_kk_keyword": False}})


    query = {"has_kk_keyword": True, "kk_relevant": None, "text": {"$nin": ["", None]}}

elif task_name == "kk_stance":
    label_list = ["pro", "against", "neutral"]
    idx_to_label = {i: lab for i,lab in enumerate(label_list)}
    encoder_path = "{}/models/best_models/encoder_dbmdz_bert-base-turkish-128k-cased_kk_stance_58.pt".format(repo_path)
    classifier_path = "{}/models/best_models/classifier_dbmdz_bert-base-turkish-128k-cased_kk_stance_58.pt".format(repo_path)
    query = {"has_kk_keyword": True, "kk_relevant": "relevant", "kk_stance": None, "text": {"$nin": ["", None]}}

elif task_name == "serdil_relevant":
    label_list = ["relevant", "irrelevant"]
    idx_to_label = {i: lab for i,lab in enumerate(label_list)}
    encoder_path = "{}/models/best_models/encoder_dbmdz_bert-base-turkish-128k-cased_serdil_relevant_51.pt".format(repo_path)
    classifier_path = "{}/models/best_models/classifier_dbmdz_bert-base-turkish-128k-cased_serdil_relevant_51.pt".format(repo_path)

    # update has_kk_keyword field first
    if database_or_input_filename == "database":
        reg_list = []
        with open("{}/data/serdil_stance/serdil_keywords_only_name.txt".format(repo_path), "r", encoding="utf-8") as f:
            for line in f:
                if line:
                    reg_list.append(line[:-1])
            reg = re.compile("|".join(reg_list), flags=re.S|re.I)
            #reg = "|".join(reg_list)

        http_reg = re.compile(r"^(?!.*(https:|http:|www\.))", flags=re.S|re.I)

        keyword_true_query = {"has_serdil_keyword": None,
                              "$and": [{"text": {"$regex": http_reg}},
                                       {"text": {"$regex": reg}}]}
        tweet_col.update_many(keyword_true_query, {"$set": {"has_serdil_keyword": True}})
        keyword_false_query = {"has_serdil_keyword": None,
                               "$or": [{"text": {"$not": {"$regex": http_reg}}},
                                       {"text": {"$not": {"$regex": reg}}}]}
        tweet_col.update_many(keyword_false_query, {"$set": {"has_serdil_keyword": False}})

    query = {"has_serdil_keyword": True, "serdil_relevant": None, "text": {"$nin": ["", None]}}
    # query = {"kadikoy_tweets_to_be_processed": True, "has_serdil_keyword": True, "serdil_relevant": None, "text": {"$nin": ["", None]}}

elif task_name == "serdil_stance":
    label_list = ["pro", "against", "neutral"]
    idx_to_label = {i: lab for i,lab in enumerate(label_list)}
    encoder_path = "{}/models/best_models/encoder_dbmdz_bert-base-turkish-128k-cased_serdil_stance_58.pt".format(repo_path)
    classifier_path = "{}/models/best_models/classifier_dbmdz_bert-base-turkish-128k-cased_serdil_stance_58.pt".format(repo_path)
    # query = {"has_serdil_keyword": True, "serdil_relevant": "relevant", "serdil_stance": None, "text": {"$nin": ["", None]}}
    query = {"kadikoy_tweets_to_be_processed": True, "has_serdil_keyword": True, "serdil_relevant": "relevant", "serdil_stance": None, "text": {"$nin": ["", None]}}

elif task_name == "imamoglu_relevant":
    label_list = ["relevant", "irrelevant"]
    idx_to_label = {i: lab for i,lab in enumerate(label_list)}
    # encoder_path = "{}/models/best_models/encoder_dbmdz_bert-base-turkish-128k-cased_imamoglu_relevant_53.pt".format(repo_path)
    # classifier_path = "{}/models/best_models/classifier_dbmdz_bert-base-turkish-128k-cased_imamoglu_relevant_53.pt".format(repo_path)
    encoder_path = "{}/models/best_models/2023-08-29_imamoglu/encoder_dbmdz_bert-base-turkish-128k-cased_imamoglu_relevant_46.pt".format(repo_path)
    classifier_path = "{}/models/best_models/2023-08-29_imamoglu/classifier_dbmdz_bert-base-turkish-128k-cased_imamoglu_relevant_46.pt".format(repo_path)

    # update has_kk_keyword field first
    if database_or_input_filename == "database":
        reg_list = []
        with open("{}/data/imamoglu_stance/imamoglu_keywords.txt".format(repo_path), "r", encoding="utf-8") as f:
            for line in f:
                if line:
                    reg_list.append(line[:-1])
            reg = re.compile("|".join(reg_list), flags=re.S|re.I)
            #reg = "|".join(reg_list)

        http_reg = re.compile(r"^(?!.*(https:|http:|www\.))", flags=re.S|re.I)

        keyword_true_query = {"has_imamoglu_keyword": None,
                              "$and": [{"text": {"$regex": http_reg}},
                                       {"text": {"$regex": reg}}]}
        tweet_col.update_many(keyword_true_query, {"$set": {"has_imamoglu_keyword": True}})
        keyword_false_query = {"has_imamoglu_keyword": None,
                               "$or": [{"text": {"$not": {"$regex": http_reg}}},
                                       {"text": {"$not": {"$regex": reg}}}]}
        tweet_col.update_many(keyword_false_query, {"$set": {"has_imamoglu_keyword": False}})

    query = {"has_imam_keyword": True, "imamoglu_relevant": None, "text": {"$nin": ["", None]}}

elif task_name == "imamoglu_stance":
    label_list = ["pro", "against", "neutral"]
    idx_to_label = {i: lab for i,lab in enumerate(label_list)}
    # encoder_path = "{}/models/best_models/encoder_dbmdz_bert-base-turkish-128k-cased_imamoglu_stance_57.pt".format(repo_path)
    # classifier_path = "{}/models/best_models/classifier_dbmdz_bert-base-turkish-128k-cased_imamoglu_stance_57.pt".format(repo_path)
    encoder_path = "{}/models/best_models/2023-08-29_imamoglu/encoder_dbmdz_bert-base-turkish-128k-cased_imamoglu_stance_52.pt".format(repo_path)
    classifier_path = "{}/models/best_models/2023-08-29_imamoglu/classifier_dbmdz_bert-base-turkish-128k-cased_imamoglu_stance_52.pt".format(repo_path)
    query = {"has_imam_keyword": True, "imamoglu_relevant": "relevant", "imamoglu_stance": None, "text": {"$nin": ["", None]}}

elif task_name == "hilmi_relevant":
    label_list = ["relevant", "irrelevant"]
    idx_to_label = {i: lab for i,lab in enumerate(label_list)}
    encoder_path = "{}/models/best_models/encoder_dbmdz_bert-base-turkish-128k-cased_hilmi_relevant_53.pt".format(repo_path)
    classifier_path = "{}/models/best_models/classifier_dbmdz_bert-base-turkish-128k-cased_hilmi_relevant_53.pt".format(repo_path)

    # update has_kk_keyword field first
    if database_or_input_filename == "database":
        reg_list = []
        with open("{}/data/hilmi_stance/hilmi_keywords.txt".format(repo_path), "r", encoding="utf-8") as f:
            for line in f:
                if line:
                    reg_list.append(line[:-1])
            reg = re.compile("|".join(reg_list), flags=re.S|re.I)
            #reg = "|".join(reg_list)

        http_reg = re.compile(r"^(?!.*(https:|http:|www\.))", flags=re.S|re.I)

        keyword_true_query = {"has_hilmi_keyword": None,
                              "$and": [{"text": {"$regex": http_reg}},
                                       {"text": {"$regex": reg}}]}
        tweet_col.update_many(keyword_true_query, {"$set": {"has_hilmi_keyword": True}})
        keyword_false_query = {"has_hilmi_keyword": None,
                               "$or": [{"text": {"$not": {"$regex": http_reg}}},
                                       {"text": {"$not": {"$regex": reg}}}]}
        tweet_col.update_many(keyword_false_query, {"$set": {"has_hilmi_keyword": False}})

    query = {"has_hilmi_keyword": True, "hilmi_relevant": None, "text": {"$nin": ["", None]}}

elif task_name == "hilmi_stance":
    label_list = ["pro", "against", "neutral"]
    idx_to_label = {i: lab for i,lab in enumerate(label_list)}
    encoder_path = "{}/models/best_models/encoder_dbmdz_bert-base-turkish-128k-cased_hilmi_stance_57.pt".format(repo_path)
    classifier_path = "{}/models/best_models/classifier_dbmdz_bert-base-turkish-128k-cased_hilmi_stance_57.pt".format(repo_path)
    query = {"has_hilmi_keyword": True, "hilmi_relevant": "relevant", "hilmi_stance": None, "text": {"$nin": ["", None]}}

elif task_name == "uskudar_relevant":
    label_list = ["relevant", "irrelevant"]
    idx_to_label = {i: lab for i,lab in enumerate(label_list)}
    encoder_path = "{}/models/best_models/encoder_dbmdz_bert-base-turkish-128k-cased_uskudar_relevant_53.pt".format(repo_path)
    classifier_path = "{}/models/best_models/classifier_dbmdz_bert-base-turkish-128k-cased_uskudar_relevant_53.pt".format(repo_path)

    # update has_kk_keyword field first
    if database_or_input_filename == "database":
        reg_list = []
        with open("{}/data/uskudar_stance/uskudar_keywords.txt".format(repo_path), "r", encoding="utf-8") as f:
            for line in f:
                if line:
                    reg_list.append(line[:-1])
            reg = re.compile("|".join(reg_list), flags=re.S|re.I)
            #reg = "|".join(reg_list)

        http_reg = re.compile(r"^(?!.*(https:|http:|www\.))", flags=re.S|re.I)

        keyword_true_query = {"has_uskudar_keyword": None,
                              "$and": [{"text": {"$regex": http_reg}},
                                       {"text": {"$regex": reg}}]}
        tweet_col.update_many(keyword_true_query, {"$set": {"has_uskudar_keyword": True}})
        keyword_false_query = {"has_uskudar_keyword": None,
                               "$or": [{"text": {"$not": {"$regex": http_reg}}},
                                       {"text": {"$not": {"$regex": reg}}}]}
        tweet_col.update_many(keyword_false_query, {"$set": {"has_uskudar_keyword": False}})

    query = {"has_uskudar_keyword": True, "uskudar_relevant": None, "text": {"$nin": ["", None]}}

elif task_name == "uskudar_stance":
    label_list = ["pro", "against", "neutral"]
    idx_to_label = {i: lab for i,lab in enumerate(label_list)}
    encoder_path = "{}/models/best_models/encoder_dbmdz_bert-base-turkish-128k-cased_uskudar_stance_57.pt".format(repo_path)
    classifier_path = "{}/models/best_models/classifier_dbmdz_bert-base-turkish-128k-cased_uskudar_stance_57.pt".format(repo_path)
    query = {"has_uskudar_keyword": True, "uskudar_relevant": "relevant", "uskudar_stance": None, "text": {"$nin": ["", None]}}

else:
    raise("Task name {} is not known!".format(task_name))

# See if there is anything to predict
if database_or_input_filename == "database":
    num_tweets_to_predict = tweet_col.count_documents(query)
    if num_tweets_to_predict == 0:
        print("No documents to predict. Exiting...")
        sys.exit(0)

device = torch.device("cuda")

# OPTIONS
return_probabilities = False
positive_threshold = 0.5

# LOAD MODEL
tokenizer = AutoTokenizer.from_pretrained(pretrained_transformers_model)
config = AutoConfig.from_pretrained(pretrained_transformers_model)
has_token_type_ids = config.type_vocab_size > 1

encoder = AutoModel.from_pretrained(pretrained_transformers_model)
encoder.to(device)
encoder.load_state_dict(torch.load(encoder_path, map_location=device))
classifier = torch.nn.Linear(encoder.config.hidden_size, 1 if len(idx_to_label) == 2 else len(idx_to_label))
classifier.to(device)
classifier.load_state_dict(torch.load(classifier_path, map_location=device))

encoder = torch.nn.DataParallel(encoder)
encoder.eval()
classifier.eval()

def preprocess(text): # Preprocess text (username and link placeholders)
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def model_predict(batch):
    if has_token_type_ids:
        input_ids, input_mask, token_type_ids = tuple(t.to(device) for t in batch.values())
        embeddings = encoder(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)[1]
    else:
        input_ids, input_mask = tuple(t.to(device) for t in batch.values())
        embeddings = encoder(input_ids, attention_mask=input_mask)[1]

    out = classifier(embeddings)

    if len(idx_to_label) == 2:
        preds = torch.sigmoid(out).detach().cpu().numpy().flatten()
        if return_probabilities:
            preds = [round(float(x), 4) for x in preds]
        else:
            preds = [idx_to_label[int(x >= positive_threshold)] for x in preds]

    else:
        out = out.detach().cpu().numpy()
        if return_probabilities:
            preds = [probs for probs in softmax(out, axis=1).tolist()] # a list of lists(of probabilities)
        else:
            preds = [idx_to_label[pred] for pred in np.argmax(out, axis=1).tolist()] # a list of labels

    return preds

# !!!IMPORTANT!!!
# Change this according to the json line format.
# Here the format for every line is like:
# {id_str: text}
def read_json_line(data):
    id_str = list(data.keys())[0]
    text = preprocess(data[id_str])

    return id_str, text

if __name__ == "__main__":
    # TODO: add progress bar
    total_processed = 0
    if database_or_input_filename == "database": # if database
        tweets_to_predict = tweet_col.find(query, ["_id", "text"])

        curr_batch = []
        for i, tweet in enumerate(tweets_to_predict):
            id_str = tweet["_id"]
            # text = preprocess(tweet["text"]) # we may want some usernames here, so we don't preprocess
            text = tweet["text"]

            if len(text) > 0:
                total_processed += 1
                curr_batch.append({"_id": id_str, "text": text})

            if len(curr_batch) == batch_size:
                texts = [d["text"] for d in curr_batch]
                inputs = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True,
                                   max_length=max_seq_length)
                preds = model_predict(inputs)
                # TODO: Think about multiple updates at the same time
                for pred_idx, pred in enumerate(preds):
                    curr_d = curr_batch[pred_idx]
                    tweet_col.update_one({"_id": curr_d["_id"]}, {"$set": {task_name: pred}})

                curr_batch = []

        # Last incomplete batch, if any
        if len(curr_batch) != 0:
            texts = [d["text"] for d in curr_batch]
            inputs = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True,
                               max_length=max_seq_length)
            preds = model_predict(inputs)
            for pred_idx, pred in enumerate(preds):
                curr_d = curr_batch[pred_idx]
                tweet_col.update_one({"_id": curr_d["_id"]}, {"$set": {task_name: pred}})


        # # for random prediction
        # number_to_predict = 250
        # total_number = tweet_col.count_documents(query)

        # curr_batch = []
        # for i, tweet in enumerate(tweets_to_predict):
        #     if random.random() > (number_to_predict/total_number): continue

        #     id_str = tweet["_id"]
        #     text = tweet["text"]

        #     if len(text) > 0:
        #         curr_batch.append({"_id": id_str, "text": text})

        # texts = [d["text"] for d in curr_batch]
        # inputs = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True,
        #                    max_length=max_seq_length)
        # preds = model_predict(inputs)
        # for pred_idx, pred in enumerate(preds):
        #     curr_d = curr_batch[pred_idx]
        #     tweet_col.update_one({"_id": curr_d["_id"]}, {"$set": {"erdogan_irrelevant_stance": pred}})


    else: # if filename
        output_file = open(output_filename, "w", encoding="utf-8")
        if database_or_input_filename.endswith(".json.gz"):
            input_file = gzip.open(database_or_input_filename, "rt", encoding="utf-8")
        elif database_or_input_filename.endswith(".json"):
            input_file = open(database_or_input_filename, "r", encoding="utf-8")
        else:
            raise("File extension should be 'json' or 'json.gz'!")

        curr_batch = []
        for i, line in enumerate(input_file):
            data = json.loads(line)
            id_str, text = read_json_line(data)

            if len(text) > 0:
                total_processed += 1
                curr_batch.append({"_id": id_str, "text": text})

            if len(curr_batch) == batch_size:
                texts = [d["text"] for d in curr_batch]
                inputs = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True,
                                   max_length=max_seq_length)
                preds = model_predict(inputs)
                for pred_idx, pred in enumerate(preds):
                    curr_d = curr_batch[pred_idx]
                    curr_d.pop("text") # No need for text in the output.
                    curr_d[task_name] = pred
                    output_file.write(json.dumps(curr_d, ensure_ascii=False) + "\n")

                curr_batch = []

        input_file.close()

        # Last incomplete batch, if any
        if len(curr_batch) != 0:
            texts = [d["text"] for d in curr_batch]
            inputs = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True,
                               max_length=max_seq_length)
            preds = model_predict(inputs)
            for pred_idx, pred in enumerate(preds):
                curr_d = curr_batch[pred_idx]
                curr_d.pop("text") # No need for text in the output.
                curr_d[task_name] = pred
                output_file.write(json.dumps(curr_d, ensure_ascii=False) + "\n")

        output_file.close()

    print("Processed {} tweets in total.".format(str(total_processed)))
