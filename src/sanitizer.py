import re
import string

def sanitize(dataset):
    result = []

    for row in dataset:
        #Method of removing emojis: turn into ascii, ignore anything not ascii, and decode once again.
        cleaned = row["text"].encode('ascii', 'ignore').decode('ascii')

        #some tweets have "&amp" at random spots
        cleaned = re.sub("&amp", "", cleaned)

        #some tweets contain urls
        cleaned = re.sub(r"https?://\S+", "", cleaned)  #http followed by maybe an "s" followed by ":" and any non-whitespace character once or more

        #cleaning punctuation
        cleaned = re.sub(r"[^\w\s]", "", cleaned)   #anything except alphanumeric characters or whitespace

        result.append({
            "tweet_id" : row["tweet_id"],
            "text" : cleaned,
            "q1_label" : row["q1_label"]
        })

    return result