import mmh3
import PIL
import langid
from PIL import Image
import os, io
import multiprocessing
import glob, random
import subprocess
from pathlib import Path
from tqdm import tqdm
import json
import tarfile
#from utils import classify_and_quality_score
import multiprocessing, functools, json, glob
from flagged_words import *
from urllib.parse import urlparse
import string
import argparse
from collections import defaultdict
import sys
import pyarrow.parquet as pq
import time, random
import json, os, glob, random
import multiprocessing
from multiprocessing import set_start_method
import os
try:
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))         
except:
    pass
import html
try:
    import ctranslate2
except:
    pass
import transformers
import  itertools
import os
import glob, random
import subprocess
from pathlib import Path
from tqdm import tqdm
import json
import json, glob, os
import math
import functools, json, glob
from flagged_words import *
from collections import Counter
import string
import argparse
from collections import defaultdict
import sys
import pyarrow.parquet as pq
import time, random
import json, os, glob, random
from torch import multiprocessing
from torch.multiprocessing import SimpleQueue
from torch import threading
import os
import json, os, glob
from tqdm import tqdm
from typing import List
import glob, json
import re
from huggingface_hub import hf_hub_download
import fasttext
from multiprocessing import Pool
from names import *

import multiprocessing, functools, json, glob, langid

import glob, json, langid, random
import math
import sys, os, string


import wget
import spacy    
from nltk.corpus import wordnet as wn
from lemminflect import getInflection
import stdnum
from date_detector import Parser
import commonregex, re
from commonregex import CommonRegex
from faker import Faker
import spacy
#from matplotlib import colors
import fasttext
#    from frcnn.visualizing_image import SingleImageViz
#    from frcnn.processing_image import Preprocess as FRCNNPreprocess
#    from frcnn.modeling_frcnn import GeneralizedRCNN
#    from frcnn.utils import Config as FRCNNConfig
#    from frcnn.utils import decode_image as frcnn_decode_image
#    import cv2
from nltk.corpus import cmudict
from autocorrect import Speller


import re
import sys, os
import re 
import random
from string import punctuation, ascii_lowercase
import gzip
import tqdm
from time import sleep
from typing import Dict, List
import os
from pathlib import Path
from tqdm import tqdm
import copy
import json
import base64
import uuid
import hashlib
import random
from io import BytesIO
import numpy as np
from numpy import asarray
from collections import deque, Counter
import numpy as np
import torch
#import torchvision
#from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import random
import itertools
import torch
import PIL
from PIL import Image
from transformers import pipeline
#from datasets import load_dataset
from torch.nn.functional import cosine_similarity
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoTokenizer, AutoModelWithLMHead
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import numpy as np
import whoosh.index as whoosh_index
from whoosh.qparser import QueryParser
from whoosh.analysis import StemmingAnalyzer, Filter


from collections import OrderedDict
from string import punctuation, ascii_lowercase
from spacy.tokens import Doc, Span


### BASIC UTILITIES

from collections import Counter


def get_ngram(text, window_size=3, lang=""):
    if not lang:
        if cjk_detect(text[:min(len(text), 100)]):
            lang = 'zh'
        else:
            lang = 'en'
    if lang in {"zh", "ja", "ko", "th", "jap"}:
        tokens = text
        ret = [
            "".join(tokens[i : i + window_size])
            for i in range(len(tokens) - window_size)
        ]
        ret = [
        "".join(tokens[i : i + window_size]) for i in range(len(tokens) - window_size)
        ]
        
    else:
        tokens = text.split(" ")
        ret = [
            " ".join(tokens[i : i + window_size]) for i in range(len(tokens) - window_size)
        ]
    return Counter(ret)


def high_ngram(text, cutoff=0.15, window_size=3, lang=""):
    if not lang:
        if cjk_detect(text[:min(len(text), 100)]):
            lang = 'zh'
        else:
            lang = 'en'
    aHash = get_ngram(text, window_size, lang)
    text_len = text.count(" ") + 1
    for key in list(aHash.keys()):
        aHash[key] = aHash[key] / text_len
    return any(a for a in aHash.values() if a > cutoff)

def cjk_detect(text):
    """
    Detects if a text contains characters from specific East Asian languages (Chinese, Japanese, Korean, Thai, and Traditional Javanese).

    Args:
        text (str): Text to check.

    Returns:
        str or None: Language code if detected; otherwise, None.
    """
    # chinese
    if re.search("[\u4e00-\u9FFF]", text):
        return "zh"
    # korean
    if re.search("[\uac00-\ud7a3]", text):
        return "ko"
    # japanese
    if re.search("[\u3040-\u30ff]", text):
        return "ja"
    # thai
    if re.search("[\u0E01-\u0E5B]", text):
        return "th"
    # traditional javanese
    if re.search("[\uA980-\uA9DF]", text):
       return "jv_tr"
    return None

def fix_too_much_ngram(text, window_size=3, lang="en", threshold=2, logger=None):
    global all_stopwords
    stopwords =  all_stopwords.get(lang, all_stopwords['en'])
    for word, cnt in get_ngram(text, window_size=3, lang="").items():
        if cnt >= threshold:
            word_arr = word.split()
            if not any(w for w in word_arr if len(w) > 3 and w.lower() not in stopwords and w.lower()[:min(len(w), 4)] not in {'said', 'says', 'sayi', 'menti', 'disc', 'talk', 'desc', 'hear', 'speak',}) :
                continue
            if word not in text:
                if logger: logger.warning(("NGRAM NOT IN TEXT", word, text))
                continue
            i = text.index(word)
            text = text[:i+1]+text[i+1:].replace("and "+word, "").replace("or "+word, "").replace(", "+word, "").replace(" "+word, "")
    text = text.split(" ")
    len_text = len(text)
    for i in range(len_text):
        if i < len_text - 2:
            if text[i] == text[i+1] and text[i] == text[i+2]:
                text[i] = None
    text = " ".join(t for t in text if t is not None)
    return text


numbering_list = ['3', '7)', '7.', '4', 'iii.', 'iii-', '8.', '4-', 'v:', 'I:', 'ii.', 'i.', 'V)', 'E)', 'I)', 'III.', 'III)', '2-', '1)', 'v-', 'III', 'I.', 'c)', '1.', 'V-', 'iv)', 'A)', 'v)', 'IV', 'C.', 'ii)', 'I', 'IV.', 'C)', 'II-', '2.', 'III-', 'IV)', 'd)', 'iii', 'i-', 'iii:', 'A.', 'B.', '1', '6)', 'ii', '8)', '3)', 'e)', 'ii-', '5-', 'II)', 'iv-', '2)', 'e.', 'IV:', 'III:', 'i)', '10.', 'V', 'V.', 'v.', 'D)', 'E.', 'iv:', 'B)', 'II', 'ii:', 'V:', 'a.', '5.', 'IV-', '9.', 'D.', '3.', '4:', '2:', 'i', 'II.', '3-', '2', 'c.', 'a)', '3:', '10)', 'd.', 'i:', 'iv.', '1-', '4.', '5', 'iv', 'iii)', 'b.', '1:', 'II:', 'v', '5:', '6.', 'b)', 'I-', '9)', '4)', '5)']

stopwords_list = ['es', 'ing', 'ed', 'include', 'includes', 'also', 'haven', 'are', 'why', 'most', "won't", 'against', 'with', 'needn', 'couldn', 'now', 'mustn', 'who', 'under', 'doing', 'am', 'aren', 'they', "didn't", 'd', 'doesn', 'if', 'he', 'her', "haven't", 'isn', 'own', 'does', 'such', 'until', 'into', 'had', 'again', 'over', "hadn't", "you'll", 't', 'by', 'be', "wasn't", 'so', 'yours', 'both', 'any', 'did', "you've", 'these', 'myself', 'o', 'hasn', "isn't", 'you', 'other', 'shan', 'being', 'yourselves', 'was', 'no', 'm', 'those', 'will', 'its', 'itself', 'have', 'down', 'weren', 'having', 'wouldn', 'herself', "mustn't", 'very', 'do', "should've", 'him', "you'd", 'below', 'just', 'that', 'for', 'which', 'but', 'nor', 'all', 'then', 'i', 'whom', 'it', 'once', 'here', 've', "you're", 'ours', "that'll", 'a', 'won', 'himself', 'where', 'this', 'your', "hasn't", 'same', 'when', 'ourselves', 'because', "needn't", 'theirs', 'from', 'mightn', 'my', 'while', 'yourself', "she's", 'each', "doesn't", 'only', 'at', 's', 'their', "wouldn't", 'shouldn', 'and', 'themselves', 'hers', 'has', 'up', 'ma', 'in', 'll', 'we', 're', 'y', 'of', 'after', 'our', "shan't", 'before', 'wasn', 'can', 'should', 'been', 'through', 'as', 'further', 'during', 'between', 'there', 'me', 'on', 'don', "shouldn't", 'more', 'out', "don't", 'the', "weren't", "aren't", "it's", 'what', 'or', "couldn't", 'hadn', "mightn't", 'his', 'above', 'to', 'how', 'few', 'off', 'them', 'didn', 'ain', 'not', 'she', 'an', 'than', 'too', 'is', 'some', 'were', 'about']

all_stopwords['en'] = set(stopwords_list + numbering_list + list(all_stopwords['en']))

# we use the old stopwords set here for backwards compatability
stopwords_set = set(stopwords_list + numbering_list)

def strip_left_stopwords(e_text, lang="en"):
  """
  Removes common stopwords from the left side of a text until a significant word is found.

  Args:
      e_text (str): The text to strip from the left side.

  Returns:
      str: Text with left-side stopwords removed.
  """
  e_text2 = []
  add_rest = False
  stopwords =  all_stopwords.get(lang, all_stopwords['en'])  
  for et in e_text.split():
      etl = et.lower()
      if add_rest or ((etl not in stopwords and etl not in common_title_words_set) or etl.strip(".") in {"a", "an", "united", "the", "new", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",  "asian", "american", "african", "european", }):
        add_rest = True
        e_text2.append(et)
  return " ".join(e_text2)


def strip_right_stopwords(e_text, lang="en"):
  """
  Removes common stopwords from the right side of a text until a significant word is found.

  Args:
      e_text (str): The text to strip from the right side.

  Returns:
      str: Text with right-side stopwords removed.
  """
  e_text2 = []
  add_rest = False
  e_text_arr = e_text.split()
  e_text_arr.reverse()
  stopwords =  all_stopwords.get(lang, all_stopwords['en'])    
  for et in e_text_arr:
      etl = et.lower()      
      if add_rest or (etl not in stopwords or etl.strip(".") in {"act", "code", "statute", "regulation", "regulations", "percent", "feet", "foot", "square", "barrells", "hour", "hours", "people", "asian", "american", "african", "european", "act", "law", "facilities", "facility", "center", "square", "rd", "street", "way", "blvd", "ave", "avenue", "states", "kingdom", "court", "corp", "corporation", "co", "company", "ltd", "llc", "llp", "incorp.", "incorporated"}):
        add_rest = True
        e_text2.append(et)
  return " ".join(reversed(e_text2))

def generate_4grams(sentence):
    words = sentence.split()
    fourgrams = [" ".join([words[i], words[i+1], words[i+2], words[i+3]]) for i in range(len(words) - 3)]    
    return fourgrams

def generate_trigrams(sentence):
    words = sentence.split()
    trigrams = [" ".join([words[i], words[i+1], words[i+2]]) for i in range(len(words) - 2)]    
    return trigrams

def generate_bigrams(sentence):
    words = sentence.split()
    bigrams = [" ".join([words[i], words[i+1]]) for i in range(len(words) - 1)]    
    return bigrams


def lang_is_cjk(lang):
  return lang in {"ko", "zh", "ja"}

lang_2_max_stopword_len = dict([(lang, max(s.count(" ")+1 if not lang_is_cjk(lang) else len(s) for s in arr)) for lang, arr in all_stopwords.items()])

def get_stopword_score(text, lang="en", max_word_len=3, cjk_scale=1.5, window=500):
    is_cjk = lang_is_cjk(lang)
    stopwords =  all_stopwords.get(lang, all_stopwords['en'])
    if not stopwords: return 1
    if window:
        text = text[:min(len(text), window)]
        if not is_cjk:
            text = text.split(" ")[:-1]
            text = " ".join(text)
    text = text.lower().strip()
    if is_cjk:
      s_arr = list("".join(text.split()))
    else:
      s_arr = text.split()
    word_len = lang_2_max_stopword_len.get(lang, max_word_len)
    len_s = len(s_arr)
    stop_cnt = 0
    total_cnt = 0
    for i in range(len_s):
      if s_arr[i] is None: continue
      for j in range(min(len_s, i+word_len), i, -1):
        word = "".join(s_arr[i:j]) if is_cjk else " ".join(s_arr[i:j])
        if word in stopwords:
          stop_cnt += 1
          s_arr[i] = "".join(s_arr[i:j]) if is_cjk else " ".join(s_arr[i:j])
          for k in range(i+1, j):
            s_arr[k] = None
          break
      total_cnt += 1
    if total_cnt == 0:
      return 0
    stopword_score =  (stop_cnt/total_cnt)
    if is_cjk: stopword_score = stopword_score*cjk_scale
    return (stopword_score)



def get_special_char_score (text, lang="en", special_characters_default=None, window=500):
  global junk
  if len(text) == 0: return 1
  #TODO: do we want to do any lang specific special_chars?
  if special_characters_default is None: special_characters_default = junk
  if window:
        text = text[:min(len(text), window)]
  ret =  len([a for a in text if a in special_characters_default])/len(text)
  if lang_is_cjk(lang):
    return ret/5
  else:
    return ret

common_title_words_set = {'introduction', 'conclusion', 'section', 'chapter', 'works', 'notes', 'note', 'further', 'see', 'references', 'reference', 'section', 'title', 'conclusion', 'intro', 'introduction', 'executive', 'summary', 'key', 'plot', 'theme'}
lang_2_max_flaggedword_len = dict([(lang, max(s.count(" ")+1 if  lang not in {"zh", "ja", "ko"} else len(s) for s in arr)) for lang, arr in flagged_words.items()])

def get_flaggedword_score(text, lang="en", max_text_len=1000, max_word_len=3):
    #if len(text) > max_text_len:
    #    text = text[:max_text_len]
    max_word_len = max(lang_2_max_flaggedword_len.get(lang, max_word_len), max_word_len)
    flaggedwords1 = flagged_words.get(lang, {})
    flaggedwords2 = flagged_words.get("en", {})
    bannedwords1 = banned_words.get(lang, {})
    bannedwords2 = banned_words.get("en", {})
    stopwords1 = all_stopwords.get(lang, {})
    stopwords2 = all_stopwords.get("en", {})
    is_cjk = lang in {"zh", "ja", "ko"}
    if is_cjk:
      text = " ".join([a for a in text.lower().split() if a not in stopwords2])
    text = text.lower().strip().replace(",", "").replace(".", "").replace("\"", "").replace("'", "")
    if is_cjk:
      s_arr = [a for a in [s.strip(strip_chars) for s in list(text)] if a]
    else:
      s_arr = [s.strip(strip_chars) for s in text.split()]
    len_s = len(s_arr)
    banned_score = 0
    flagged_score = 0
    hate_score = 0
    total_cnt = 0
    for i in range(len_s):
      if s_arr[i] is None: continue
      word_len = max_word_len
      for j in range(min(len_s, i+word_len),i,-1):
        if is_cjk:
          word = "".join([s for s in s_arr[i:j] if s])
        else:    
          word = " ".join([s for s in s_arr[i:j] if s])
        if not word: break
        is_flagged = word in flaggedwords1 or word in flaggedwords2
        is_hate = word in hatewords
        is_banned = word in bannedwords1 or word in bannedwords2 
        if is_flagged or is_banned or is_hate:
          if is_flagged: flagged_score += 1
          if is_banned: banned_score += 1
          if is_hate: hate_score += 1
          s_arr[i] =  word
          for k in range(i+1, j):
            s_arr[k] = None
      s = s_arr[i]
      if s not in stopwords1 and (is_cjk or len(s) > 3):
        total_cnt += 1
    if total_cnt == 0: total_cnt = 1
    flagged_score = (flagged_score/total_cnt)
    hate_score = hate_score/total_cnt
    banned_score = banned_score/total_cnt
    return  flagged_score, banned_score, hate_score


docHash = {}
sentHash = {}
prefixHash = {}
# WARNING: don't do dedup twice in the same process!!
def dedup(data, delete_sent_if_greater_percentage=0.05, delete_doc_if_sent_greater_percentage=0.75, strip_header_footers=True):
  #print ("dedup")
  if random.randint(0,1000000)==0:
    for key in list(prefixHash.keys()):
      if prefixHash[key] <= 3 or random.randint(0,1):
        del prefixHash[key]
  if random.randint(0,10000000)==0:
    for key in list(sentHash.keys()):
      if sentHash[key] <= 3 or random.randint(0,1):
        del sentHash[key]
  if random.randint(0,100000000)==0:
    for key in list(docHash.keys()):
      if docHash[key] <= 3 or random.randint(0,1):
        del docHash[key]
  new_text = []
  new_meta = []
  for text, metadata in zip(data['text'].split("<|endoftext|>"), data['metadata']):
      
      lang = metadata['lang']
      if type(lang) is list:
          metadata['langs'] = lang
          lang = metadata['lang'] = lang[0]
      is_cjk = lang in {"zh", "ja", "ko"}
      text0 = text
      text = text.strip()
      text_arr =  text.lower().split()
      text2 = [a.strip(strip_chars) for a in text_arr[20:min(len(text_arr), 220)]][:-1]  
      text2 = [a[:4] if len(a) > 4 else a for a in text2 if len(a) > 1]
      code = hash("".join(text2))
      if code in docHash:
          # duplicate document
          docHash[code] = docHash.get(code, 0) + 1
          continue
      docHash[code] = docHash.get(code, 0) + 1
      doc_code = code
      # sentence dedup
      text2 = text
      if lang == "hi":
          text2 = text2.replace("|", ". ")
      text2 = text2.replace("。",". ").replace("|>", "|>. ").replace("<|", ". <|").replace(".\n", ". ").replace("? ", "?. ").replace("! ", "!. ").replace("; ", ";. ").replace("| ", "|. ").replace("\n", ". ").\
          replace("1>", "1>. ").replace("2>", "2>. ").replace("3>", "3>. ").replace("4>", "4>. ").replace("5>", "5>. ").replace("6>", "6>. ").replace("7>", "7>. ").\
          replace("8>", "8>. ").replace("9>", "9>. ").replace("0>", "0>. ").replace("<", ". <").replace("<image>", "<image>. ").replace("<audio>", "<audio>. ").replace(".</", ". </")

      l_arr = [l2 for l2 in text2.split(". ") if len(l2) > 30]
      num_sents = len(list(set(l_arr)))
      sent_dups = []
     
      for l2, cnt in Counter(l_arr).items():
        # there can be many repeated sentences in the same example because we are concatenating similar documents.
        # we need to decide what to do about this.
        l3 = l2.strip("|.。?!").lower()
        code4 = hash(l3)
        score1 = get_special_char_score(l2, lang)
        score2 = get_stopword_score(l2, lang)
        #if code4 in sentHash:
        #    print (("found dup", score1, score2, l2))
        if not ('math' in metadata['source'] or "{" in l3 or "\ndef " in l3 or "):" in l3 or  "${" in l3):
            if code4 in sentHash and sentHash[code4] > 2:
                if (((score1 > 0.1 or score2 < 0.05)) or "http" in l3 or "terms of use" in l3 or "view pdf" in l3 or "view doc" in l3 or "print this page" in l3 or "read more" in l3 or " click " in " "+l3 or "privacy policy" in l3 or "jump to" in l3 or "disclaimer" in l3 or "you are here" in l3 or "send email" in l3 or "this page" in l3 or "this site" in l3 or "more info" in l3 or "this website" in l3 or "our site" in l3 or "our website" in l3 or "the link" in l3 or "visiting www" in l3 or "apache license" in l3 or "fandom" in l3 or "foodista" in l3 or "wiki" in l3 or "gutenberg.org" in l3 or "creative commons" in l3 or "cc-by" in l3 or "from wiki" in l3 or "free media" in l3 or "cookies" in l3  or "Creative Commons Attribution" in l2 or  "(CC-BY) 4.0 License" in l2):
                    text = text.replace(l2+". ", "")
                    text = text.replace(l2+".", "")                    
                    text = text.replace(l2+" ", "")
                    text = text.replace(l2, "")
            if code4 in sentHash:
              sent_dups.append(l2)
            sentHash[code4] = sentHash.get(code4, 0) + 1
            
      if num_sents and len(sent_dups)/num_sents >= delete_doc_if_sent_greater_percentage:
          continue
      if False:
        if num_sents and len(sent_dups)/num_sents >= delete_sent_if_greater_percentage:
          for l2 in sent_dups:
            text = text.replace(l2+". ", "")
            text = text.replace(l2+".", "")                    
            text = text.replace(l2+" ", "")
            text = text.replace(l2, "")
      if strip_header_footers:
          sent_dups = []
          # prefix dups
          for i in range(3):
            if is_cjk:
                text_arr = list(text)
                join_char = ""
            else:
                text_arr = text.split(" ")
                join_char = " "                
            if  any(a for a in text_arr[:min(len(text_arr), 4)] if ">" in a and "<" in a): continue
            text2 = [a.strip(strip_chars) for a in text[:min(len(text), 300)].lower().split(" ")]
            code = hash("".join(text2[:6]))
            if prefixHash.get(code,0) >= 3:
              pattern = " ".join(text.split(" ")[:6])
              text = " ".join(text.split(" ")[6:]).replace(pattern, "").strip()
              prefixHash[code] = prefixHash.get(code, 0) + 1
              continue
            else:
              prefixHash[code] = prefixHash.get(code, 0) + 1
            code = hash("".join(text2[:5]))
            if prefixHash.get(code,0) >= 4:
              pattern = " ".join(text.split(" ")[:5])
              text = " ".join(text.split(" ")[5:]).replace(pattern, "").strip()
              prefixHash[code] = prefixHash.get(code, 0) + 1
              continue
            else:
              prefixHash[code] = prefixHash.get(code, 0) + 1
            code = hash("".join(text2[:4]))
            if prefixHash.get(code,0) >= 5:
              pattern = " ".join(text.split(" ")[:4])
              text = " ".join(text.split(" ")[4:]).replace(pattern, "").strip()
              prefixHash[code] = prefixHash.get(code, 0) + 1
              continue
            else:
              prefixHash[code] = prefixHash.get(code, 0) + 1
            break

          for i in range(3):
            text_arr = text.split(" ")              
            if  any(a for a in text_arr[max(0, len(text_arr)-4):] if ">" in a and "<" in a): continue              
            text2 = [a.strip(strip_chars) for a in reversed(text.split(" ")) ][:-1]
            code = hash("".join(text2[:6]))
            if prefixHash.get(code,0) >= 3:
              text = " ".join(text.split(" ")[:-6]).strip()
              prefixHash[code] = prefixHash.get(code, 0) + 1
              continue
            else:
              prefixHash[code] = prefixHash.get(code, 0) + 1
            code = hash("".join(text2[:5]))
            if prefixHash.get(code,0) >= 4:
              text = " ".join(text.split(" ")[:-5]).strip()
              continue
            else:
              prefixHash[code] = prefixHash.get(code, 0) + 1
            code = hash("".join(text2[:4]))
            if prefixHash.get(code,0) >= 5:
              text = " ".join(text.split(" ")[:-4]).strip()
              continue
            else:
              prefixHash[code] = prefixHash.get(code, 0) + 1
            break
          if not text: continue
          text = text.strip()
          text = text[0].upper()+ text[1:]
          text_arr =  text.lower().split()
          text2 = [a.strip(strip_chars) for a in text_arr[20:min(len(text_arr), 80)]][:-1]  
          text2 = [a[:4] if len(a) > 4 else a for a in text2 if len(a) > 1]
          code = hash("".join(text2))
          if code != doc_code:
            if code in docHash :
              # duplicate document
              docHash[code] = docHash.get(code, 0) + 1
              continue
            docHash[code] = docHash.get(code, 0) + 1
      new_text.append(text)
      new_meta.append(metadata)
      
  if not new_text: return None
  data['text'] = "<|endoftext|>".join(new_text)
  data['metadata'] = new_meta
  return data

common_pile_sites = None
white_list_sites = None
def is_idx_match(data):
    global common_pile_sites, white_list_sites
    if white_list_sites is None: # common_pile_sites
        print ("loading sites")
        #common_pile_sites = set(json.load(open(os.path.abspath(os.path.dirname(__file__))+"/"+"common_pile_urls.json")))
        white_list_sites = set(json.load(open(os.path.abspath(os.path.dirname(__file__))+"/"+"white_list_urls.json")))
        print ("loaded sites list")
    if 'idx' not in data and 'url' in data:
      data['idx'] = data['url']
    if "://" in data['idx'].split("://",1)[-1]: return None, None
    idx = data['idx'].split("://",1)[-1]
    # data from loc.gov is often garbled
    if  "loc.gov" in idx or 'slashdot.org' in idx or "yahoo.com" in idx or "google.com" in idx or "amazon.com" in idx or "cnbc.com" in idx or "facebook.com" in idx or "youtube.com" in idx or "instagram.com" in idx or "twitter.com" in idx or "facebook.com" in idx or "whatsapp.com" in idx or "microsoft.com" in idx or "reddit.com" in idx or "yahoo.co.jp" in idx or "tiktok.com" in idx or "baidu.com" in idx or "linkedin.com" in idx or "netflix.com" in idx or "pornhub" in idx or "xxx" in idx or "dzen.ru" in idx or "naver.com" in idx or "live.com" in idx or "bet.br" in idx or "office.com" in idx or "bing.com" in idx or "bilibili.com" in idx or "pinterest.com" in idx or "xvideos.com" in idx or "twitch.tv" in idx or "xhamster.com" in idx or "temu.com" in idx or "vk.com" in idx or "mail.ru" in idx or "sharepoint.com" in idx or "weather.com" in idx or "samsung.com" in idx or "globo.com" in idx or ".t.me/" in idx or "canva.com" in idx or "duckduckgo.com" in idx or "xnxx.com" in idx or "xhamster43.desi" in idx or "nytimes.com" in idx or "deepseek.com" in idx or "zoom.us" in idx or "stripchat.com" in idx or "quora.com" in idx:
        return None, None        
    if len(idx) > 80:
        idx = idx[:80]
    if "/" not in idx:
        idx = idx+"/"

        
    is_oss =   ("elifesciences.org/" in idx or ".biodiversitylibrary.org/" in idx or ".free.law/" in idx or ".europeana.eu/" in idx or ".publicdomainreview.org/" in idx or ".wisdomcommons.org/" in idx or ".intratext.com/" in idx or "wikimedia.com" in idx or ".mediawiki.org/" in idx or ".wikimedia.org/" in idx or ".wikidata.org/" in idx or \
                ".wikipedia.org/" in idx or ".wikisource.org/" in idx or ".wikifunctions.org/" in idx or ".wikiquote.org/" in idx or ".wikinews.org/" in idx or ".wikivoyage.org/" in idx or ".wiktionary.org/" in idx or ".wikibooks.org/" in idx or \
                ".mediawiki.com/" in idx or ".wikimedia.com/" in idx or ".wikidata.com/" in idx or \
                ".wikipedia.com/" in idx or ".wikisource.com/" in idx or ".wikifunctions.com/" in idx or ".wikiquote.com/" in idx or ".wikinews.com/" in idx or ".wikivoyage.com/" in idx or ".wiktionary.com/" in idx or ".wikibooks.com/" in idx or \
                ".courtlistener.com/" in idx or ".case.law/" in idx or \
                "pressbooks.oer.hawaii.edu/" in idx or ".huggingface.co/docs/" in idx or \
                ".opencourselibrary.org/" in idx or ".medbiq.org/" in idx or ".doabooks.org/" in idx or ".bccampus.ca/" in idx or \
                "open.umn.edu/opentextbooks/" in idx or "www.gutenberg.org/" in idx or ".mozilla.org/"  in idx or "www.eclipse.org/" in idx or \
                ".apache.org/" in idx or ".python.org/" in idx or ".pytorch.org/" in idx or ".numpy.org/" in idx or ".scipy.org/" in idx or ".opencv.org/" in idx or \
                ".scikit-learn.org/" in idx or ".pydata.org/" in idx or ".matplotlib.org/" in idx or ".palletsprojects.com/" in idx or \
                ".sqlalchemy.org/" in idx or ".pypi.org/" in idx or ".sympy.org/" in idx or ".nltk.org/" in idx or \
                ".scrapy.org/" in idx or ".owasp.org/" in idx or \
               ".creativecommons.org/" in idx or "stackoverflow." in idx or "stackexchange." in idx  or  'askubuntu.' in idx or "mathoverflow." in idx or "superuser." in idx or "stackapps." in idx or "serverfault." in idx or \
                ".wikia.com/" in idx or ".foodista.com/" in idx or ".fandom.com/" in idx or ".attack.mitre.org/" in idx)
    text = data['text']
    head = text[:100].lower()
    tail = text[-100:].lower()
    
    if is_oss or idx in white_list_sites or \
       ".mil/" in idx or ".vlada.mk" in idx or ".vlada.cz" in idx or ".kormany.hu" in idx or  "regeringen." in idx or ".rijksoverheid.nl" in idx or ".government.nl" in idx or ".regeringen.se" in idx or  ".regeringen.dk" in idx or  ".regeringen.no" in idx or ".bund.de" in idx or ".bundesregierung.de" in idx or  ".government.ru" in idx or ".gc.ca" in idx or \
       ".admin.ch" in idx or  'www.gob.cl/' in idx or  'www.gob.ec/' in idx or  'guatemala.gob.gt/' in idx or  'presidencia.gob.hn/' in idx or  'www.gob.mx/' in idx or  'presidencia.gob.pa/' in idx or  'www.gob.pe/' in idx or  'gob.es/' in idx or  'argentina.gob.ar/' in idx or \
        "tanzania.go.tz/" in idx or ".indonesia.go.id/" in idx or ".go.kr/" in idx or ".go.jp/" in idx or  "thailand.go.th/" in idx or ".europa.eu/" in idx or ".un/" in idx or ".int/" in idx or ".govt." in idx or "www.gub.uy" in idx  or idx.endswith(".gov") or ".gov/" in idx or '.gov.' in idx or '.gouv.' in idx:
        
        if "ymca.int" in idx: return None, None
        return True, is_oss
    if  idx not in white_list_sites and ("cc-0" in head or "creative common"  in head or "cc-by" in head or  "creative common"  in tail or "cc-by" in tail or "cc-0" in head):
        if not filter_copyright_and_content_issues(data):
            return True, True
    return None, None

def filter_copyright_and_content_issues(data):
      text = data['text']
      if "gutenberg.org" not in data['idx'] and \
          ".gov/" not in data['idx'] and ".mil/" not in data['idx'] and ".go.jp" not in data['idx'] and \
          ".gov.au" not in data['idx'] and ".gov.uk" not in data['idx']:
          if len(text) > 1000: text = text[:1000]
          # for gutenberg we will fix this with upsampling and debiasing
          flagged_score, banned_score, hate_score = get_flaggedword_score(text,lang)
          if (flagged_score > 0.2 and "wikipedia.org/" not in data['idx']) or flagged_score > 0.25:
              return True
          if flagged_score > 0.05 and banned_score > 0.05:
              return True
          if hate_score > 0.1:
              return True
          if flagged_score > 0.05 and hate_score > 0.05:
              return True
          if not data['text']: return True
          # some general spam terms
          spam_terms = sum ([1 for  a in ["Free," "Cash," "Money," "Win," "Prize," "Bonus," "Earn extra income",
                                          "Limited time," "Discount," "Offer," "Buy now," "Special promotion," "Deal",
                                          "Act now," "Urgent," "Don't delete," "Immediate response", "Cheap", "Low cost", 
                                          "Risk-free," "No obligation," "Guarantee," "100% satisfied",
                                          "Miracle," "Cure," "Lose weight fast," "without prescription",
                                          "without a prescription", "No prescription needed"] if " "+a+" " in data['text'] or " "+a.lower()+" " in data['text'] or \
                             a+" " in data['text'] or a.lower()+" " in data['text']])
          
          if spam_terms > 3 or \
             (spam_terms > 0  and \
              any(a in data['text'] or a.lower() in data['text'] for a in ['weight loss', ' dating app', 'dating site', 'diet pill', 'erectile dys', 'Viagra', 'Cialis'])):
              return True

      
      text=data['text']
      if "Copyright 19" in text or "Copyright 20" in text or "Copyright ©" in text: return True
      if "Copyright: Zhang" in text or "Content owned & provided" in text or "Copyright American Chemical Society" in text or "All Rights Reserved" in text or "protected by Copyright" in text or "© Copyright"in text or "Copyright©" in text or "© Copyright" in text or "Copyrights and Proprietary Information" in text or "Copyright by" in text or "Copyright: Federal" in text or "Copyright 2" in text or "Copyright 19" in text or "Copyright (c)"in text or "Copyright ©" in text or "contained herein is strictly prohibited" in text or "commercial use must be authorized" in text or "This copyrighted, evidence-based medicine" in text or "All rights reserved"in text or "all rights reserved" in text:
          return True

      if "Creative Commons Attribution-NonCommercial" in text or ("Creative Commons" in text and "NonCommercial" in text):
          return True
      
      for t2, metadata in  zip(data['text'].split("<|endoftext|>"), data.get('metadata', {})):
          
          if 'license_header_footer' not in metadata:
              head_tail = t2[:100].lower() + t2[-100:].lower()
          else:
              if not (type(metadata['license_header_footer']) is str):
                  metadata['license_header_footer'] = str(metadata['license_header_footer'])
              head_tail = t2[:100].lower() + t2[-100:].lower() + metadata['license_header_footer'].lower()
          if "(by-nc)" in head_tail or "cc-by-nc" in head_tail or "by-nc-sa" in head_tail or "cc-by-nc-sa" in head_tail or "by-nc-sa" in t2 or "by-sa-nc" in head_tail or \
             "cc-by-nd" in head_tail or "by-sa-nd" in head_tail or "by-nc-nd" in head_tail or "by-nd-sa" in t2 or "by-sa-nd" in head_tail or \
             "cc by nc" in head_tail or "by nc sa" in head_tail or "cc by nc sa" in head_tail or "by nc sa" in t2 or "by sa nc" in head_tail or \
             "cc by nd" in head_tail or "by sa nd" in head_tail or "by nc nd" in head_tail or "by nd sa" in t2 or "by sa nd" in head_tail:
              return True

          if ('music' in head_tail or 'photo' in head_tail or 'flickr' in head_tail or 'picture' in head_tail or 'image' in head_tail) and ("cc-by " in head_tail or "cc-0 " in head_tail or "cc-by-" in head_tail or  "creative common" in head_tail):
              return True
          if ("cc-by " in head_tail or "cc-0 " in head_tail or "cc-by-" in head_tail or  "creative common" in head_tail) and ('noncommercial' in head_tail or 'non-commercial' in head_tail or 'non commercial' in head_tail):
              return True
          if 'rights reserved' in head_tail or  "copying prohibit" in head_tail or  "copyright" in head_tail:
              return True
          if "content owned & provided" in head_tail or "copyright american chemical society" in head_tail or "all rights reserved" in head_tail or "protected by copyright" in head_tail or "© copyright"in head_tail or "copyright©" in head_tail or "© copyright" in head_tail or "copyrights and proprietary information" in head_tail or "copyright by" in head_tail or "copyright: zhang"in head_tail or "copyright: federal" in head_tail or "copyright 2" in head_tail or "copyright 19" in head_tail or "copyright (c)"in head_tail or "copyright ©" in head_tail or "contained herein is strictly prohibited" in head_tail or "for commercial use must be authorized" in head_tail or "this copyrighted, evidence-based medicine" in head_tail:
                return True
          if '©' in head_tail:
                return True
      
          found = False
          for s in non_derivative:
              if s in head_tail:
                  found = True
                  break
          if not found:
              for s in non_commercial:
                  if s in head_tail:
                      found = True
                      break
          if found:
              return True
    
      return False

non_commercial = ['nonkomersial', 'necomercial', 'icke kommersiell', '비영리', 'ikke kommerciel', 'noncommercial', 'non-commercial', 'niet-commercieel', 'nekomerciāls', 'mittetulunduslik', 'noncomercial', 'nekomercinis', 'nichtgewerblich', '비상업적', 'некоммерческое', 'nicht-kommerziell', 'nach tráchtála', 'ikke kommersiell', 'שאינו מסחרי', 'ei kaupallista käyttöä', 'niekomercyjny', 'गैर-व्यावसायिक', 'ei kaupallista', 'non commerciale', '非商业性', 'ticari olmayan', 'nekomercialno', 'غير تجاري', 'μη εμπορική', '非商业', 'nem kereskedelmi', 'mhux kummerċjali', 'nekomerčné', 'não comercial', '営利目的外', 'non commercial', 'no comercial', 'nekomerční', '非営利', 'nicht kommerziell']

non_derivative = [
    'noderivs',
    'noderivatives',     
    'non derivative', 
    'no derivatives', 
    'no derivative', 
    'ingen bearbejdelse',       # Danish
    'ei muutoksia',             # Finnish
    'keine Bearbeitung',        # German
    'ohne Bearbeitung',         # German alternative
    'bez przeróbek',            # Polish
    'pas de modification',      # French
    'sans modification',        # French alternative
    'senza modifiche',          # Italian
    'ingen endringer',          # Norwegian
    'inga bearbetningar',       # Swedish
    'geen afgeleide werken',    # Dutch
    'χωρίς παράγωγα έργα',      # Greek
    'sin obras derivadas',      # Spanish
    'sem obras derivadas',      # Portuguese
    'bez izvedenih del',        # Slovenian
    'bez izvedenica',           # Croatian, Serbian
    'bez odvozených děl',       # Czech
    'nedrīkst atvasināt',       # Latvian
    'järeltöötluseta',          # Estonian
    'nem származékos',          # Hungarian
    'nessuna opera derivata',   # Italian alternative
    'без производных',          # Russian
    '不可衍生',                 # Chinese simplified
    '禁止改作',                 # Chinese alternative
    '改変禁止',                 # Japanese
    '2차적 저작물 금지',          # Korean
    'ללא יצירות נגזרות',        # Hebrew
    'ingen afledte værker',     # Danish alternative
    'inte bearbetad',           # Swedish alternative
    'ei johdannaisia',          # Finnish alternative
    'ingen bearbeidelse',       # Norwegian alternative
    'بدون اشتقاق',              # Arabic
    'sin derivados',            # Spanish alternative short
    'без похідних творів',      # Ukrainian
    'nu lucrări derivate',      # Romanian
    'nessuna derivazione',      # Italian alternative short
]


regions = ['American', 'English', 'French', 'Indian', 'Spanish', 'Chinese', 'Afrikaans', 'Tosk Albanian', 'Amharic', 'Aragonese', 'Arabic', 'Egyptian Arabic', 'Asturian', 'Assamese', 'Avaric', 'South Azerbaijani', 'Azerbaijani', 'Bavarian', 'Bashkir', 'Central Bikol', 'Belarusian', 'Bulgarian', 'Bihari', 'Bengali', 'Tibetan', 'Bishnupriya', 'Breton', 'Bosnian', 'Russia Buriat', 'Catalan', 'Chavacano', 'Cebuano', 'Chechen', 'Central Kurdish', 'Czech', 'Chuvash', 'Welsh', 'Danish', 'German', 'Dimli', 'Lower Sorbian', 'Dhivehi', 'Greek', 'Modern Greek', 'Emilian-Romagnol', 'English', 'Esperanto',  'Estonian', 'Basque', 'Persian', 'Finnish', 'Northern Frisian',  'Western Frisian', 'Irish', 'Scottish Gaelic', 'Galician', 'Guarani', 'Goan Konkani', 'Gujarati', 'Hebrew', 'Hindi', 'Croatian', 'Upper Sorbian', 'Haitian', 'Hungarian', 'Armenian', 'Interlingua', 'Indonesian', 'Interlingue', 'Iloko', 'Ido', 'Icelandic', 'Italian', 'Japanese', 'Lojban', 'Javanese', 'Georgian', 'Kazakh', 'Central Khmer', 'Kannada', 'Korean', 'Karachay-Balkar', 'Kurdish', 'Komi', 'Cornish', 'Kirghiz', 'Latin', 'Luxembourgish', 'Lezghian', 'Limburgan', 'Lombard', 'Lao', 'Northern Luri', 'Lithuanian', 'Latvian', 'Maithili', 'Malagasy', 'Eastern Mari', 'Minangkabau', 'Macedonian', 'Malayalam', 'Mongolian', 'Western Mari', 'Marathi', 'Malay', 'Maltese', 'Mirandese', 'Burmese', 'Erzya', 'Mazanderani', 'Nahuatl languages', 'Neapolitan', 'Low German', 'Nepali', 'Newari', 'Dutch', 'Norwegian Nynorsk', 'Norwegian', 'Occitan', 'Oriya', 'Ossetian', 'Pampanga', 'Panjabi', 'Polish', 'Piemontese', 'Western Panjabi', 'Pushto', 'Portuguese', 'Quechua', 'Romansh', 'Romanian', 'Russian', 'Yakut', 'Sanskrit', 'Sicilian', 'Sindhi', 'Serbo-Croatian', 'Sinhala', 'Slovak', 'Slovenian', 'Somali', 'Albanian', 'Serbian', 'Sundanese', 'Swedish', 'Swahili', 'Tamil', 'Telugu', 'Tajik', 'Thai', 'Turkmen', 'Tagalog', 'Turkish', 'Tatar', 'Tuvinian', 'Uighur', 'Ukrainian', 'Urdu', 'Uzbek', 'Venetian', 'Vietnamese', 'Volapük', 'Waray', 'Walloon', 'Wu Chinese', 'Kalmyk', 'Mingrelian', 'Yiddish', 'Yoruba',]


def remove_junk_lines(text):
    text = text.split("\n")
    text_arr2 =[]
    seen = {}
    for text2 in text:
      if len(text2) == 1: continue
      if text2 and text2[0]==text2[0].upper():
          if " " in text2:
              a, b = text2.split(" ",1)
              try:
                  int(a)
                  text2 = b.strip()
              except:
                  pass
      if text2 and text2[0]==text2[0].upper():              
          if seen.get(text2,0) > 5: # remove page numbers and similar dups
              continue
          seen[text2] = seen.get(text2,0)+1
          score1 = get_special_char_score(text2, lang)
          if score1 > 0.15 and len(text2) > 100:
              #print (("dropping", text2))
              continue
          text_arr2.append("\n"+text2.rstrip())
      else:
          text_arr2.append(" "+text2.rstrip())
    text = "".join(text_arr2)
    text = text.replace(" v. .", ".").replace(".  ", ". ").replace(",  ", ", ").replace(",  ", ", ").replace(", ,", ",").replace(", ,", ",").replace(", ,", ",").replace(", .", ".").replace(",.", ".").replace(". . ", ". ")        
    text = "\n".join(t.rstrip() for t in text.split("\n"))
    return text.strip()

def cleanup_raw_text(text, lang, cleanup_sents=False):
    stopwords =  all_stopwords.get(lang, all_stopwords['en'])    
    text = text.strip()
    text_old = text
    text = html.unescape(text)
    text = remove_citations(text)
    text = remove_junk_lines(text)
    score1 = get_special_char_score(text, lang)
    if score1 > 0.15:
        return ""
    score2 = get_stopword_score(text, lang)
    if score2  < 0.05:
        return ""
    if cleanup_sents:
        sents = text.split(". ")
        sents2 = []
        for text2 in sents:
            if "-cv-" in text2: continue            
            text3 = []
            for w in text2.split(" "):
                if w and len(w) <= 2 and text3 and len(text3[-1]) <= 2 and w[0] in "qwertyuiopasdfghjklzxcvbnm~" and (len(w) <=1 or w not in stopwords): continue
                text3.append(w)
            text2 = " ".join(text3)
            if len(text2) >= 10:
                score1 = get_special_char_score(text2, lang)
                if score1 > 0.15:
                    text2 = ".."
            if text2.endswith(" v") or text2.endswith(" vs") or text2.endswith(" V") or text2.endswith(" Vs") or "U.S.C" in text2 or "2d at " in text2 or "L.Ed" in text2 or "S.Ct" in text2 or "th Cir" in text2 or "nd Cir" in text2:
                score2 = get_stopword_score(text, lang)
                if score2  < 0.05:
                    text2 = ".."                    
            sents2.append(text2)
        text = ". ".join(sents2)
        if not text: return ""    
        has_period = False
        if text[-1] == ".":
            has_period = True
        text = text.replace(" v. ...", " ...").replace("... ...", "...").replace("... ...", "...").replace("... ...", "...").replace("... ...", "...").strip(". ")
        if has_period:
            text= text+"."
        if " ... " in text:
            text_arr = text.split(" ... ")
            text_arr2= []
            for text2 in text_arr:
                if len(text2) < 10: continue
                score1 = get_special_char_score(text2, lang)
                if score1 > 0.15:
                    continue
                elif len(text2) > 50:
                    score2 = get_stopword_score(text2, lang)
                    if score2  < 0.05:
                        continue
                text_arr2.append(text2)
            text= " ... ".join(text_arr2).strip()
            has_period = False
            if not text: return ""
            if text[-1] == ".":
                has_period = True
            text = text.replace(" v. ...", " ...").replace("... ...", "...").replace("... ...", "...").replace("... ...", "...").replace("... ...", "...").strip(". ")
            if has_period:
                text= text+"."
        if text:
            text = remove_citations(text)
            text = remove_junk_lines(text)
    text = text.replace(" v. .", ".").replace(".  ", ". ").replace(",  ", ", ").replace(",  ", ", ").replace(", ,", ",").replace(", ,", ",").replace(", ,", ",").replace(", .", ".").replace(",.", ".").replace(". . ", ". ").strip()
    if text != text_old:
        #print((text,))
        pass
    return text

        

citation_patterns = [
    # Original patterns for author-based citations
    r'\s(?:[A-Z][a-z]+(?:\sJr\.)?,\s[A-Z][a-z]+(?:\s[A-Z]\.)?;\s)+[A-Z][a-z]+,\s[A-Z][a-z]+\s[A-Z]\.,?',
    r'\b(?:(v\.\s*\d+,\s*no\.\s*\d+)|(p{1,2}\.\s*\d+(?:-\d+)?))\.?',
    r'\s((?:[A-Z]\.\s)+[A-Z][a-z]{4,};\s*)+',
    r'(?i)(Subsec\.|§|Pub\.\s+L\.|Stat\.|ch\.|title)\s+[^\s]+\s*\d+[^\s]*',
    r'\b(?:Sec\.|§|R\.\s*S\.|Stat\.|L\.|ch\.|title)\s+[\w\-.,]+\d+\b',
    r'title\s+[IVXL]+\s*,\s*§',
    r'\*\s*([A-Za-z]+(\sv\.\s)?[A-Za-z\s.]+,)\*\s*\d+\s+[F]\.\s?\d*d?\s+\d*\s*\(\s*\d+(?:st|nd|rd|th)\s*Cir\.\s*\d{4}\)',
    r'\b\d+\s+(?:Mo\.|N\. Y\.|Ill\.|Pa\.|Fed\.|F\.\s?\d*d?|S\. W\.|N\. W\.|N\. E\.|C\. C\. A\.|Pac\.|Cal\.|Ohio St\.|Mass\.|Wis\.|Minn\.|Ky\.|La\.|Ind\.|Tex\.|Miss\.|Conn\.|Kan\.|N\. J\.|R\. I\.|Mont\.|Iowa|Hun|Barb\.|Macq\.)[\s.]*\d+',
    r'\b(?:Cir\.|App\.|Supp\.|Ct\.|Dist\.|Exch\.|Eq\.|Commw\.)\b',
    r'\b(?:affirmed|reversed|cert\.|denied|ex rel|supra)\b.*?\)',
    # U.S. patent numbers
    r'\bU\.?\s?S\.?\s*Pat\.?\s*Nos?\.?\s*\d{1,3}(?:,\d{3})*(?:\s*,\s*\d{1,3}(?:,\d{3})*)*\b',        
     # Book/publication references
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s+[A-Z][a-z]+,\s+\d{4}\.?\s+(?:pp?\.\s+)?\d+-\d+\b',
        
]    

import re

def remove_citations(text):
    cleaned_text = text
    for pattern in citation_patterns:
        cleaned_text = re.sub(pattern, ' ', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = cleaned_text.replace(" v. .", ".").replace(".  ", ". ").replace(",  ", ", ").replace(",  ", ", ").replace(", ,", ",").replace(", ,", ",").replace(", ,", ",").replace(", .", ".").replace(",.", ".").replace(". . ", ". ")        
    return cleaned_text.strip()

        
curated_stopwords = {"patent", "title", "page", "united", "states", "edgar", "court", "federal", "district", "register", "congress", "congressional", "record", "volume", "number", "judgement", "exhibit", "section", "science", "news", "u.s.", "case", "law"}

def read_file(file):    
    print (file)
    if True:#try:
        if "common-pile" in file:
            for l in open(file, "rb"):
                try:
                    data = json.loads(l)
                except:
                    continue
                text = data['text']
                del data['text']
                data['source'] = file
                if 'meeadata' in data:
                    data['metadata'] = data['meeadata']
                if 'metadata' in data and type(data['metadata']) is str:
                    data['metadata'] = json.loads(data['metadata'])
                if 'meta' in data and type(data['meta']) is str:
                    data['meta'] = json.loads(data['meta'])                
                
                metadata = data.get("metadata", data.get('meta', {}))
                if 'meta' in data and 'language' in data['meta']:
                    data['lang'] = data['meta']['language']
                elif 'metadata' in data and 'language' in data['metadata']:
                    data['lang'] = data['metadata']['language']
                elif 'meta' in data and 'lang' in data['meta']:
                    data['lang'] = data['meta']['lang']
                elif 'metadata' in data and 'lang' in data['metadata']:
                    data['lang'] = data['metadata']['lang']
                elif 'lang' not in data:
                    data['lang'] = "en"
                    
                if 'meta' in data and 'license' in data['meta']:
                    data['license_header_footer'] = data['meta']['license']
                elif 'metadata' in data and 'license' in data['metadata']:
                    data['license_header_footer'] = data['metadata']['license']
                elif 'meta' in data and 'oa_license' in data['meta']:
                    data['license_header_footer'] = data['meta']['oa_license']
                elif 'metadata' in data and 'oa_license' in data['metadata']:
                    data['license_header_footer'] = data['metadata']['oa_license']
                idx = ""
                if 'url' in data:
                    idx = data['url']
                elif 'metadata' in data and 'url' in data['metadata']:
                    idx = data['metadata']['url']
                elif 'meta' in data and 'url' in data['meta']:
                    idx = data['meta']['url']
                elif 'metadata' in data and 'text_file_url' in data['metadata']:
                    idx = data['metadata']['text_file_url']
                elif 'meta' in data and 'text_file_url' in data['meta']:
                    idx = data['meta']['text_file_url']
                elif 'meta' in data and 'ia_url' in data['meta']:
                    idx = data['meta']['ia_url']
                elif 'metadata' in data and 'ia_url' in data['metadata']:
                    idx = data['metadata']['oa_url']
                elif 'meta' in data and 'oa_url' in data['meta']:
                    idx = data['meta']['oa_url']
                elif 'metadata' in data and 'oa_url' in data['metadata']:
                    idx = data['metadata']['oa_url']
                elif 'wikipedia' in file and 'metadata' in data and 'title' in data['metadata']:
                    idx = "https://"+data['metadata']['title'].replace(" ", "_")+".wikipedia.org/wiki/"+data['lang']
                elif 'wikipedia' in file and 'meta' in data and 'title' in data['meta']:
                    idx = "https://"+data['meta']['title'].replace(" ", "_")+".wikipedia.org/wiki/"+data['lang']
                elif 'stackex' in file and 'metadata' in data and 'source' in data['metadata']:
                    idx = "https://"+data['metadata']['source']+".com"                    
                elif 'stackex' in file and 'meta' in data and 'source' in data['meta']:
                    idx = "https://"+data['meta']['source'] +".com"
                if not idx and 'meta' in data:
                    for key in data['meta']:
                        if "url" in key:
                            idx = data['meta'][key]
                            break
                elif not idx and 'metadata' in data:
                    for key in data['metadata']:
                        if "url" in key:
                            idx = data['metadata'][key]
                            break
                if not idx:
                    lang = data["lang"]
                    stopwords =  all_stopwords.get(lang, {})
                    en_stopwords =  all_stopwords.get("en", {})                
                    is_cjk = lang_is_cjk(lang)
                    if not is_cjk:
                        tail = text.lower().split()
                        if len(tail) > 20:
                            tail = tail[-20:]
                        tail = [a[:4] if len(a) > 4 else a for a in [a.strip(strip_chars) for a in tail if len(a) > 2] if len(a) > 2 and a not in stopwords and a not in en_stopwords and a not in curated_stopwords]
                    else:
                        tail = text.lower().split()
                        if len(tail) > 20:
                            tail = tail[-20:]
                        tail = "".join(tail)
                        tail = [a for a in [a.strip(strip_chars) for a in tail] if a not in stopwords]
                    tail = "".join(tail)
                    if len(tail) <=1: 
                        tail += str(random.randint(0,9))+str(random.randint(0,9))
                    idx= tail+"_"+data['source'].split("/")[-1].replace("www.", "")
                    for c in unix_chars:
                        idx = idx.replace(c, "")
                    # this is a hack to make the non url data at the end in a sorted list. this has the advantage of filtering out data that have no urls but might be duplicates.
                    idx = "z://"+idx
                # {'authors':
                idx = idx.replace("\\", "")
                if "://" not in idx:
                    idx = "https://"+idx
                data['idx'] = idx
                text = remove_citations(text)                            
                if 'loc.gov' in idx or "fandom" in idx or "wikia" in idx:
                    text = cleanup_raw_text(text, data['lang'])
                data['license_header_footer'] = data.get('license_header_footer', '')
                data['is_govt'] = False
                if "pubmed" in file or "uk_han" in file or "regula" in file or "usgpo" in file or "uspto" in file or "library_of_congress" in file or "caselaw" in file:
                #if "http" not in data['idx']:
                    data['is_govt'] = True
                data = {'idx': idx, 'text': text, 'media_list': [], 'metadata': [data]}
                if not data['metadata'][0]['is_govt'] and not data['metadata'][0]['license_header_footer']: 
                    data['metadata'][0]['is_govt'] = True
                yield data
        elif "curated" in file:
            #TODO: 
            for l in open(file, "rb"):
                try:
                    data = json.loads(l)
                except:
                    continue
                text = data['text']
                del data['text']
                data['source'] = file
                if 'metadata' in data and type(data['metadata']) is str:
                    data['metadata'] = json.loads(data['metadata'])
                if 'meta' in data and type(data['meta']) is str:
                    data['meta'] = json.loads(data['meta'])                
                
                metadata = data.get("metadata", data.get('meta', {}))
                # Toxicity info
                toxicity_info = metadata.get("toxicity", [])
                formatted_toxicity = "\n".join(
                    f"* {label}: {score:.3f}" for label, score in toxicity_info if score >= 0.2
                ) 
                if formatted_toxicity:
                    if random.randint(0,1):
                        text += "<|endofsection|>"+formatted_toxicity
                    else:
                        text = formatted_toxicity+"<|endofsection|>"+text                
                if "wikibooks" in file:
                    data["lang"] = file.split("/")[-1].replace(".jsonl", "").strip()
                elif 'meta' in data and 'language' in data['meta']:
                    data['lang'] = data['meta']['language']
                elif 'metadata' in data and 'language' in data['metadata']:
                    data['lang'] = data['metadata']['language']
                elif 'meta' in data and 'lang' in data['meta']:
                    data['lang'] = data['meta']['lang']
                elif 'metadata' in data and 'lang' in data['metadata']:
                    data['lang'] = data['metadata']['lang']
                elif 'lang' not in data:
                    data['lang'] = "en"
                if 'url' in data:
                    idx = data['url']
                elif 'wikipedia' in file and 'metadata' in data and 'title' in data['metadata']:
                    idx = "https://"+data['metadata']['title'].replace(" ", "_")+".wikipedia.org/wiki/"+data['lang']
                elif 'wikipedia' in file and 'meta' in data and 'title' in data['meta']:
                    idx = "https://"+data['meta']['title'].replace(" ", "_")+".wikipedia.org/wiki/"+data['lang']
                elif 'stackex' in file and 'metadata' in data and 'source' in data['metadata']:
                    idx = "https://"+data['metadata']['source']+".com"                    
                elif 'stackex' in file and 'meta' in data and 'source' in data['meta']:
                    idx = "https://"+data['meta']['source'] +".com"                   
                elif 'metadata' in data and 'url' in data['metadata']:
                    idx = data['metadata']['url']
                elif 'meta' in data and 'url' in data['meta']:
                    idx = data['meta']['url']
                else:
                    lang = data["lang"]
                    stopwords =  all_stopwords.get(lang, {})
                    en_stopwords =  all_stopwords.get("en", {})                
                    is_cjk = lang_is_cjk(lang)
                    if not is_cjk:
                        tail = text.lower().split()
                        if len(tail) > 20:
                            tail = tail[-20:]
                        tail = [a[:4] if len(a) > 4 else a for a in [a.strip(strip_chars) for a in tail if len(a) > 2] if len(a) > 2 and a not in stopwords and a not in en_stopwords and a not in curated_stopwords]
                    else:
                        tail = text.lower().split()
                        if len(tail) > 20:
                            tail = tail[-20:]
                        tail = "".join(tail)
                        tail = [a for a in [a.strip(strip_chars) for a in tail] if a not in stopwords]
                    tail = "".join(tail)
                    if len(tail) <=1: 
                        tail += str(random.randint(0,9))+str(random.randint(0,9))
                    idx= tail+"_"+data['source'].split("/")[-1].replace("www.", "")
                    for c in unix_chars:
                        idx = idx.replace(c, "")
                    # this is a hack to make the non url data at the end in a sorted list. this has the advantage of filtering out data that have no urls but might be duplicates.
                    idx = "z://"+idx
                    
                idx = idx.replace("\\", "")                    
                data['idx'] = idx
                text = remove_citations(text)                            
                if 'loc.gov' in idx or "fandom" in idx or "wikia" in idx:
                    text = cleanup_raw_text(text, data['lang'])
                data['license_header_footer'] = ""
                data['is_govt'] = False
                # THIS IS NOT RIGHT. TODO: fix per type
                if "http" not in data['idx']:
                    data['is_govt'] = True
                data = {'idx': idx, 'text': text, 'media_list': [], 'metadata': [data]}
                #print (data)
                yield data
        elif "FineFine" in file:
            for l in open(file, "rb"):
                try:
                    data = json.loads(l)
                except:
                    continue
                idx = data['url']
                text = data['text']
                del data['url']
                del data['text']
                data['source'] = file
                data['lang'] = 'en'
                data['idx'] = idx
                if 'loc.gov' in idx:
                    text = cleanup_raw_text(text, 'en')
                data = {'idx': idx, 'text': text, 'media_list': [], 'metadata': [data]}
                yield data
        elif "nemo" in file:
            for l in open(file, "rb"):
                try:
                    data = json.loads(l)
                except:
                    continue
                idx = data['url']
                text = data['text']
                del data['url']
                del data['text']
                data['idx'] = idx                
                data['source'] = file
                data['lang'] = 'en'
                yield  {'idx': idx, 'text': text, 'media_list': [], 'metadata': [data]}


        elif "MAGA" in file:
            df = pq.read_table(file).to_pylist()
            for data in df:
                text = data['content_split']
                del data['content_split']
                try:
                    idx = json.loads(data['meta']['meta_extra'])
                except:
                    continue
                idx = idx['url']
                data['source'] = file
                data['lang'] = 'en'
                data['idx'] = idx
                if data['meta']['raw_text'].strip() not in text:
                    text =  cleanup_raw_text(data['meta']['raw_text'], 'en') + "<|endoftext|>" + text
                    del data['meta']['raw_text']                                                
                    data =  {'idx': idx, 'text': text, 'media_list': [], 'metadata': [data, copy.copy(data)]}
                    yield data
                    continue
                del data['meta']['raw_text']                
                yield {'idx': idx, 'text': text, 'media_list': [], 'metadata': [data]}                

        elif "txt360" in file:
            for l in open(file, "rb"):
                try:
                    data = json.loads(l)
                except:
                    continue
                idx = data['meta']['url']
                text = data['text']
                del data['text']
                data['source'] = file
                data['lang'] = 'en'
                data['idx'] = idx
                if 'loc.gov' in idx:
                    text = cleanup_raw_text(text, 'en')
                data = {'idx': idx, 'text': text, 'media_list': [], 'metadata': [data]}                                
                yield data
        else:
            print ("unknown file", file)
    #except:
    #    print ("error in file", file)        
    #    return


def process (arg):
    global model, tokenizer, device, args, num_devices
    global white_list_sites
    file, device_no, args = arg
    if white_list_sites is None:
        white_list_sites = set(json.load(open("white_list_urls.json")))
    if args.add_related:
        init_cosmo()
        init_seed2()        
    model, tokenizer = init_model(device_no, args)
    ret = []
    upsample_batch = []
    if os.path.exists(file.replace(args.input_dir, args.output_dir+"/done/")): return file, ret
    for data in read_file(file):
        for metadata in data['metadata']:
            lang = metadata['lang']
            if type(lang) is list:
                metadata['langs'] = lang
                lang = metadata['lang'] = lang[0]
        
        lang = data['metadata'][0]['lang']
        if 'lang' not in data:
            data['lang'] = lang
        if "kl3m" in data['metadata'][0]['source'] or "curated" in data['metadata'][0]['source'] or "common-pile" in data['metadata'][0]['source']:
            is_match = True            
            is_oss = False
        else:
            is_match, is_oss = is_idx_match(data)
        if not is_match: continue
        if args.add_related and ("kl3m" not in data['metadata'][0]['source'] and "mint" not in data['metadata'][0]['source']):        
            data = add_related(data)
        if lang != 'hi':
            text = data['text']                
            if "|" in text[:100]:
                text  = text[:100].split("|")[-1]+text[100:]
                if "|" in text[-100:]:
                    text  = text[:-100]  + text[-100:].split("|")[0]
                data['text'] = text
        orig = data
        data = dedup(data)
        if not data:
            continue
        orig= None
        #print(data)
        if True:            
            text_arr = []
            meta_arr = []
            for text, metadata  in zip(data['text'].split("<|endoftext|>"), data['metadata']):
                head = text[:100].lower()
                tail = text[-100:].lower()
                    
                if "cc-by " in head or "cc-by " in tail or "cc-0 " in head or "cc-0 " in tail or "cc-by-" in head or "cc-by-" in tail or \
                   "creative common" in head or "creative common" in tail or "public domain" in head or "public domain" in tail:
                    metadata['license_header_footer'] = head + " ... " + tail
                    metadata['is_govt'] = False
                elif 'is_govt' in metadata:
                    pass
                elif 'kl3m' in metadata['source']:
                    metadata['license_header_footer'] = ""
                    metadata['is_govt'] = True
                else:
                    metadata['license_header_footer'] = ""
                    idx = metadata['idx']
                    if ".mil/" in idx or ".vlada.mk" in idx or ".vlada.cz" in idx or ".kormany.hu" in idx or  "regeringen." in idx or ".rijksoverheid.\
                    nl" in idx or ".government.nl" in idx or ".regeringen.se" in idx or  ".regeringen.dk" in idx or  ".regeringen.no" in idx or ".bund.de" in idx or ".bundesregierung.de" in idx or  ".government.ru" in idx or ".gc.ca" in idx or \
                    ".admin.ch" in idx or  'www.gob.cl/' in idx or  'www.gob.ec/' in idx or  'guatemala.gob.gt/' in idx or  'presidencia.gob.hn/' in idx or  'www.gob.mx/' in idx or  'presidencia.gob.pa/' in idx or  'www.gob.pe/' in idx or  'gob.es/' in idx or  'argentina.gob.ar/' in idx or \
                    "tanzania.go.tz/" in idx or ".indonesia.go.id/" in idx or ".go.kr/" in idx or ".go.jp/" in idx or  "thailand.go.th/" in idx or ".europa.eu/" in idx or ".un/" in idx or ".int/" in idx or ".govt." in idx or "www.gub.uy" in idx or ".gov/" in idx or '.gov.' in idx or '.gouv.' in idx:
                        metadata['is_govt'] = True
                    else:
                        metadata['is_govt'] = False

                # do some misc text cleanup/fixing
                text = " ".join(["[URL]" if ("http:" in w or "https:" in w or "www." in w) else ("[BASE64_CODE]" if "base64" in w and len(w) > 10 else w) for w in text.split(" ")])
                if "wiki" in data['idx']:
                    text = re.sub(r'(\[\d+\])+', '', text)
                
                text = text.replace(".[", ". [").replace("…", "...").\
                    replace("Creative Commons Attribution-ShareAlike 3.0 Unported License (CC BY-SA)", "a permissive license").\
                    replace("Creative Commons Attribution 4.0 International", "a permissive license").\
                    replace("…", "...").replace("Creative Commons Attribution", "a permissive license").\
                    replace("Creative Commons Attribution-ShareAlike", "a permissive license").\
                    replace("Creative Commons", "a permissive license").\
                    replace(" the a ", " a ").replace(" a a ", " a ").replace("license License", "license").replace("license license", "license").replace("[edit]", "")
                text = remove_citations(text)                            
                text_arr.append(text)
                meta_arr.append(metadata)
            if not text_arr: continue
            text  = "<|endoftext|>".join(text_arr)
            text = html.unescape(text)            
            if len(text) < 100: continue
            data['text'] = text
            data['metadata'] = meta_arr
            ret.append(data)
            # upsample_batch things we know we won't strip b/c of permissions
            if any(meta for meta in data['metadata'] if meta['idx'] in white_list_sites or meta['is_govt'] or 'curated' in meta['source'] or 'common-pile' in meta['source']):
                upsample_batch.append(data)
                if len(upsample_batch) > 400:
                    generate_upsample(upsample_batch)
                    upsample_batch = []
    generate_upsample(upsample_batch)
    return (file, ret)
    
def parse_args():
    global args
    parser = argparse.ArgumentParser(description="Parse rank and world size.")
    parser.add_argument("--target_dir", type=str, default="./datasets/working/mixture_vitae_", help="The target dataset.")
    parser.add_argument("--rank", type=int, default=0, help="Rank of the process (default: 0)")    
    parser.add_argument("--sample", type=int, default=0, help="Only sample number of files for testing (default: 0)")
    parser.add_argument("--subset", type=str, default="", help="subset of the data")        
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes (default: 1)")
    parser.add_argument("--cache_dir",  type=str, default="./cache",  help="")

    args = parser.parse_args()
    return args


def get_rank() -> int:
    try:
        rank = int(os.environ['SLURM_PROCID'])
    except:
        rank = args.rank
    return rank


def _get_tasks_per_node() -> int:
    try:
        return int(os.environ['SLURM_NTASKS_PER_NODE'])
    except:
        return 1


def _get_num_nodes() -> int:
  try:
    return int(os.environ['SLURM_JOB_NUM_NODES'])
  except:
    return args.world_size


def get_world_size() -> int:
    return _get_num_nodes() * _get_tasks_per_node()

def wait_for_other_ranks(output_dir):
      ws = get_world_size()
      with open(output_dir+"/"+str(get_rank())+".rank_done", "w") as outf: pass
      num_done = len(glob.glob(output_dir+"/*.rank_done"))
      while num_done < ws:
          time.sleep(30)
          with open(output_dir+"/"+str(get_rank())+".rank_done", "w") as outf: pass
          num_done = len(glob.glob(output_dir+"/*.rank_done"))
    
        

if __name__ == "__main__":
    args = parse_args()
    args.output_dir = args.target_dir+"1/"
    args.input_dir =  "./datasets/working/"
    subset= args.subset
    if subset:
        args.output_dir = args.output_dir.rstrip("/")+"_"+subset+"/"
    args.all_files =[]
    print (args)
    os.system(f"mkdir -p {args.output_dir}")
    os.system("rm -rf "+args.output_dir+"/"+str(get_rank())+".rank_done")
    if  True: #not os.path.exists(args.output_dir+"/"+str(get_rank())+".rank_done"):
        # Iterate through all files in the directory and subdirectories
        root_dir = './datasets/working/mixture_vitae_curated/'
        args.all_files.extend(list(set(list(glob.glob(root_dir + '*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*.jsonl', recursive=True)) +  list(glob.glob(root_dir + '*/*/*.jsonl', recursive=True)))))

    args.all_files = [file for file in args.all_files if not os.path.exists(file.replace(args.input_dir, args.output_dir+"/done/"))]
    subset = subset.split(",")
    if subset:
        all_files2 = []
        for s in subset:
            all_files2.extend([f for f in args.all_files if args.input_dir+s in f])
        args.all_files = list(set(all_files2))
    if args.sample:
        random.shuffle(args.all_files)
        args.all_files  = args.all_files[:args.sample]
    args.all_files.sort()
    ws = get_world_size()
    rank = args.rank = get_rank()
    print ("starting rank", rank)
    rank2files = {}
    j = -1
    for file in args.all_files:
        j += 1
        for k in range(ws):
            if j == k:
                p = rank2files[k] = rank2files.get(k,[])
                p.append(file)
                if j == ws-1:
                    j = -1
                break
    os.system(f"mkdir -p {args.output_dir}/rank_{rank}")
    if rank in rank2files:
        files = rank2files[rank]
        random.shuffle(files)
        files = [(file, i%num_devices, args) for i, file in enumerate(files)]
        idx2outf = {}
        #if args.add_related:
        #    init_cosmo()
        #    init_seed2()        
        #with multiprocessing.Pool(4 if num_devices <= 1 else num_devices) as pool:    
        #    for file, ret in pool.imap_unordered(process, files):
        for file in files:
                file, ret = process(file)
                for data in ret:
                    if len(idx2outf) > 100:
                        for key in list(idx2outf.keys()):
                            if random.randint(0,1):
                                idx2outf[key].close()
                                del idx2outf[key]
                    #data = dedup(data)
                    #if not data:
                    #    continue
                    fix_idx(data)
                    idx = data['idx']
                    idx = idx.split("://",1)[-1]
                    for c in unix_chars:
                        idx = idx.replace(c, "")
                    if len(idx) > 3:
                        idx= idx[:3]
                    if len(idx) <= 1:
                        idx = idx+str(random.randint(0,100))
                    if len(idx) > 3:
                        idx= idx[:3]
                    hash_32 = mmh3.hash(idx, seed=42)
                    idx = str(hash_32).strip("-")[:3]
                    if idx == "207": # this is a hack to fix a previous bug. 
                        idx = data['idx']
                        idx = idx.split("://",1)[-1]                    
                        for c in unix_chars:
                            idx = idx.replace(c, "")
                        if len(idx) > 4:
                            idx= idx[:4]
                        if len(idx) <= 1:
                            idx = idx+str(random.randint(0,100))
                        if len(idx) > 4:
                            idx= idx[:4]
                        hash_32 = mmh3.hash(idx, seed=42)
                        idx = str(hash_32).strip("-")[:3]
                    if idx not in idx2outf:
                        idx2outf[idx] = open(f"{args.output_dir}/rank_{rank}/"+idx+".jsonl", "a+")
                    outf = idx2outf[idx]
                    outf.write(json.dumps(data)+"\n")
                    if random.randint(0,1000)==0:
                        outf.close()
                        idx2outf[idx] = open(f"{args.output_dir}/rank_{rank}/"+idx+".jsonl", "a+")
                os.system("mkdir -p "+ "/".join(file.replace(args.input_dir, args.output_dir+"/done/").split("/")[:-1]))
                with open(file.replace(args.input_dir, args.output_dir+"/done/"), "w") as outf: pass
            
    wait_for_other_ranks(args.output_dir)
