import torch
from sentence_splitter import SentenceSplitter, split_text_into_sentences
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import time
import random
import re
# import string

# start = time.time()
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text,num_return_sequences,num_beams):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text


num_beams = 10
num_return_sequences = 1
# context = "The ultimate test of your knowledge is your capacity to convey it to another."
# i = get_response(context,num_return_sequences,num_beams)
# print(i)

splitter = SentenceSplitter(language='en', non_breaking_prefix_file='./sentence-splitter/sentence_splitter/non_breaking_prefixes/en.txt')
# print(splitter.split(text='This is a paragraph. It contains several sentences. "But why," you ask?'))
f = open('./TXTs/definir-tech-items-descr-EN-PARAPHRASED.txt', 'a')
print('file read');

with open("./TXTs/definir-tech-items-descr-EN.txt") as file_in:
   for line in file_in:
      start = time.time()
      if re.match(r"^[0-9]+; ", line):
        line_parts = line.split(";")
        ID = line_parts[0]
        phrase = line_parts[1]
        sentences = splitter.split(phrase)
        res = ID+";"
        s_list = []
        for s in sentences:
          s = s.replace('"', '') 
          temp = get_response(s,num_return_sequences,num_beams);
          # res = res+" "+temp[0]
          s_list.append(temp[0])
          # print(".\n")
        # print(s_list)
        # random.shuffle(s_list)
        res = res + " ".join(s_list)
        f.write(res + "\n")
        f.flush()
        print(".", end = '')
      else:
        print("r", end = '')

      # print(res)
      # print(time.time() - start)

f.close()


# phrases = [

# "Grocery stores located in the Netherlands just on the cashier border can speak French a little",
# 'Good stuff", quality", wide range and easy to reach by car !!!',
# 'More and more organic options", helpful staff", well-kept shop.',
# 'Beautiful", neat and nice and cool shop. Dirty shopping baskets.',
# 'Grocery stores located in the Netherlands just on the cashier border can speak French a little'

# ]

# for phrase in phrases:
#   print("-"*100)
#   print("Input_phrase: ", phrase)
#   print("-"*100)
#   para_phrases = get_response(phrase,num_return_sequences,num_beams)
#   for para_phrase in para_phrases:
#    print(para_phrase)