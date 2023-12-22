import os
import torch
import readline
from gllava.utils_conv.conv_template import get_conv_template
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("/home/pirenjie/pretrained_weights/vicuna-7b")
model = LlamaForCausalLM.from_pretrained("/home/pirenjie/pretrained_weights/vicuna-7b")
model.eval()
readline.parse_and_bind("")
while True:
    try:
        sentence = input("Enter sentence (end for finishing): ")
        input_ids = tokenizer(sentence, return_tensors="pt").input_ids
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)
        print(f"perplexity for '{sentence}' is: {perplexity}")
    except:
        sentence = input("q for exit program: ")
        if sentence == "q":
            break