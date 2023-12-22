import os
import torch
import readline
from gllava.utils_conv.conv_template import get_conv_template
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from transformers import LlamaForCausalLM, LlamaTokenizer

# tokenizer = LlamaTokenizer.from_pretrained("/home/pirenjie/pretrained_weights/llama-7b")
tokenizer = LlamaTokenizer.from_pretrained("/home/pirenjie/pretrained_weights/vicuna-7b")
# model = LlamaForCausalLM.from_pretrained("/home/pirenjie/pretrained_weights/llama-7b")
model = LlamaForCausalLM.from_pretrained("/home/pirenjie/pretrained_weights/vicuna-7b")
model.eval()
readline.parse_and_bind("")
i = 0
while True:
    try:
        conv = get_conv_template("vicuna_v1.1")
        i =0
        while True:
            sentence = input("Enter sentence (end for finishing): ")
            if sentence == "end":
                all_sentence = conv.get_prompt()
                input_ids = tokenizer(all_sentence, return_tensors="pt").input_ids
                with torch.no_grad():
                    outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                perplexity = torch.exp(loss)
                print(f"perplexity for '{all_sentence}' is: {perplexity}")
                break
            conv.append_message(conv.roles[i%2], sentence)
            i+=1
    except:
        sentence = input("q for exit program: ")
        if sentence == "q":
            break