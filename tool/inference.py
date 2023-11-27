import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AdamW
from data_set import GPTDataset
import data_preprocess
from tqdm import trange
from tqdm import tqdm
import os


py = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_text(model, tokenizer, text, n_words=100):
    model.eval()
    text = tokenizer.encode(text)
    inputs, past_key_values = torch.tensor([text]).to(device), None
    generated_text = []

    with torch.no_grad():
        for _ in range(n_words):
            output = model(inputs, past_key_values=past_key_values)
            logits = output.logits
            past_key_values = output.past_key_values
            log_probs = F.softmax(logits[:, -1], dim=-1)
            inputs = torch.multinomial(log_probs, 1)
            generated_text.append(inputs.item())

            if tokenizer.decode(inputs.item()) == "<|END|>":  # 定义 eos 作为终止标记
                break

    return tokenizer.decode(generated_text)


plm = "EleutherAI/pythia-70m"

tokenizer = AutoTokenizer.from_pretrained(plm)
special_tokens_dict = {"bos_token": "<|endoftext|>", "sep_token": "####", "eos_token": "<|END|>"}  
tokenizer.add_special_tokens(special_tokens_dict)
PAD_IDX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)


##load model
model = AutoModelForCausalLM.from_pretrained(plm)
model.resize_token_embeddings(len(tokenizer))
##remove map_location if not running on cpu
model.load_state_dict(torch.load('model.pth',map_location=torch.device('cpu')))

answer = []

with open('First_Phase_Release(Correction)/First_Phase_Text_Dataset/10.txt') as f:
    lines = f.readlines()
    for line in tqdm(lines):  ## 在這邊直接加入正則應該比較好
        if line == "":
            continue
        else:
            generated_text = sample_text(model, tokenizer, "<|endoftext|>"+line+"####", n_words=100)
            if(generated_text != "PHI:Null<|END|>"):
                answer.append(generated_text[:-8])
                # 在這個部分

with open('test_answer.txt', "w+") as a:
    for line in answer:
        a.write(line+"\n")



#generated_text = sample_text(model, tokenizer, "<|endoftext|>D.O.B:  24/8/1993####", n_words=100)
#print(generated_text)

