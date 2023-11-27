import os

def process_annotation_file(lines, task_opt = "all"):
    entity_dict = {}
    for line in lines:
        items = line.strip('\n').split('\t')

        if task_opt == "all":
            # 如果是 all 代表不管不是時間類型的資料我都要拿來訓練
            if len(items) == 5:
                item_dict = {
                    'phi' : items [1],
                    'st_idx' :int (items[2]),
                    'ed_idx' : int(items[3]),
                    'entity' : items [4],
                }
            elif len(items) == 6:
                item_dict = {
                'phi' : items[1],
                'st_idx' : int(items[2]),
                'ed_idx' :int(items[3]),
                'entity': items [4],
                'normalize_time' : items[5],}
        elif task_opt == "task1":
            # 如果是 task1 那我只想訓練能解決 task1 的 annotation dict
            item_dict = {
                'phi' : items [1],
                'st_idx' :int (items[2]),
                'ed_idx' : int(items[3]),
                'entity' : items[4],
            }
        elif task_opt == "task2" and len(items) == 6:
            # 這個就是只想解決 task2 的 annotation dict
            item_dict = {
                'phi' : items[1],
                'st_idx' : int(items[2]),
                'ed_idx' :int(items[3]),
                'entity': items [4],
                'normalize_time' : items[5],}
        else:
            # pass 掉這不合規的資料，包括如果針對 task2 卻不是時間資訊的
            continue
        if items[0] not in entity_dict:
            entity_dict[items[0]] = [item_dict]
        else:
            entity_dict[items[0]].append(item_dict)
    return entity_dict


def generate_annotated_medical_report(anno_file_path, task_opt):
    '''
    有可能是關於生成 annotation data 的前置作業
    task_opt: "task1" | "all" 如果放入 "task1" 只會生出 task1 的訓練資料 不會有時間正規化
    '''
    with open(anno_file_path, "r", encoding='utf-8-sig') as f: # 加了後面的 encoding 去除掉文件 ufe 開頭的問題
        anno_lines = f.readlines()
        annos_dict = process_annotation_file(anno_lines, task_opt)
        return annos_dict

def process_medical_report(txt_name, medical_report_folder, annos_dict, special_tokens_dict):
    '''
    生成 training data
    '''
    file_name = txt_name + '.txt'
    sents = open(os.path.join(medical_report_folder, file_name), "r").readlines()
    article = "".join(sents)
    bounary, item_idx, temp_seq, seq_pairs = 0, 0, "", []
    for w_idx, word in enumerate(article):
        if item_idx < len(annos_dict[txt_name]) and w_idx == annos_dict[txt_name][item_idx]["st_idx"]:
            phi_key = annos_dict[txt_name][item_idx]['phi']
            phi_value = annos_dict[txt_name][item_idx]['entity']
            if "normalize_time" in annos_dict[txt_name][item_idx]:
                temp_seq += f"{phi_key}: {phi_value}=>{annos_dict[txt_name][item_idx]['normalize_time']}\n"
            else:
                temp_seq += f"{phi_key}:{phi_value}\n"
                if item_idx == len(annos_dict[txt_name]) - 1:
                    continue
            item_idx += 1
        if word == "\n":
            new_line_idx = w_idx + 1
            if temp_seq == "":
                temp_seq = "PHI:Null"
            seq_pair = special_tokens_dict['bos_token'] + article[bounary: new_line_idx] + special_tokens_dict['sep_token'] + temp_seq + special_tokens_dict['eos_token']
            bounary = new_line_idx
            seq_pairs.append(seq_pair)
            temp_seq = ""
    return seq_pairs

# testing
if __name__ == "__main__":
    task_opt = "all"
    special_tokens_dict = {"bos_token": "<|endoftext|>", "sep_token": "####", "eos_token": "<|END|>"}  
    annos_dict = generate_annotated_medical_report("First_Phase_Release(Correction)/answer.txt", task_opt)
    seq_pairs = []
    file_names = os.listdir("First_Phase_Release(Correction)/First_Phase_Text_Dataset")
    for file_name in file_names:
        file_name = file_name.replace(".txt", "")
        seq_pairs.extend(process_medical_report(file_name, "First_Phase_Release(Correction)/First_Phase_Text_Dataset", annos_dict, special_tokens_dict))
    # print(seq_pairs)