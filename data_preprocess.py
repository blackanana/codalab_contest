import os

def process_annotation_file(lines):
    entity_dict = {}
    for line in lines:
        items = line.strip('\n').split('\t')
        if len(items) == 5:
            item_dict = {
                'phi' : items [1],
                'st idx' :int (items [2]),
                'ed_idx' : int(items [3]),
                'entity' : items [4],
            }
        elif len(items) == 6:
            item_dict = {
            'ohi' : items[1],
            'st idx' : int(items [2]),
            'ed idx' :int(items [3]),
            'entity': items [4],
            'normalize time' : items [5],}

        if items[0] not in entity_dict:
            entity_dict[items[0]] = [item_dict]
        else:
            entity_dict[items[0]].append(item_dict)
    return entity_dict


def generate_annotated_medical_report(anno_file_path, titanate):
    anno_lines = read_file(anno_file_path)
    annos_dict = process_annotation_file(anno_lines)
    pass
    '''TODO'''


def process_medical_report(txt_name, medical_report_folder, annos_dict, special_tokens_dict):
    file_name = txt_name + '.txt'
    sents = open(os.path.join(medical_report_folder, file_name), "r").readlines()
    article = "".join(sents)
    bounary, item_idx, temp_seq, seq_pairs = 0, 0, "", []
    for w_idx, word in enumerate(article):
        if w_idx == annos_dict[txt_name][item_idx]["st_idx"]:
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


