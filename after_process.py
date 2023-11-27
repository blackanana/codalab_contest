def remove_empty_data(file_name):
    result = []
    with open(file_name) as f:
        for line in f:
            line = line.strip()
            data = line.split("\t")
            if data[2] == data[3]:
                continue
            result.append(line)
    with open(file_name, "w") as f:
        f.write("\n".join(result))

def remove_none_exist_category(src, dst):
    lines = None
    with open(src) as f:
        lines = f.read().split("\n")
    with open(dst, "w") as f:
        categories = open("answer/phi_category.txt").read().split("\n")
        result = []
        for line in lines:
            category = line.split("\t")[1]
            if category in categories:
                result.append(line)
        f.write("\n".join(result))
        
if __name__ == "__main__":
    # remove_none_exist_category("task1_answer.txt", "answer/task1_answer.txt")
    remove_empty_data("answer/task1_answer.txt")