import json


'''
A helper function to generate SNLI formats
'''
def gen_snli_format():
    lines = [x.strip() for x in open("uniform-prior/tracie_train_uniform_prior.txt").readlines()]
    f_out = open("uniform-prior-snli/train.jsonl", "w")
    for line in lines:
        sent1 = line.split("\t")[0].split(" story: ")[0]
        sent2 = line.split("\t")[0].split(" story: ")[1]
        label = "entailment"
        if "negative" in line.split("\t")[1]:
            label = "contradiction"
        obj = {
            "query": sent1,
            "story": sent2,
            "gold_label": label
        }
        f_out.write(json.dumps(obj) + "\n")
