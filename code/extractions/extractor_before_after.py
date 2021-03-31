import spacy
import jsonlines
from nltk.stem.wordnet import WordNetLemmatizer


def get_tmparg(tags):
    ret = []
    for i, tag in enumerate(tags):
        if "B-ARGM-TMP" == tag:
            cur_group = [i, -1]
            for j in range(i + 1, len(tags)):
                if tags[j] != "I-ARGM-TMP":
                    cur_group[1] = j
                    break
            ret.append(cur_group)
    return ret


def build_verb_map(obj):
    ret_map = {}
    for verb in obj['verbs']:
        tags = verb['tags']
        relevant_indices = []
        reformat_arg_0 = []
        verb_idx = -1
        for i, t in enumerate(tags):
            if t != "O" and "ARGM-TMP" not in t and "ARG0" not in t:
                relevant_indices.append(i)
            if "ARG0" in t:
                if verb_idx != -1:
                    reformat_arg_0.append(i)
                else:
                    relevant_indices.append(i)
            if t == "B-V":
                verb_idx = i
        if len(reformat_arg_0) > 0:
            relative_verb_idx = -1
            for idx, val in enumerate(relevant_indices):
                if val == verb_idx:
                    relative_verb_idx = idx
                    break
            if relative_verb_idx > -1:
                relevant_indices = relevant_indices[0:relative_verb_idx] + reformat_arg_0 + relevant_indices[relative_verb_idx:]

        if verb_idx != -1:
            ret_map[verb_idx] = [relevant_indices, verb]
    return ret_map


def get_relevant_phrase(words, indices, doc, verb_idx):
    phrase = ""
    for i in indices:
        word = words[i]
        if i == verb_idx:
            passive = False
            for j in indices:
                if doc[j].head.i == i and doc[j].dep_ == "nsubjpass":
                    passive = True
                    break
            if not passive:
                word = WordNetLemmatizer().lemmatize(word, 'v')
        phrase += word + " "
    return phrase.strip()

def get_srl_file():
    raise NotImplementedError

'''
This function extracts before/after event pairs from AllenNLP's SRL model.
'''
def analyze_before_after(out_path):
    vv_count = 0
    vn_count = 0
    sent_count = 0
    nlp = spacy.load("en_core_web_sm", disable='ner')
    nlp.tokenizer = nlp.tokenizer.tokens_from_list
    skip_aux_list = [
        "this", "that"
    ]
    output_file = open(out_path, "w")
    progress = 0
    with jsonlines.open(get_srl_file()) as reader:
        for obj_list in reader:
            progress += 1
            if progress % 1000 == 0:
                print("Finished {} batches.".format(str(progress)))
            if progress == 5000:
                break
            for obj in obj_list:
                sent_count += 1
                words = obj['words']
                docs = nlp.pipe([words])
                doc = None
                for d in docs:
                    doc = d
                    break
                verb_map = build_verb_map(obj)
                for verb_idx in verb_map:
                    verb = verb_map[verb_idx][1]
                    tags = verb['tags']
                    tmpargs = get_tmparg(tags)
                    for start, end in tmpargs:
                        label = None
                        for i in range(start, end):
                            if words[i] in ["before", "after"]:
                                label = words[i]
                        if label is None:
                            continue

                        phrase_aux = ""
                        phrase_main = get_relevant_phrase(words, verb_map[verb_idx][0], doc, verb_idx)
                        abort = False
                        for i in range(start, end):
                            if i in verb_map:
                                if len(verb_map[i][0]) == 1:
                                    abort = True
                                phrase_aux = get_relevant_phrase(words, verb_map[i][0], doc, i)
                                break
                        if len(verb_map[verb_idx][0]) == 1:
                            abort = True
                        from_verb = True
                        if len(phrase_aux) == 0:
                            from_verb = False
                            aux_indices = []
                            found_noun = False
                            for i in range(start, end):
                                if words[i].lower() in ["before", "after"]:
                                    for j in range(i + 1, end):
                                        if doc[j].pos_ == "NOUN":
                                            found_noun = True
                                        aux_indices.append(j)
                            phrase_aux = get_relevant_phrase(words, aux_indices, doc, -1)
                            if len(phrase_aux.split()) == 0 or not found_noun:
                                abort = True

                        if phrase_aux in skip_aux_list:
                            continue
                        if abort:
                            continue
                        if from_verb:
                            vv_count += 1
                        else:
                            vn_count += 1
                        output_file.write(phrase_main + "\t" + phrase_aux + "\t" + label + "\n")