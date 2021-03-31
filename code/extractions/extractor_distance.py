import json
import random
import copy
import re


def get_no_tmp_phrase(srl_obj):
    skip_list = []
    for verbs in srl_obj['verbs']:
        for i, t in enumerate(verbs['tags']):
            if "ARGM-TMP" in t:
                skip_list.append(i)
    ret = ""
    for i, w in enumerate(srl_obj['words']):
        if i not in skip_list:
            ret += w + " "
    return ret.strip()


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def get_relevant_phrase(words, tags):
    ret = ""
    for i, t in enumerate(tags):
        if t != "O" and "ARGM-TMP" not in t:
            ret += words[i] + " "
    return ret.strip()


def get_verb_idx(tags):
    for i, t in enumerate(tags):
        if t == "B-V":
            return i
    return None


months = [
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
]


def get_int_val(tok):
    year = None
    try:
        year = int(tok)
    except:
        pass
    return year


class TimeStruct:
    def __init__(self, minute, hour, day, month, year):
        self.minute = minute
        self.hour = hour
        self.day = day
        self.month = month
        self.year = year

    def __str__(self):
        return "{} {} {} {}:{}".format(str(self.year), str(self.month), str(self.day), str(self.hour), str(self.minute))


def extract_in(toks):
    year = None
    month = None
    if toks[0] != "in":
        return TimeStruct(None, None, None, month, year)
    for i, t in enumerate(toks):
        if t == "in" and i < len(toks) - 1:
            if toks[i+1] in months:
                month = toks[i+1]
                if i+2 < len(toks):
                    year = get_int_val(toks[i+2])
            else:
                year = get_int_val(toks[i+1])
            if year is not None or month is not None:
                break

    return TimeStruct(None, None, None, month, year)


def extract_on(toks):
    month = None
    year = None
    date = None
    if toks[0] != "on":
        return TimeStruct(None, None, None, None, None)
    for i, t in enumerate(toks):
        if t == "on":
            for j in range(i+1, min(i+5, len(toks))):
                if toks[j] in months:
                    month = toks[j]
                else:
                    cur_tok = toks[j]
                    if cur_tok.endswith("th") or cur_tok.endswith("rd") or cur_tok.endswith("st"):
                        cur_tok = cur_tok[:-2]
                    intval = get_int_val(cur_tok)
                    if intval is not None:
                        if 1000 < intval < 3000:
                            year = intval
                        elif 0 < intval < 32:
                            date = intval
    return TimeStruct(None, None, date, month, year)


def extract_at(toks):
    hour = None
    minute = None
    if toks[0] != "at":
        return TimeStruct(None, None, None, None, None)
    for i, t in enumerate(toks):
        if t == "at" and i < len(toks) - 1:
            cr_tok = toks[i+1]
            pm_override = False
            found_unit = False
            if cr_tok.endswith("pm"):
                cr_tok = cr_tok[:-2]
                pm_override = True
                found_unit = True
            if cr_tok.endswith("am"):
                cr_tok = cr_tok[:-2]
                found_unit = True
            if ":" in cr_tok:
                hour = get_int_val(cr_tok.split(":")[0])
                if hour is not None and hour > 24:
                    hour = None
                minute = get_int_val(cr_tok.split(":")[1])
                if minute is not None and minute > 59:
                    minute = None
            else:
                hour = get_int_val(cr_tok)
                if hour is not None and hour > 24:
                    hour = None
                for j in range(i+1, min(i+6, len(toks))):
                    if toks[j] in ["am", "a.m", "a.m." "pm", "p.m", "p.m.", "afternoon", "morning", "day"]:
                        found_unit = True
                if not found_unit:
                    hour = None
            for j in range(i+1, min(i+6, len(toks))):
                if toks[j] in ["p.m", "p.m.", "pm", "afternoon"] or pm_override:
                    if hour is not None and hour < 12:
                        hour += 12
                    if hour is not None and hour > 24:
                        hour = None
    return TimeStruct(minute, hour, None, None, None)


def combine_timex(l):
    ret = TimeStruct(None, None, None, None, None)
    for t in l:
        if t.minute is not None:
            ret.minute = t.minute
        if t.hour is not None:
            ret.hour = t.hour
        if t.day is not None:
            ret.day = t.day
        if t.month is not None:
            ret.month = t.month
        if t.year is not None:
            ret.year = t.year
    return ret


def get_useful_count(timex):
    ret = 0
    if timex.minute is not None:
        ret += 1
    if timex.hour is not None:
        ret += 1
    if timex.day is not None:
        ret += 1
    if timex.month is not None:
        ret += 1
    if timex.year is not None:
        ret += 1
    return ret


def default_timex(timex):
    ret_cpy = copy.deepcopy(timex)
    month_mapping = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    if timex.year is None:
        ret_cpy.year = 0
    if timex.month is None:
        ret_cpy.month = 1
    else:
        ret_cpy.month = month_mapping[timex.month]
    if timex.day is None:
        ret_cpy.day = 1
    if timex.hour is None:
        ret_cpy.hour = 0
    if timex.minute is None:
        ret_cpy.minute = 0
    return ret_cpy


def get_label(diff_in_hours):
    if 0 < diff_in_hours < 0.5:
        return "<extra_id_99>"
    if 0.5 <= diff_in_hours < 12.0:
        return "<extra_id_98>"
    if 12.0 <= diff_in_hours < 84.0:
        return "<extra_id_97>"
    if 84.0 <= diff_in_hours < 336.0:
        return "<extra_id_96>"
    if 336.0 <= diff_in_hours < 4320.0:
        return "<extra_id_95>"
    if 4320 <= diff_in_hours < 43800.0:
        return "<extra_id_94>"
    return "<extra_id_93>"


def calc_label(timex_1, timex_2):
    timex_1 = default_timex(timex_1)
    timex_2 = default_timex(timex_2)
    if timex_1.year == 0 and timex_2.year != 0:
        return None, None
    if timex_2.year == 0 and timex_1.year != 0:
        return None, None

    timex_1_val = timex_1.year * 8760.0 + (timex_1.month - 1) * 720.0 + (timex_1.day - 1) * 24.0 + timex_1.hour * 1.0 + (timex_1.minute / float(60.0))
    timex_2_val = timex_2.year * 8760.0 + (timex_2.month - 1) * 720.0 + (timex_2.day - 1) * 24.0 + timex_2.hour * 1.0 + (timex_2.minute / float(60.0))
    if timex_1_val == timex_2_val:
        return None, None
    if timex_1_val < timex_2_val:
        return "before", get_label(abs(timex_2_val - timex_1_val))
    else:
        return "after", get_label(abs(timex_2_val - timex_1_val))


def get_temporal_words(words, tags):
    ret = []
    for i, t in enumerate(tags):
        if t == "B-ARGM-TMP":
            end = i + 1
            for j in range(i+1, len(tags)):
                if tags[j] == "I-ARGM-TMP":
                    end = j + 1
                else:
                    break
            cur = []
            for k in range(i, end):
                cur.append(words[k].lower())
            ret.append(cur)
    return ret


def extract_timex(srl_objs):
    idx_accum = 0
    verb_phrase_to_tmp_map = {}
    paragraph = ""
    for srl_obj in srl_objs:
        for verb in srl_obj['verbs']:
            verb_phrase = get_relevant_phrase(srl_obj['words'], verb['tags'])
            tmp_phrases = get_temporal_words(srl_obj['words'], verb['tags'])
            all_extractions = []
            for p in tmp_phrases:
                t_1 = extract_on(p)
                t_2 = extract_in(p)
                t_3 = extract_at(p)
                all_extractions.append(t_1)
                all_extractions.append(t_2)
                all_extractions.append(t_3)
            timex = combine_timex(all_extractions)
            if get_verb_idx(verb['tags']) is None:
                continue
            map_idx = get_verb_idx(verb['tags']) + idx_accum
            verb_phrase_to_tmp_map[map_idx] = [verb_phrase, timex]
        idx_accum += len(srl_obj['words'])
        paragraph += " ".join(srl_obj['words']) + " "

    minute_record = None
    hour_record = None
    day_record = None
    month_record = None
    year_record = None
    final_list = []
    for i in range(0, idx_accum + 200):
        if i in verb_phrase_to_tmp_map:
            phrase, timex = verb_phrase_to_tmp_map[i]
            if get_useful_count(timex) == 0:
                continue
            if timex.year is not None:
                year_record = timex.year
                month_record = None
                day_record = None
            else:
                timex.year = year_record
            if timex.month is not None:
                month_record = timex.month
                day_record = None
            else:
                timex.month = month_record
            if timex.day is not None:
                day_record = timex.day
            else:
                timex.day = day_record
            if get_useful_count(timex) != 0 and len(phrase.split()) > 3:
                final_list.append([phrase, timex])

    counter_map = {
    }
    all_sentences = []
    all_length = 0
    for srl_obj in srl_objs:
        all_sentences.append(get_no_tmp_phrase(srl_obj))
        all_length += len(get_no_tmp_phrase(srl_obj).split())
    to_remove_sent = set()
    all_sent_ids = list(range(0, len(srl_objs)))
    while all_length > 100:
        selected = random.choice(all_sent_ids)
        if selected not in to_remove_sent:
            to_remove_sent.add(selected)
            all_length -= len(all_sentences[selected].split())

    selected_sentences = []
    for i, sent in enumerate(all_sentences):
        if i not in to_remove_sent:
            selected_sentences.append(sent)
    if random.random() < 0.85:
        random.shuffle(selected_sentences)
    concat = " ".join(selected_sentences)
    ret = []
    for i, (phrase_1, timex_1) in enumerate(final_list):
        for j in range(i+1, len(final_list)):
            phrase_2, timex_2 = final_list[j]
            tmp_label, dist_label = calc_label(timex_1, timex_2)
            if tmp_label is None:
                continue
            ret.append([concat, phrase_1, phrase_2, tmp_label, dist_label])
            if dist_label not in counter_map:
                counter_map[dist_label] = 0
            counter_map[dist_label] += 1
    return ret


def flip_label(l):
    if l == "before":
        return "after"
    return "before"


def get_srl_file():
    raise NotImplementedError


def get_story_file():
    raise NotImplementedError


'''
Function that extracts event pairs with distances between start times.
'''
def format_train_t5_paragraph_with_distance(out_path):
    all_stories = [x.strip() for x in open(get_story_file()).readlines()]
    all_srl = [x.strip() for x in open(get_srl_file()).readlines()]
    srl_map = {}
    f_out = open(out_path, "w")
    for srl in all_srl:
        obj = json.loads(srl)
        for sent in obj:
            key = "".join(sent['words']).lower().replace(" ", "")
            srl_map[key] = sent
    all_results = []
    for story in all_stories:
        sentences = story.split("\t")
        if sentences[0].startswith("-----------"):
            continue
        srl_objs = []
        for m, sentence in enumerate(sentences):
            sent_key = sentence.replace(" ", "").lower()
            if sent_key not in srl_map:
                continue
            srl_objs.append(srl_map[sent_key])
        all_results += extract_timex(srl_objs)
    key_limit = {}
    random.shuffle(all_results)
    for story, phrase_1, phrase_2, tmp_label, dist_label in all_results:
        if dist_label not in key_limit:
            key_limit[dist_label] = 0
        key_limit[dist_label] += 1
        if key_limit[dist_label] > 200000:
            continue
        right = "story: {}".format(story)
        if random.random() < 0.5:
            phrase_first = phrase_2
            phrase_second = phrase_1
            gold_label = flip_label(tmp_label)
            if random.random() < 0.5:
                display_label = gold_label
                answer_label = "positive"
            else:
                display_label = flip_label(gold_label)
                answer_label = "negative"
            left = "event: {} starts {} {}".format(phrase_first, display_label, phrase_second)
            answer = "answer: {} {}".format(answer_label, dist_label)
        else:
            phrase_first = phrase_1
            phrase_second = phrase_2
            gold_label = tmp_label
            if random.random() < 0.5:
                display_label = gold_label
                answer_label = "positive"
            else:
                display_label = flip_label(gold_label)
                answer_label = "negative"
            left = "event: {} starts {} {}".format(phrase_first, display_label, phrase_second)
            answer = "answer: {} {}".format(answer_label, dist_label)
        f_out.write(left + " " + right + "\t" + answer + "\n")
