import json
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import spacy
import torch
from nltk import word_tokenize
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModel

nlp = spacy.load("en_core_web_sm")

with open('cor.json') as data:
    file = json.load(data)

dialogues = []

for d in file[:2]:
    samples = defaultdict(dict)
    result = d['completions'][0]['result']
    texts_without_labels = d['data']['text']
    for sample in result:
        speaker = texts_without_labels[int(sample['value']['start'])]['speaker']
        samples[sample['id']]['speaker'] = speaker
        samples[sample['id']]['text'] = sample['value']['text']
        samples[sample['id']]['start'] = int(sample['value']['start'])
        if 'paragraphlabels' in sample['value']:
            samples[sample['id']]['paragraphlabels'] = sample['value']['paragraphlabels'][0]
        if 'choices' in sample['value']:
            samples[sample['id']]['choices'] = sample['value']['choices'][0]

    sorted_samples = sorted([(samples[sample_id]['start'], sample_id) for sample_id in samples])
    texts = []
    labels = []
    speakers = []
    for _, sample_id in sorted_samples:
        if samples[sample_id]['text'] != 'PAUSE':
            texts.append(str(samples[sample_id]['text']).replace('\n', ''))
            speakers.append(samples[sample_id]['speaker'])
            paragraph_labels = samples[sample_id].get('paragraphlabels', '')
            choices = samples[sample_id].get('choices', '')
            labels.append(paragraph_labels + '.' + choices)
    dialogues.append((texts, labels, speakers))

train_data = dialogues[1][0]
test_data = dialogues[0][0]

train_labels = dialogues[1][1]
test_labels = dialogues[0][1]

def delete_odds(list_with_lines):
    for i in range(len(list_with_lines)):
        if 'them' in list_with_lines[i]:
            list_with_lines[i] = list_with_lines[i].replace('them', ' them ')
        if ' em ' in list_with_lines[i]:
            list_with_lines[i] = list_with_lines[i].replace(' em ', ' them ')
        if 'laugh' in list_with_lines[i]:
            list_with_lines[i] = list_with_lines[i].replace('laugh', '')
        if 'uh?' in list_with_lines[i]:
            list_with_lines[i] = list_with_lines[i].replace('uh?', '')
        if 'ʔuh' in list_with_lines[i]:
            list_with_lines[i] = list_with_lines[i].replace('ʔuh', '')
        if 'ʔ' in list_with_lines[i]:
            list_with_lines[i] = list_with_lines[i].replace('ʔ', '')
    return list_with_lines

previous_lines_sust=[]
sustains=[]
sus_tags=[]
for i in range(len(train_data)):
    if 'Sustain' in train_labels[i]:
            previous_lines_sust.append(train_data[i-1])
            sustains.append(train_data[i])
            sus_tags.append(train_labels[i])
for i in range(len(test_data)):
    if 'Sustain' in test_labels[i]:
        previous_lines_sust.append(test_data[i-1])
        sustains.append(test_data[i])
        sus_tags.append(test_labels[i])

for i in range(len(sus_tags)):
    if 'Append' in sus_tags[i]:
        sus_tags[i]=re.sub('Append','Prolong', sus_tags[i])

sustains=delete_odds(sustains)

responds=[]
previous_responds=[]
respond_tags=[]
for i in range(len(train_data)):
    if 'Answer' in train_labels[i]:
        continue
    if 'Disengage' in train_labels[i]:
        continue
    elif 'Respond' in train_labels[i]:
        responds.append(train_data[i])
        previous_responds.append(train_data[i-1])
        respond_tags.append(train_labels[i])

for i in range(len(test_data)):
    if 'Answer' in test_labels[i]:
        continue
    if 'Disengage' in test_labels[i]:
        continue
    elif 'Respond' in test_labels[i]:
        responds.append(test_data[i])
        previous_responds.append(test_data[i-1])
        respond_tags.append(test_labels[i])

def clean_responds(respond_tags):
    for i in range(len(respond_tags)):
        if 'Decline' in respond_tags[i]:
            respond_tags[i]=re.sub('Decline','Contradict',respond_tags[i])
        if 'Accept' in respond_tags[i]:
            respond_tags[i]=re.sub('Accept','Affirm',respond_tags[i])
        tag_list=respond_tags[i].split('.')[-2:]
        respond_tags[i]='.'.join(tag_list)
    return respond_tags

respond_tags=clean_responds(respond_tags)

previous_lines=[]
replies=[]
tags=[]
for i in range(len(train_data)):
    if 'Reply' in train_labels[i]:
        if '?' not in train_labels[i]:
            previous_lines.append(train_data[i-1])
            replies.append(train_data[i])
            tags.append(train_labels[i])

for i in range(len(test_data)):
    if 'Reply' in test_labels[i]:
        if '?' not in test_labels[i]:
            previous_lines.append(test_data[i-1])
            replies.append(test_data[i])
            tags.append(test_labels[i])

for i in range(len(tags)):
    tag_list = tags[i].split('.')[-2:]
    tags[i]=str('.'.join(tag_list))
    if 'Answer' in tags[i]:
        tags[i]='Response.Resolve.'

train_speakers=dialogues[1][2]
test_speakers = dialogues[0][2]

train_data = delete_odds(train_data)
test_data = delete_odds(test_data)

def get_cut_labels(labels):
    for i in range(len(labels)):
        if labels[i].startswith('Open'):
            labels[i] = 'Open.'
        if labels[i].startswith('React.Rejoinder.'):
            labels[i] = 'React.Rejoinder.'
        if labels[i].startswith('React.Respond.'):
            labels[i] = 'React.Respond.'
        if labels[i].startswith('Sustain.Continue.'):
            labels[i] = 'Sustain.Continue.'
    return labels

cut_train_labels = get_cut_labels(train_labels)

cut_test_labels = get_cut_labels(test_labels)

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/bert-base-cased-conversational")
embed_model = AutoModel.from_pretrained("DeepPavlov/bert-base-cased-conversational")

def get_embeddings(data):
    outputs = []
    for text in data:
        with torch.no_grad():
            input_ph= tokenizer(text, padding=True, truncation=True, max_length=30,return_tensors="pt")
            output_ph = embed_model(**input_ph)
    #        train_outputs.append(output_ph.pooler_output.cpu().numpy())
            sentence_embedding = output_ph.last_hidden_state.mean(dim=1).cpu().numpy()
        outputs.append(sentence_embedding)
    outputs = np.concatenate(outputs)
    return outputs

all_outputs=[]
all_outputs.extend(get_embeddings(train_data))
all_outputs.extend(get_embeddings(test_data))

all_cuts=[]
all_cuts.extend(cut_train_labels)
all_cuts.extend(cut_test_labels)

path2 = 'res/SBC058.json'
with open(path2) as proba:
    proba=json.load(proba)
    dialogue = proba['text']
    dialogue = [i for i in dialogue if not (i['phrase'] == 'PAUSE')]
    phrases = []
    speakers=[]
    for i in range(len(dialogue)):
        phrases.append(dialogue[i]['phrase'])
        speakers.append(dialogue[i]['speaker'])

phrases = delete_odds(phrases)

test_embeddings = get_embeddings(phrases)

model = LogisticRegression(C=0.01,class_weight='balanced')
model.fit(all_outputs,all_cuts)
y_pred_proba = model.predict_proba(test_embeddings)
y_pred = model.predict(test_embeddings)

y_pred =list(y_pred)

#Develop
for i in range(len(y_pred)):
    if y_pred[i]=='Sustain.Continue.':
        if speakers[i]!=speakers[i-1]:
            if y_pred[i-1]=='Sustain.Continue.':
                y_pred[i]='React.Respond.Develop.'

#Open is always the first one
y_pred[0]='Open.'
first_speaker=speakers[0]
for i in range(len(speakers)):
    if speakers[i]==first_speaker:
        y_pred[i]='Open.'
    else:
        break

interrogative_words = ['whose', 'what', 'which', 'who', 'whom', 'what', 'which','why', 'where', 'when', 'how']

questions=[]
for i in range(len(phrases)):
    if '?' in phrases[i]:
        questions.append(phrases[i])

with open('track_list') as track_list:
    track_list= track_list.readlines()
train_que =[]
train_tags=[]
for line in track_list:
    line = line.split('/')
    train_que.append(line[0])
    train_tags.append(line[1][:-1])

train_em_que = get_embeddings(train_que)

question_embeddings = get_embeddings(questions)

lr2 = LogisticRegression(C=0.01, class_weight='balanced')
lr2.fit(train_em_que,train_tags)
tags_for_track = lr2.predict(question_embeddings)

true_tracks = {'1':'Track.Check', '2':'Track.Confirm', '3':'Track.Clarify','4':'Track.Probe'}

def map_tracks(tags_for_track, true_tags):
    real_tracks =[]
    for i in range(len(list(tags_for_track))):
        if tags_for_track[i]=='5':
            real_tracks.append(tags_for_track[i])
        else:
            real_tracks.append(true_tracks[tags_for_track[i]])
    return real_tracks

tags_for_track=map_tracks(tags_for_track,true_tracks)

for i in range(len(phrases)):
    if phrases[i] in questions:
        que_index = questions.index(phrases[i])
        if speakers[i]!=speakers[i-1]:
            if y_pred[i]=='React.Respond.' and tags_for_track[que_index]!='5':
                y_pred[i]='React.Rejoinder.'+tags_for_track[que_index]
            if  y_pred[i]=='React.Rejoinder.' and tags_for_track[que_index]!='5':
                y_pred[i]+=tags_for_track[que_index]
            if y_pred[i]=='React.Rejoinder.' and tags_for_track[que_index]=='5':
                for word in interrogative_words:
                        if word in phrases[i]:
                            y_pred[i]='React.Rejoinder.Rebound'
                        else:
                            y_pred[i]='React.Rejoinder.Re-challenge'
            if y_pred[i]=='Sustain.Continue.':
                y_pred[i]='React.Rejoinder.'+tags_for_track[que_index]
            if y_pred[i]=='Open.':
                pass
        if speakers[i]==speakers[i-1]:
            if  y_pred[i]=='React.Rejoinder.' and tags_for_track[que_index]!='5':
                y_pred[i]+=tags_for_track[que_index]
            if y_pred[i]=='React.Rejoinder.' and tags_for_track[que_index]=='5':
                for word in interrogative_words:
                        if word in phrases[i]:
                            y_pred[i]='React.Rejoinder.Rebound'
                        else:
                            y_pred[i]='React.Rejoinder.Re-challenge'
            if y_pred[i]=='Open.':
                pass
            if y_pred[i]=='Sustain.Continue.':
                y_pred[i]='Sustain.Continue.Monitor'

for i in range(len(speakers)):
    if '?' not in phrases[i-1] and '?' not in phrases[i]:
        if y_pred[i]=='React.Respond.':
            if speakers[i-1]!=speakers[i] and speakers[i+1]!=speakers[i]:
                if len(word_tokenize(phrases[i]))==2:
                    doc=nlp(phrases[i])
                    for token in doc:
                        token_pos=token.pos_
                        if token_pos!='VERB':
                            y_pred[i]='React.Respond.Support.Register'

for i in range(len(phrases)):
    if speakers[i-1]==speakers[i]:
        if phrases[i] not in questions:
            if y_pred[i-1]=='Sustain.Continue.':
                    if y_pred[i+1]=='Sustain.Continue.':
                        y_pred[i]='Sustain.Continue.'
                    else:
                        if y_pred[i]!='Open.':
                            if phrases[i] not in questions:
                                y_pred[i]='Sustain.Continue.'

for i in range(len(phrases)):
    if phrases[i] not in questions:
        if phrases[i-1] not in questions:
            if speakers[i-1]!=speakers[i]:
                if y_pred[i]=='Sustain.Continue.':
                    y_pred[i]='React.Respond.Develop.'

test_sustains=[]
for i in range(len(phrases)):
    if y_pred[i]=='Sustain.Continue.' or y_pred[i]=='React.Respond.Develop.':
        test_sustains.append(phrases[i])

train_sustains=get_embeddings(sustains)
test_sustains_emb=get_embeddings(test_sustains)

lr_sus = LogisticRegression(C=0.01, class_weight='balanced')
lr_sus.fit(train_sustains,sus_tags)
tags_for_sus= lr_sus.predict(test_sustains_emb)

for i in range(len(test_sustains)):
    sus_index = phrases.index(test_sustains[i])
    if y_pred[sus_index]=='Sustain.Continue.':
        y_pred[sus_index]=tags_for_sus[i]
    if y_pred[sus_index]=='React.Respond.Develop.':
        cut_tags=tags_for_sus[i].split('.')[-1]
        if cut_tags!='Monitor':
            y_pred[sus_index]+=cut_tags

for i in range(len(phrases)):
    if y_pred[i]=='Sustain.Continue.':
        if 'you know' in phrases[i].lower():
            y_pred[i]='Sustain.Continue.Monitor'
        else:
            y_pred[i]='Sustain.Continue.Prolong.Extend'

#replies после вопросов
try_replies=[]
test_prev_lines = []
indexes = []
test_responds=[]
test_responds_prev_lines = []
re_indexes = []
for i in range(len(speakers)):
    if phrases[i-1] in questions:
        if y_pred[i]=='React.Respond.':
            if speakers[i]!=speakers[i-1]:
                test_prev_lines.append(phrases[i-1])
                try_replies.append(phrases[i])
                indexes.append(i)
            else:
                test_responds_prev_lines.append(phrases[i-1])
                test_responds.append(phrases[i])
                re_indexes.append(i)
#                 print(speakers[i-1],phrases[i-1],y_pred[i-1])
#                 print(speakers[i],phrases[i],y_pred[i])
#                 print(speakers[i+1],phrases[i+1],y_pred[i+1])
#                 print('__________')

try_reply_emb = get_embeddings(try_replies)

test_prev = get_embeddings(test_prev_lines)

test_concat = np.concatenate([try_reply_emb,test_prev],axis=1)

train_embed_replies=get_embeddings(replies)

train_prev_lines=get_embeddings(previous_lines)

reply_concatenate = np.concatenate([train_embed_replies,train_prev_lines], axis=1)

lr_reply = LogisticRegression(C=0.01, class_weight='balanced')
lr_reply.fit(reply_concatenate,tags)
tags_for_reply = lr_reply.predict(test_concat)

for i in range(len(try_replies)):
    y_pred[indexes[i]]=y_pred[indexes[i]]+tags_for_reply[i]

for i in range(len(try_replies)):
    if 'yes' in try_replies[i].lower():
        if y_pred[indexes[i]]=='React.Respond.Reply.Disagree':
            if 'Confirm' in y_pred[indexes[i]-1]:
                y_pred[indexes[i]]='React.Respond.Reply.Affirm'

for i in range(len(speakers)):
    if phrases[i-1] not in questions:
        if y_pred[i]=='React.Respond.':
            print(speakers[i-1],phrases[i-1],y_pred[i-1])
            print(speakers[i],phrases[i],y_pred[i])
            print(speakers[i+1],phrases[i+1],y_pred[i+1])
            print('__________')

#replies после других предложений
test_responds=[]
test_responds_prev_lines = []
re_indexes = []
for i in range(len(speakers)):
    if phrases[i-1] not in questions:
        if y_pred[i]=='React.Respond.':
            test_responds_prev_lines.append(phrases[i-1])
            test_responds.append(phrases[i])
            re_indexes.append(i)

test_responds_emb=get_embeddings(test_responds)

test_prev_responds=get_embeddings(test_responds_prev_lines)

test_responds_concatenate = np.concatenate([test_responds_emb,test_prev_responds], axis=1)

train_emb_responds=get_embeddings(responds)

train_prev_responds=get_embeddings(previous_responds)

responds_concatenate = np.concatenate([train_emb_responds,train_prev_responds], axis=1)

lr_responds = LogisticRegression(C=0.5, class_weight='balanced')
lr_responds.fit(responds_concatenate,respond_tags)
tags_for_responds = lr_responds.predict(test_responds_concatenate)

for i in range(len(test_responds)):
    y_pred[re_indexes[i]]=y_pred[re_indexes[i]]+tags_for_responds[i]
    if ' no ' in test_responds[i].lower():
        y_pred[re_indexes[i]]='Rejoinder.Counter'

for i in range(len(phrases)):
    if y_pred[i]=='React.Rejoinder.':
        if 'Develop' in y_pred[i-1]:
            if speakers[i-1]==speakers[i+1]:
                y_pred[i]='Sustain.Continue.Prolong.Extend'
        if ' no ' in phrases[i].lower():
            y_pred[i]='React.Rejoinder.Counter'
        if phrases[i-1] in questions:
            if speakers[i-1]!=speakers[i]:
                y_pred[i]='React.Rejoinder.Response.Resolve'
            else:
                y_pred[i]='React.Rejoinder.Re-challenge'
        else:
            y_pred[i]='React.Respond.Develop.Extend'

for i in range(len(phrases)):
    if y_pred[i]=='Open.':
        if len(word_tokenize(phrases[i]))<3:
            poses=[]
            doc=nlp(phrases[i])
            for token in doc:
                poses.append(token.pos_)
            if 'PROPN' in poses:
                y_pred[i]='Open.Attend'

def load_pickle(filepath):
    documents_f = open(filepath, 'rb')
    file = pickle.load(documents_f)
    documents_f.close()
    return file

def divide_into_sentences(document):
    return [sent for sent in document.sents]


def number_of_fine_grained_pos_tags(sent):
    tag_dict = {'-LRB-': 0, '-RRB-': 0, ',': 0, ':': 0, '.': 0, "''": 0, '""': 0, '#': 0,
                '``': 0, '$': 0, 'ADD': 0, 'AFX': 0, 'BES': 0, 'CC': 0, 'CD': 0, 'DT': 0,
                'EX': 0, 'FW': 0, 'GW': 0, 'HVS': 0, 'HYPH': 0, 'IN': 0, 'JJ': 0, 'JJR': 0,
                'JJS': 0, 'LS': 0, 'MD': 0, 'NFP': 0, 'NIL': 0, 'NN': 0, 'NNP': 0, 'NNPS': 0,
                'NNS': 0, 'PDT': 0, 'POS': 0, 'PRP': 0, 'PRP$': 0, 'RB': 0, 'RBR': 0, 'RBS': 0,
                'RP': 0, '_SP': 0, 'SYM': 0, 'TO': 0, 'UH': 0, 'VB': 0, 'VBD': 0, 'VBG': 0,
                'VBN': 0, 'VBP': 0, 'VBZ': 0, 'WDT': 0, 'WP': 0, 'WP$': 0, 'WRB': 0, 'XX': 0,
                'OOV': 0, 'TRAILING_SPACE': 0}
    for token in sent:
        if token.is_oov:
            tag_dict['OOV'] += 1
        elif token.tag_ == '':
            tag_dict['TRAILING_SPACE'] += 1
        else:
            tag_dict[token.tag_] += 1

    return tag_dict


def number_of_dependency_tags(sent):
    dep_dict = {'acl': 0, 'advcl': 0, 'advmod': 0, 'amod': 0, 'appos': 0, 'aux': 0, 'case': 0,
                'cc': 0, 'ccomp': 0, 'clf': 0, 'compound': 0, 'conj': 0, 'cop': 0, 'csubj': 0,
                'dep': 0, 'det': 0, 'discourse': 0, 'dislocated': 0, 'expl': 0, 'fixed': 0,
                'flat': 0, 'goeswith': 0, 'iobj': 0, 'list': 0, 'mark': 0, 'nmod': 0, 'nsubj': 0,
                'nummod': 0, 'obj': 0, 'obl': 0, 'orphan': 0, 'parataxis': 0, 'prep': 0, 'punct': 0,
                'pobj': 0, 'dobj': 0, 'attr': 0, 'relcl': 0, 'quantmod': 0, 'nsubjpass': 0,
                'reparandum': 0, 'ROOT': 0, 'vocative': 0, 'xcomp': 0, 'auxpass': 0, 'agent': 0,
                'poss': 0, 'pcomp': 0, 'npadvmod': 0, 'predet': 0, 'neg': 0, 'prt': 0, 'dative': 0,
                'oprd': 0, 'preconj': 0, 'acomp': 0, 'csubjpass': 0, 'meta': 0, 'intj': 0,
                'TRAILING_DEP': 0}

    for token in sent:
        if token.dep_ == '':
            dep_dict['TRAILING_DEP'] += 1
        else:
            try:
                dep_dict[token.dep_] += 1
            except:
                print('Unknown dependency for token: "' + token.orth_ + '". Passing.')

    return dep_dict

def number_of_specific_entities(sent):
    entity_dict = {
    'PERSON': 0, 'NORP': 0, 'FAC': 0, 'ORG': 0, 'GPE': 0, 'LOC': 0,
    'PRODUCT': 0, 'EVENT': 0, 'WORK_OF_ART': 0, 'LAW': 0, 'LANGUAGE': 0,
    'DATE': 0, 'TIME': 0, 'PERCENT': 0, 'MONEY': 0, 'QUANTITY': 0,
    'ORDINAL': 0, 'CARDINAL': 0 }
    entities = [ent.label_ for ent in sent.as_doc().ents]
    for entity in entities:
        entity_dict[entity]+=1
    return entity_dict


def predict(test_sent, classifier, scaler=None):
    parsed_test = divide_into_sentences(nlp(test_sent))
    # Get features
    sentence_with_features = {}
    entities_dict = number_of_specific_entities(parsed_test[0])
    sentence_with_features.update(entities_dict)
    pos_dict = number_of_fine_grained_pos_tags(parsed_test[0])
    sentence_with_features.update(pos_dict)
    # dep_dict = number_of_dependency_tags(parsed_test[0])
    # sentence_with_features.update(dep_dict)
    df = pd.DataFrame(sentence_with_features, index=[0])
    if scaler:
        df = scaler.transform(df)

    prediction = classifier.predict(df)
    if prediction == 0:
        open_list.append('Fact')
    else:
        open_list.append('Opinion')

nn_classifier = load_pickle('nn_classifier.pickle')

scaler = load_pickle('scaler.pickle')

open_list = []

open_phrases=[]
open_index=[]
for i in range(len(phrases)):
    if y_pred[i]=='Open.':
        predict(phrases[i],nn_classifier,scaler)
        open_phrases.append(phrases[i])
        open_index.append(i)

for i in range(len(open_phrases)):
    if open_list[i]=='Fact':
        if open_phrases[i] not in questions:
            open_list[i]='Give.Fact.'
        else:
            open_list[i]='Demand.Fact.'
    else:
        if open_phrases[i] not in questions:
            open_list[i]='Give.Opinion.'
        else:
            open_list[i]='Demand.Opinion.'

for i in range(len(open_phrases)):
    y_pred[open_index[i]]=y_pred[open_index[i]]+open_list[i]

y_true_tests = [
    'Open.Give.Fact.', 'React.Respond.Reply.Agree', 'React.Respond.Support.Engage', 'React.Rejoinder.Track.Check',
    'Sustain.Continue.Prolong.Enhance', 'React.Respond.Develop.Elaborate', 'React.Respond.Support.Register',
    'React.Respond.Reply.Disagree', 'Sustain.Continue.Prolong.Elaborate', 'React.Respond.Develop.Elaborate',
    'React.Respond.Reply.Acknowledge', 'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Affirm',
    'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Affirm', 'React.Rejoinder.Track.Clarify',
    'React.Respond.Response.Resolve.', 'Sustain.Continue.Prolong.Elaborate', 'Sustain.Continue.Prolong.Extend',
    'Sustain.Continue.Prolong.Elaborate', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend',
    'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Monitor', 'Open.Demand.Fact.',
    'Sustain.Continue.Prolong.Elaborate', 'React.Respond.Develop.Enhance', 'Sustain.Continue.Prolong.Enhance',
    'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Enhance',
    'React.Rejoinder.Track.Confirm', 'React.Rejoinder.Response.Resolve', 'Open.Give.Fact.',
    'React.Rejoinder.Track.Probe', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Enhance',
    'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Affirm', 'React.Respond.Develop.Enhance',
    'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend',
    'Sustain.Continue.Prolong.Extend', 'Open.Give.Fact.', 'Open.Give.Fact.', 'React.Respond.Develop.Extend',
    'React.Respond.Support.Register', 'React.Respond.Reply.Agree', 'React.Respond.Reply.Acknowledge', 'Open.Give.Fact.',
    'React.Rejoinder.Track.Clarify', 'React.Respond.', 'React.Respond.Reply.Affirm', 'Sustain.Continue.Prolong.Enhance',
    'React.Respond.Develop.Elaborate', 'React.Respond.Support.Register', 'React.Respond.Develop.Extend',
    'React.Respond.Support.Register', 'React.Respond.Develop.Extend', 'React.Respond.Support.Register',
    'React.Respond.Develop.Extend', 'Open.Give.Fact.', 'React.Respond.Reply.Agree',
    'Sustain.Continue.Prolong.Elaborate', 'React.Rejoinder.Track.Confirm', 'React.Respond.',
    'React.Respond.Support.Register', 'Open.Attend', 'React.Rejoinder.Re-challenge', 'Sustain.Continue.Prolong.Extend',
    'Sustain.Continue.Prolong.Extend', 'React.Respond.Develop.Extend', 'React.Respond.Develop.Extend',
    'Sustain.Continue.Prolong.Extend', 'React.Rejoinder.Track.Confirm', 'React.Rejoinder.Track.Check',
    'React.Rejoinder.Track.Confirm', 'React.Rejoinder.Track.Probe', 'Sustain.Continue.Prolong.Extend',
    'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Develop.Enhance',
    'React.Respond.Reply.Affirm', 'React.Rejoinder.Track.Clarify', 'React.Respond.Reply.Disawow',
    'Sustain.Continue.Prolong.Extend', 'React.Respond.Develop.Extend', 'React.Respond.Develop.Elaborate',
    'React.Respond.Support.Register', 'React.Respond.Develop.Extend', 'React.Rejoinder.Re-challenge',
    'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Develop.Enhance',
    'React.Respond.Reply.Affirm', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend',
    'React.Respond.Support.Register', 'React.Respond.Develop.Enhance', 'Sustain.Continue.Prolong.Extend',
    'React.Respond.Develop.Extend', 'React.Respond.Reply.Affirm', 'Sustain.Continue.Prolong.Enhance',
    'Sustain.Continue.Prolong.Enhance', 'React.Respond.Support.Register', 'React.Rejoinder.Track.Confirm',
    'React.Respond.Reply.Affirm', 'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Affirm',
    'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend',
    'Sustain.Continue.Prolong.Enhance', 'React.Rejoinder.Track.Confirm', 'Sustain.Continue.Prolong.Extend',
    'React.Respond.Support.Register', 'React.Respond.Develop.Extend', 'Sustain.Continue.Monitor',
    'Sustain.Continue.Prolong.Enhance', 'React.Rejoinder.Track.Check', 'React.Respond.Response.Resolve.',
    'React.Rejoinder.Track.Confirm', 'React.Rejoinder.Track.Clarify', 'React.Respond.Reply.Disagree',
    'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Disagree', 'React.Respond.Develop.Enhance',
    'React.Respond.Support.Register', 'React.Respond.Reply.Affirm', 'Open.Give.Fact.',
    'Sustain.Continue.Prolong.Enhance', 'React.Respond.Support.Register', 'React.Respond.Develop.Extend',
    'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Extend',
    'Sustain.Continue.Prolong.Enhance', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Enhance',
    'React.Respond.Support.Register', 'React.Rejoinder.Track.Clarify', 'React.Rejoinder.Track.Clarify',
    'React.Rejoinder.Track.Clarify', 'React.Respond.Reply.Affirm', 'Sustain.Continue.Prolong.Enhance',
    'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Enhance',
    'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Affirm', 'Sustain.Continue.Prolong.Elaborate',
    'React.Respond.Develop.Enhance', 'React.Rejoinder.Track.Check', 'React.Respond.Reply.Affirm',
    'Sustain.Continue.Prolong.Elaborate', 'React.Rejoinder.Track.Probe', 'React.Respond.Reply.Affirm',
    'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Support.Register',
    'React.Respond.Develop.Extend', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Extend',
    'Sustain.Continue.Prolong.Extend', 'React.Respond.Support.Register', 'React.Respond.Develop.Extend',
    'Sustain.Continue.Prolong.Enhance', 'React.Respond.Develop.Elaborate', 'React.Respond.Support.Register',
    'React.Respond.Support.Register', 'React.Respond.Support.Register', 'React.Respond.Support.Register',
    'React.Rejoinder.Re-challenge', 'React.Respond.Reply.Disawow', 'Open.Demand.Fact.', 'React.Respond.',
    'Open.Give.Fact.', 'React.Respond.Develop.Elaborate', 'React.Rejoinder.Track.Clarify', 'React.Respond.Reply.Affirm',
    'Sustain.Continue.Prolong.Extend', 'React.Respond.Reply.Disawow', 'Sustain.Continue.Prolong.Extend',
    'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Develop.Enhance',
    'React.Rejoinder.Track.Confirm', 'React.Rejoinder.Track.Clarify', 'React.Respond.Response.Resolve.',
    'React.Respond.Support.Register', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Elaborate',
    'React.Rejoinder.Track.Clarify', 'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Affirm',
    'React.Respond.Reply.Affirm', 'React.Respond.Support.Register', 'React.Respond.Develop.Extend',
    'Sustain.Continue.Monitor', 'Sustain.Continue.Prolong.Elaborate', 'Sustain.Continue.Prolong.Elaborate',
    'Sustain.Continue.Monitor', 'React.Respond.Support.Register', 'React.Respond.Develop.Enhance',
    'React.Rejoinder.Track.Check', 'React.Rejoinder.Track.Check', 'React.Rejoinder.Response.Resolve',
    'React.Respond.Develop.Extend', 'React.Respond.Reply.Affirm', 'Sustain.Continue.Prolong.Extend', 'Open.Give.Fact.', 'React.Rejoinder.Re-challenge', 'React.Respond.Reply.Affirm', 'React.Rejoinder.Track.Check', 'React.Rejoinder.Re-challenge', 'Open.Give.Opinion.', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'React.Rejoinder.Track.Clarify', 'React.Rejoinder.Track.Probe', 'React.Respond.Reply.Affirm', 'Sustain.Continue.Prolong.Elaborate', 'React.Rejoinder.Track.Confirm', 'React.Rejoinder.Track.Check', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Support.Register', 'React.Respond.Develop.Elaborate', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Elaborate', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Elaborate', 'React.Respond.Develop.', 'React.Respond.Support.Register', 'React.Respond.Reply.Agree', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Elaborate', 'React.Rejoinder.Track.Confirm', 'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Affirm', 'React.Respond.Develop.Enhance', 'React.Respond.Develop.Elaborate', 'React.Respond.Develop.Extend', 'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Affirm', 'Sustain.Continue.Prolong.Enhance', 'React.Respond.Support.Register', 'React.Respond.Develop.Extend', 'React.Rejoinder.Track.Clarify', 'React.Rejoinder.Response.Resolve', 'React.Respond.Develop.Extend', 'React.Respond.Support.Register', 'React.Respond.Reply.Affirm', 'React.Respond.Develop.Enhance', 'Sustain.Continue.Prolong.Elaborate', 'React.Rejoinder.Track.Confirm', 'React.Respond.', 'Open.Attend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'React.Respond.Support.Register', 'React.Respond.Develop.Enhance', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Monitor', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Elaborate', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Monitor', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Elaborate', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Monitor', 'Open.Give.Fact.', 'React.Respond.Develop.Extend', 'React.Respond.Support.Register', 'React.Respond.Develop.Extend', 'React.Respond.Reply.Agree', 'Open.Give.Fact.', 'React.Respond.Develop.Elaborate', 'Open.Give.Fact.', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'Open.Give.Fact.', 'React.Rejoinder.Re-challenge', 'React.Respond.Reply.Affirm', 'React.Respond.Support.Register', 'React.Respond.Develop.Extend', 'React.Respond.Support.Engage', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Enhance', 'React.Respond.Reply.Acknowledge', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Elaborate', 'Open.Give.Fact.', 'React.Respond.Reply.Agree', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'React.Rejoinder.Track.Clarify', 'React.Respond.', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Elaborate', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Enhance', 'React.Respond.Develop.Enhance', 'React.Respond.Support.Register', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Extend', 'React.Rejoinder.Track.Check', 'React.Rejoinder.Response.Resolve', 'React.Respond.Support.Engage', 'Sustain.Continue.Prolong.Elaborate', 'Sustain.Continue.Prolong.Enhance', 'React.Rejoinder.Track.Check', 'React.Rejoinder.Response.Resolve', 'React.Rejoinder.Track.Confirm', 'Sustain.Continue.Prolong.Enhance', 'React.Respond.Develop.Enhance', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Affirm', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Elaborate', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Affirm', 'React.Rejoinder.Re-challenge', 'React.Respond.Reply.Affirm', 'React.Respond.Support.Register', 'Sustain.Continue.Prolong.Extend', 'React.Rejoinder.Re-challenge', 'React.Respond.', 'React.Respond.Develop.Extend', 'React.Rejoinder.Re-challenge', 'React.Rejoinder.Re-challenge', 'Open.Give.Fact.', 'React.Rejoinder.Track.Check', 'React.Respond.Reply.Affirm', 'React.Respond.Support.Register', 'React.Respond.Develop.Elaborate', 'React.Rejoinder.Track.Confirm', 'React.Rejoinder.Re-challenge', 'React.Rejoinder.Track.Clarify', 'React.Rejoinder.Track.Probe', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Develop.Extend', 'React.Rejoinder.Track.Probe', 'React.Respond.Reply.Affirm', 'Sustain.Continue.Prolong.Enhance', 'React.Rejoinder.Track.Clarify', 'Sustain.Continue.Prolong.Enhance', 'React.Respond.Develop.Extend', 'React.Respond.Reply.Disagree', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Monitor', 'Sustain.Continue.Prolong.Elaborate', 'React.Respond.Develop.Enhance', 'React.Respond.Support.Register', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Develop.Elaborate', 'React.Respond.Develop.Extend', 'React.Respond.Reply.Disagree', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Reply.Acknowledge', 'React.Respond.Support.Register', 'Open.Give.Fact.', 'React.Respond.Develop.Elaborate', 'React.Rejoinder.Re-challenge', 'React.Respond.Reply.Disagree', 'React.Respond.Develop.Enhance', 'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Affirm', 'React.Rejoinder.Re-challenge', 'React.Respond.Reply.Affirm', 'React.Respond.Support.Register', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Enhance', 'React.Respond.Support.Register', 'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Affirm', 'Open.Give.Fact.', 'Sustain.Continue.Monitor', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Support.Register', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Elaborate', 'React.Respond.Develop.Extend', 'React.Respond.Develop.Enhance', 'React.Respond.Reply.Disawow', 'React.Respond.Reply.Agree', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Elaborate', 'React.Respond.Develop.Elaborate', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Support.Register', 'React.Respond.Develop.Extend', 'React.Respond.Support.Register', 'React.Respond.Develop.Enhance', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Monitor', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Monitor', 'Sustain.Continue.Prolong.Enhance', 'React.Rejoinder.Track.Probe', 'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Affirm', 'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Affirm', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Affirm', 'React.Respond.Develop.Enhance', 'React.Respond.Support.Register', 'React.Respond.Support.Register', 'React.Respond.Develop.Elaborate', 'React.Respond.Reply.Disawow', 'React.Rejoinder.Re-challenge', 'React.Respond.Reply.Affirm', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Support.Register', 'React.Rejoinder.Track.Clarify', 'React.Respond.Reply.Affirm', 'React.Rejoinder.Track.Probe', 'React.Respond.Reply.Affirm', 'React.Respond.Support.Register', 'React.Respond.Reply.Affirm', 'React.Rejoinder.Rebound', 'React.Rejoinder.Track.Probe', 'React.Respond.Reply.Affirm', 'React.Rejoinder.Track.Clarify', 'React.Rejoinder.Track.Confirm', 'React.Rejoinder.Track.Check', 'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Disawow', 'React.Respond.Develop.Extend', 'React.Respond.Develop.Elaborate', 'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Decline', 'React.Respond.Develop.Elaborate', 'Open.Give.Fact.', 'React.Respond.Support.Register', 'React.Rejoinder.Re-challenge', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Enhance', 'React.Respond.Support.Register', 'Sustain.Continue.Prolong.Enhance', 'Open.Give.Fact.', 'React.Rejoinder.Track.Probe', 'React.Respond.Reply.Affirm', 'Open.Give.Fact.', 'React.Respond.Support.Register', 'React.Respond.Develop.Elaborate', 'React.Respond.Support.Register', 'Sustain.Continue.Prolong.Elaborate', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Elaborate', 'Sustain.Continue.Prolong.Elaborate', 'React.Respond.Support.Register', 'Sustain.Continue.Prolong.Enhance', 'React.Rejoinder.Re-challenge', 'React.Respond.Reply.Decline', 'Sustain.Continue.Prolong.Elaborate', 'React.Respond.Support.Register', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Monitor', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Enhance', 'React.Respond.Support.Register', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Enhance', 'React.Respond.Develop.Extend', 'React.Rejoinder.Track.Probe', 'React.Rejoinder.Track.Clarify', 'React.Respond.Response.Resolve.', 'React.Respond.Support.Register', 'React.Respond.Develop.Enhance', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Reply.Agree', 'React.Respond.Develop.Enhance', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Monitor', 'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Decline', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Reply.Acknowledge', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Elaborate', 'React.Respond.Reply.Affirm', 'Sustain.Continue.Prolong.Elaborate', 'React.Respond.Support.Register', 'React.Respond.Reply.Affirm', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Elaborate', 'React.Respond.Develop.Enhance', 'React.Respond.Support.Register', 'React.Respond.Develop.Extend', 'React.Respond.Support.Register', 'React.Rejoinder.Track.Clarify', 'React.Rejoinder.Response.Resolve', 'React.Rejoinder.Track.Confirm', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Support.Register', 'Sustain.Continue.Prolong.Enhance', 'React.Rejoinder.Track.Probe', 'React.Respond.Reply.Affirm', 'React.Rejoinder.Re-challenge', 'React.Respond.Reply.Affirm', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Elaborate', 'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Affirm', 'React.Respond.Reply.Agree', 'React.Respond.Support.Register', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Monitor', 'React.Rejoinder.Track.Clarify', 'Open.Give.Fact.', 'React.Respond.Support.Register', 'Open.Give.Fact.', 'React.Respond.Develop.Elaborate', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'React.Rejoinder.Track.Check', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'React.Respond.Support.Register', 'React.Respond.Develop.Extend', 'React.Respond.Reply.Affirm', 'React.Rejoinder.Re-challenge', 'React.Rejoinder.Track.Check', 'React.Respond.Response.Resolve.', 'React.Respond.Support.Register', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Elaborate', 'Sustain.Continue.Prolong.Elaborate', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Develop.Extend', 'React.Respond.Develop.Enhance', 'React.Rejoinder.Track.Clarify', 'Sustain.Continue.Prolong.Enhance', 'React.Rejoinder.Track.Probe', 'React.Rejoinder.Re-challenge', 'React.Rejoinder.Track.Confirm', 'Sustain.Continue.Prolong.Enhance', 'Open.Give.Fact.', 'React.Respond.Support.Register', 'Sustain.Continue.Prolong.Enhance', 'React.Rejoinder.Track.Clarify', 'React.Rejoinder.Track.Confirm', 'React.Rejoinder.Track.Probe', 'React.Respond.Reply.Affirm', 'React.Respond.Develop.Extend', 'React.Respond.Develop.Enhance', 'Open.Give.Fact.', 'React.Respond.Develop.Extend', 'React.Respond.Develop.Enhance', 'React.Respond.Support.Register', 'Open.Demand.Fact.', 'React.Rejoinder.Re-challenge', 'Sustain.Continue.Prolong.Elaborate', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Support.Register', 'Open.Give.Fact.', 'React.Respond.Support.Register', 'React.Respond.Reply.Affirm', 'React.Respond.Develop.Enhance', 'React.Respond.Reply.Agree', 'React.Rejoinder.Re-challenge', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Monitor', 'Sustain.Continue.Prolong.Enhance', 'React.Respond.Develop.Extend', 'React.Rejoinder.Track.Clarify', 'Sustain.Continue.Prolong.Elaborate', 'Sustain.Continue.Prolong.Elaborate', 'React.Rejoinder.Track.Clarify', 'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Affirm', 'React.Respond.Develop.Extend', 'React.Respond.Support.Register', 'Sustain.Continue.Prolong.Enhance', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Extend', 'React.Rejoinder.Track.Check', 'Sustain.Continue.Prolong.Extend', 'React.Rejoinder.Track.Confirm', 'React.Rejoinder.Track.Clarify', 'React.Rejoinder.Track.Check', 'React.Respond.Reply.Affirm', 'Rejoinder.Counter', 'React.Respond.Develop.Extend', 'React.Rejoinder.Track.Probe', 'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Affirm', 'React.Respond.Support.Register', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Elaborate', 'Sustain.Continue.Prolong.Extend', 'React.Rejoinder.Re-challenge', 'Sustain.Continue.Prolong.Extend', 'Open.Give.Fact.', 'React.Rejoinder.Track.Confirm', 'React.Respond.Response.Resolve.', 'React.Respond.Develop.Extend', 'React.Rejoinder.Track.Probe', 'React.Rejoinder.Rebound', 'React.Respond.Reply.Affirm', 'React.Respond.Develop.Extend', 'React.Respond.Support.Register', 'React.Respond.Develop.Extend', 'React.Respond.Reply.Affirm', 'Sustain.Continue.Prolong.Elaborate', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Elaborate', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'Open.Give.Fact.', 'React.Respond.Support.Register', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Enhance', 'React.Respond.Develop.Extend', 'React.Respond.Develop.Elaborate', 'React.Respond.Reply.Affirm', 'Open.Give.Fact.', 'React.Respond.Reply.Affirm', 'React.Respond.Reply.Affirm', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Elaborate', 'React.Respond.Support.Register', 'React.Rejoinder.Track.Clarify', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Elaborate', 'React.Respond.Support.Register', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Monitor', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Elaborate', 'React.Rejoinder.Track.Check', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Support.Register', 'React.Rejoinder.Re-challenge', 'React.Respond.Reply.Affirm', 'React.Respond.Support.Register', 'Open.Give.Fact.', 'Sustain.Continue.Monitor', 'React.Rejoinder.Track.Clarify', 'React.Rejoinder.Track.Probe', 'React.Respond.Reply.Disagree', 'React.Rejoinder.Re-challenge', 'React.Respond.Reply.Disagree', 'React.Respond.Support.Register', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Support.Register', 'React.Respond.Develop.Enhance', 'React.Rejoinder.Track.Clarify', 'React.Rejoinder.Re-challenge', 'React.Respond.', 'React.Respond.Develop.Extend', 'React.Respond.Support.Register', 'React.Respond.Develop.Extend', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Enhance', 'React.Rejoinder.Track.Check', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'React.Respond.Support.Register', 'React.Respond.Develop.Elaborate', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Support.Register', 'Open.Demand.Fact.', 'React.Respond.Reply.Affirm', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Elaborate', 'React.Respond.Reply.Affirm', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Elaborate', 'Sustain.Continue.Prolong.Elaborate', 'React.Respond.Reply.Affirm', 'React.Respond.Support.Register', 'Sustain.Continue.Prolong.Elaborate', 'React.Respond.Support.Register', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Extend', 'React.Rejoinder.Track.Check', 'React.Respond.', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Reply.Affirm', 'React.Respond.Develop.Extend', 'React.Respond.Reply.Agree', 'React.Rejoinder.Re-challenge', 'React.Rejoinder.Re-challenge', 'React.Respond.Reply.Disawow', 'React.Rejoinder.Track.Confirm', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Elaborate', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Support.Register', 'Sustain.Continue.Prolong.Extend', 'React.Rejoinder.Track.Check', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Elaborate', 'Sustain.Continue.Prolong.Extend', 'React.Respond.Support.Register', 'Sustain.Continue.Prolong.Elaborate', 'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Affirm', 'React.Respond.Support.Register', 'React.Respond.Support.Register', 'React.Respond.Develop.Extend', 'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Affirm', 'React.Respond.Reply.Disawow', 'React.Respond.Develop.Enhance', 'React.Respond.Reply.Affirm', 'React.Respond.Develop.Enhance', 'Sustain.Continue.Monitor', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'React.Respond.Develop.Enhance', 'React.Respond.Develop.', 'React.Rejoinder.Re-challenge', 'Sustain.Continue.Prolong.Enhance', 'React.Respond.Support.Register', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Enhance', 'Open.Give.Fact.', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'React.Respond.Develop.Elaborate', 'React.Respond.Support.Register', 'React.Respond.Reply.Agree', 'React.Respond.Reply.Affirm', 'React.Respond.Reply.Affirm', 'React.Respond.Develop.Elaborate', 'React.Respond.Support.Register', 'React.Respond.Develop.Extend', 'Sustain.Continue.Monitor', 'Sustain.Continue.Prolong.Extend', 'React.Rejoinder.Track.Probe', 'React.Rejoinder.Track.Check', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Monitor', 'React.Respond.Support.Register', 'Sustain.Continue.Prolong.Enhance', 'React.Respond.Develop.Extend', 'React.Respond.Develop.Enhance', 'React.Respond.Develop.Elaborate', 'React.Respond.Develop.Extend', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Enhance', 'React.Respond.Develop.Extend', 'React.Respond.Reply.Affirm', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Elaborate', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Elaborate', 'React.Rejoinder.Re-challenge', 'React.Respond.', 'React.Respond.Develop.Enhance', 'React.Rejoinder.Re-challenge', 'React.Respond.Reply.Affirm', 'React.Rejoinder.Track.Confirm', 'Sustain.Continue.Prolong.Elaborate', 'React.Respond.Develop.Extend', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Elaborate', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Extend', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Elaborate', 'React.Respond.Develop.Enhance', 'React.Respond.Support.Register', 'React.Rejoinder.Track.Confirm', 'React.Respond.Reply.Disagree', 'Sustain.Continue.Prolong.Enhance', 'Sustain.Continue.Prolong.Elaborate', 'React.Respond.Develop.Extend', 'React.Respond.Reply.Agree', 'React.Respond.Develop.Enhance', 'React.Respond.Develop.Enhance', 'Open.Give.Fact.']

for a,b in zip(y_true_tests, y_pred):
    assert a==b, y_pred
