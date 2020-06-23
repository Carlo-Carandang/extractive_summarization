# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 19:35:10 2019

@author: carandangc
"""

import os
import sys
import json
import nltk
import numpy
from IPython.display import IFrame
from IPython.core.display import display

# load text here
text = "N. is a 19 year old single caucasian female, who is attending university and works part-time as a cashier for a pizza restaurant. She has a history of recurrent depressive episodes, which starts in the late fall/early winter months, and remits during the spring. She uses light therapy every morning for 30 minutes, as without it, she would be unable to function. Although her family doctor has recommended that she consider antidepressant medication treatment, N. has adamantly refused, stating that it is unnatural, as she tries to ingest healthy and natural foods. When she becomes depressed, she has symptoms of low moods, anhedonia, excessive fatigue, excessive sleep, increased appetite, carbohydrate cravings, and weight gain. Her family history is significant for her mother suffering from bipolar disorder, and who dies in a car accident when N. was only 12 years old. It is highly suspected that the car accident was a suicide. N. has refused to see a therapist, stating she does not need psychological help, and that she can deal with her mother’s death on her own. After the death of her mother, she and her brother were sent to live with their estranged father and step-mother, and their infant daughter, N.’s half-sister. She is the oldest of 3 siblings. Suffice it to say, N. did not get along well with her father and step-mother, and was jealous of all the attention her infant sister got from the family. When N. became depressed in the winter months, she locked herself in her room, and would sleep constantly, not contributing to the chores the family needed. When she was not sleeping, she would leave and spend days with her friends, without any check-ins with her parents. Eventually,she ran away from home after her graduation from high school, and was not found until her father shamed her on Facebook that his daughter was missing and is now a runaway. Humiliated, N. contacted her father and assured him she was safe. She eventually enrolled in university, and only has cursory contact with her father and siblings. Her first year in university was difficult, and she almost failed her classes due to sleeping too much when her depression would worsen like clockwork in the fall and winter. When spring would come around, she would have increased energy and could work for 12 hours a day and go out with her friends at night with only minimal sleep. She spends her summer doing lots of outdoor activities, as she likes to soak in the sun, and dreads each day after June 21st, when the sun exposure would decrease little by little with each passing day. Diagnosis: Major depressive disorder, recurrent, in partial remission, with seasonal pattern. Rule-out bipolar disorder, given the history of hypomanic symptoms, and family history of bipolar disorder. The patient had only a partial response to the current dose of light therapy. Treatment: 1) Will increase light therapy duration to 60 minutes every morning, from a 10,000 lux light box. Will need to monitor for any signs of mania, as light therapy may switch someone with an underlying bipolar disorder from depression to mania. 2) If no response in 1 to 2 weeks, then consider augmenting light therapy with cognitive behavioral therapy (CBT). 3) Recommend other helpful strategies for depression, including exercise, meditation, yoga, and outdoor time to increase sunlight exposure. 4) Treatment with antidepressant medications is a relative contraindication, as bipolar disorder has not been ruled out. 5) Return to clinic in 2 weeks for follow-up."

x = 1
clinic_notes = []
for e in range(x):
    clinic_notes.append({'title': 'Case ' + str(x),'content': text})

'''
out_file = os.path.join('clinic_notes.json')
f = open(out_file, 'w+')
f.write(json.dumps(clinic_notes, indent=1))
f.close()

print('Wrote output file to {0}'.format(f.name))

clinic_notes = "clinic_notes.json"

clinic_notes = json.loads(open(clinic_notes).read())
'''

N = 200  # Number of words to consider
CLUSTER_THRESHOLD = 5  # Distance between words to consider
TOP_SENTENCES = 5  # Number of sentences to return for a "top n" summary

stop_words = nltk.corpus.stopwords.words('english') + [
    '.',
    ',',
    '--',
    '\'s',
    '?',
    ')',
    '(',
    ':',
    '\'',
    '\'re',
    '"',
    '-',
    '}',
    '{',
    u'—',
    '>',
    '<',
    '...'
    ]

# Approach taken from "The Automatic Creation of Literature Abstracts" by H.P. Luhn
def _score_sentences(sentences, important_words):
    scores = []
    sentence_idx = 0

    for s in [nltk.tokenize.word_tokenize(s) for s in sentences]:

        word_idx = []

        # For each word in the word list...
        for w in important_words:
            try:
                # Compute an index for where any important words occur in the sentence.
                word_idx.append(s.index(w))
            except ValueError: # w not in this particular sentence
                pass

        word_idx.sort()

        # It is possible that some sentences may not contain any important words at all.
        if len(word_idx)== 0: continue

        # Using the word index, compute clusters by using a max distance threshold
        # for any two consecutive words.

        clusters = []
        cluster = [word_idx[0]]
        i = 1
        while i < len(word_idx):
            if word_idx[i] - word_idx[i - 1] < CLUSTER_THRESHOLD:
                cluster.append(word_idx[i])
            else:
                clusters.append(cluster[:])
                cluster = [word_idx[i]]
            i += 1
        clusters.append(cluster)

        # Score each cluster. The max score for any given cluster is the score 
        # for the sentence.

        max_cluster_score = 0
        
        for c in clusters:
            significant_words_in_cluster = len(c)
            # true clusters also contain insignificant words, so we get 
            # the total cluster length by checking the indices
            total_words_in_cluster = c[-1] - c[0] + 1
            score = 1.0 * significant_words_in_cluster**2 / total_words_in_cluster

            if score > max_cluster_score:
                max_cluster_score = score

        scores.append((sentence_idx, max_cluster_score))
        sentence_idx += 1

    return scores

def summarize(txt):
    sentences = [s for s in nltk.tokenize.sent_tokenize(txt)]
    normalized_sentences = [s.lower() for s in sentences]

    words = [w.lower() for sentence in normalized_sentences for w in
             nltk.tokenize.word_tokenize(sentence)]

    fdist = nltk.FreqDist(words)
    
    # Remove stopwords from fdist
    for sw in stop_words:
        del fdist[sw]

    top_n_words = [w[0] for w in fdist.most_common(N)]

    scored_sentences = _score_sentences(normalized_sentences, top_n_words)

    # Summarization Approach 1:
    # Filter out nonsignificant sentences by using the average score plus a
    # fraction of the std dev as a filter

    avg = numpy.mean([s[1] for s in scored_sentences])
    std = numpy.std([s[1] for s in scored_sentences])
    mean_scored = [(sent_idx, score) for (sent_idx, score) in scored_sentences
                   if score > avg + 0.5 * std]

    # Summarization Approach 2:
    # Another approach would be to return only the top N ranked sentences

    top_n_scored = sorted(scored_sentences, key=lambda s: s[1])[-TOP_SENTENCES:]
    top_n_scored = sorted(top_n_scored, key=lambda s: s[0])

    # Decorate the post object with summaries

    return dict(top_n_summary=[sentences[idx] for (idx, score) in top_n_scored],
                mean_scored_summary=[sentences[idx] for (idx, score) in mean_scored])

for note in clinic_notes: 
    note.update(summarize(note['content']))

    print(note['title'])
    print('=' * len(note['title']))
    print()
    print('Top N Summary')
    print('-------------')
    print(' '.join(note['top_n_summary']))
    print()
    print('Mean Scored Summary')
    print('-------------------')
    print(' '.join(note['mean_scored_summary']))
    print()
    
# Visualizing document summarization results with HTML output
HTML_TEMPLATE = """<html>
    <head>
        <title>{0}</title>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8"/>
    </head>
    <body>{1}</body>
</html>"""

for note in clinic_notes:
   
    # Uses previously defined summarize function.
    note.update(summarize(note['content']))

    # You could also store a version of the full post with key sentences marked up
    # for analysis with simple string replacement...

    for summary_type in ['top_n_summary', 'mean_scored_summary']:
        note[summary_type + '_marked_up'] = '<p>{0}</p>'.format(note['content'])
        
        for s in note[summary_type]:
            note[summary_type + '_marked_up'] = \
            note[summary_type + '_marked_up'].replace(s, '<strong>{0}</strong>'.format(s))

        filename = note['title'].replace("?", "") + '.summary.' + summary_type + '.html'
        
        f = open(os.path.join(filename), 'wb')
        html = HTML_TEMPLATE.format(note['title'] + ' Summary', note[summary_type + '_marked_up'])    
        f.write(html.encode('utf-8'))
        f.close()

        print("Data written to", f.name)

# Display any of these files with an inline frame. This displays the
# last file processed by using the last value of f.name...
print()
print("Displaying {0}:".format(f.name))
display(IFrame('files/{0}'.format(f.name), '100%', '600px'))