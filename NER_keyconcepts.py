# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:58:23 2019

@author: carandangc
"""

import nltk
import json
import os
from IPython.display import IFrame
from IPython.core.display import display

clinic_notes = "clinic_notes.json"

def extract_interactions(txt):
    sentences = nltk.tokenize.sent_tokenize(txt)
    tokens = [nltk.tokenize.word_tokenize(s) for s in sentences]
    pos_tagged_tokens = [nltk.pos_tag(t) for t in tokens]

    entity_interactions = []
    for sentence in pos_tagged_tokens:

        all_entity_chunks = []
        previous_pos = None
        current_entity_chunk = []

        for (token, pos) in sentence:

            if pos == previous_pos and pos.startswith('NN'):
                current_entity_chunk.append(token)
            elif pos.startswith('NN'):
                if current_entity_chunk != []:
                    all_entity_chunks.append((' '.join(current_entity_chunk),
                            pos))
                current_entity_chunk = [token]

            previous_pos = pos

        if len(all_entity_chunks) > 1:
            entity_interactions.append(all_entity_chunks)
        else:
            entity_interactions.append([])

    assert len(entity_interactions) == len(sentences)

    return dict(entity_interactions=entity_interactions,
                sentences=sentences)

clinic_notes = json.loads(open(clinic_notes).read())

# Display selected interactions on a per-sentence basis

for note in clinic_notes:

    note.update(extract_interactions(note['content']))

    print(note['title'])
    print('-' * len(note['title']))
    for interactions in note['entity_interactions']:
        print('; '.join([i[0] for i in interactions]))
    print()

# Visualizing interactions between entities with HTML output 
HTML_TEMPLATE = """<html>
    <head>
        <title>{0}</title>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8"/>
    </head>
    <body>{1}</body>
</html>"""

for note in clinic_notes:

    note.update(extract_interactions(note['content']))

    # Display output as markup with entities presented in bold text

    note['markup'] = []

    for sentence_idx in range(len(note['sentences'])):

        s = note['sentences'][sentence_idx]
        for (term, _) in note['entity_interactions'][sentence_idx]:
            s = s.replace(term, '<strong>{0}</strong>'.format(term))

        note['markup'] += [s] 
            
    filename = note['title'].replace("?", "") + '.entity_interactions.html'
    f = open(os.path.join(filename), 'wb')
    html = HTML_TEMPLATE.format(note['title'] + ' Interactions', ' '.join(note['markup']))
    f.write(html.encode('utf-8'))
    f.close()

    print('Data written to', f.name)
    
    # Display any of these files with an inline frame. This displays the
    # last file processed by using the last value of f.name...
    
    print('Displaying {0}:'.format(f.name))
    display(IFrame('files/{0}'.format(f.name), '100%', '600px'))