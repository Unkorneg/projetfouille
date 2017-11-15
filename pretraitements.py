import nltk
import re
import treetaggerwrapper
import string

stopwords = ['au', 'aux', 'avec', 'ce', 'ces', 'dans', 'de', 'des', 'du', 'elle', 'en', 'et', 'eux', 'il', 'je', 'la', 'le', 'leur', 'lui', 'ma', 'me', 'même', 'mes', 'moi', 'mon', 'nos', 'notre', 'nous', 'on', 'ou', 'par', 'pour', 'qu', 'que', 'qui', 'sa', 'se', 'ses', 'son', 'sur', 'ta', 'te', 'tes', 'toi', 'ton', 'tu', 'un', 'une', 'vos', 'votre', 'vous', 'c', 'd', 'j', 'l', 'à', 'm', 'n', 's', 't', 'y', "n'est", 'été', 'étée', 'étées', 'étés', 'étant', 'étante', 'étants', 'étantes', 'suis', 'es', 'est', 'sommes', 'êtes', 'sont', 'serai', 'seras', 'sera', 'serons', 'serez', 'seront', 'serais', 'serait', 'serions',
             'seriez', 'seraient', 'étais', 'était', 'étions', 'étiez', 'étaient', 'fus', 'fut', 'fûmes', 'fûtes', 'furent', 'sois', 'soit', 'soyons', 'soyez', 'soient', 'fusse', 'fusses', 'fût', 'fussions', 'fussiez', 'fussent', 'ayant', 'ayante', 'ayantes', 'ayants', 'eu', 'eue', 'eues', 'eus', 'ai', 'as', 'avons', 'avez', 'ont', 'aurai', 'auras', 'aura', 'aurons', 'aurez', 'auront', 'aurais', 'aurait', 'aurions', 'auriez', 'auraient', 'avais', 'avait', 'avions', 'aviez', 'avaient', 'eut', 'eûmes', 'eûtes', 'eurent', 'aie', 'aies', 'ait', 'ayons', 'ayez', 'aient', 'eusse', 'eusses', 'eût', 'eussions', 'eussiez', 'eussent', 'être', 'avoir']
tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr')


def removeURL(text):
    return re.sub(r'http\S+', '', text)


def removeRallongements(text):
    return re.sub(r'(.)\1+', r'\1\1', text)


def tokenize(text):
    tok = nltk.wordpunct_tokenize(text)
    # tok = nltk.word_tokenize(text, language='french')
    return [w.lower() for w in tok]


def removeStopwords(tokens):
    return [w for w in tokens if w not in stopwords]


def removePunctuation(tokens):
    return [w for w in tokens if w not in (string.punctuation + '«»')]


def lemmatise(text):
    res = [tag[2].lower() for tag in treetaggerwrapper.make_tags(
        tagger.tag_text(text), exclude_nottags=True)]
    return [w for w in res if w not in (string.punctuation + '«»')]
    
## TEST ##

# text = "DSK n'est pas un libertiiiiin mais un homme brutal http://bit.ly/HqgvN6 A lire sur @courrierinter"

# text = removeURL(text)
# text = removeRallongements(text)

# tok = lemmatise(text) #lemmatisation + tokenization
# tok = tokenize(text)

# tok = removeStopwords(tok)
# tok = removePunctuation(tok)
# print (tok)
