from sklearn.feature_extraction.text import TfidfVectorizer

vctr = TfidfVectorizer(ngram_range = (2,2),stop_words='english')

string1 = 'This is our second seminar of Advanced Analytics, we are learning about the text classification of emails'

print('Origin: \n' + string1 +'\n ')
print('Preprocess: \n', vctr.build_preprocessor()(string1))
print('Tokenize: \n', vctr.build_tokenizer()(string1))
print('Analyze: \n', vctr.build_analyzer()(string1))




