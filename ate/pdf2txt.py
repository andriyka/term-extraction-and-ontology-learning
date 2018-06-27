import ate
import ConfigParser
import nltk

from unidecode import unidecode
from os import listdir
from os.path import isfile, join

config = ConfigParser.ConfigParser()
config.readfp(open('configuration/config.ini'))

pdf_dir=config.get('main', 'pdf_dir')
txt_dir=config.get('main', 'txt_dir')

ligatures = {0xFB00: u'ff', 0xFB01: u'fi', 0xA733: u'aa',
            0x00E6: u'ae', 0xA735: u'ao', 0xA737: u'au', 0xA739: u'av',
           0x1F670: u'et', 0xFB03: u'ffi', 0xFB04: u'ffl', 0xFB02: u'fl',
           0x0153: u'oe', 0xA74F: u'oo', 0x1E9E: u'fs', 0x00DF: u'fz', 0xFB06: u'st',
           0xFB05: u'ft', 0x1D6B: u'ue'}

pdf_files=sorted([ (pdf_dir, f) for f in listdir(pdf_dir) if isfile(join(pdf_dir, f)) and f.lower().endswith(".pdf")])

# def extract_sentences(line):
#     sentences = line.split('.')
#     reminder = sentences[-1] if sentences[-1][-1] == '.' else None
#     return sentences, reminder
#
# def preprocess_text(input_str):
#     lines = input_str.split('\n')
#     sentences = []
#     r_prev = None
#     for line in lines:
#         s, r = extract_sentences(line)
#         if r_prev:
#             s[0] = r_prev + s[0]
#             r
#         sentences.append(s)
#         if r:

def preprocess(file_str):
    cleaned = list(map(lambda x: x.replace('-\n', '').replace('\n', ' ').replace(u'', 'fi'), nltk.sent_tokenize(file_str)))
    return '\n'.join(cleaned)


print pdf_files
for f in pdf_files:
    fpath_out=join(txt_dir, f[1][:-4]+'.txt')
    #print pdf_dir,'/',f[1],"=>", fpath_out
    ftxt = open(fpath_out,'w')
    try:
        dirty = ate.pdf_to_text_textract(join(f[0], f[1])).replace("\n"," ")
        dirty = unidecode(unicode(dirty, encoding="utf-8"))
        clean = preprocess(dirty)
        ftxt.write(clean)
    except TypeError as e:
        print(e)
        print "error reading file "+ fpath_out
    
    #ftxt.write(ate.pdf_to_text_pypdf(join(f[0], f[1])).replace("\n"," "))
    ftxt.close()



