import ate
import ConfigParser
import nltk

from unidecode import unidecode
from os import listdir
from os.path import isfile, join

ligatures = {0xFB00: u'ff', 0xFB01: u'fi', 0xA733: u'aa',
            0x00E6: u'ae', 0xA735: u'ao', 0xA737: u'au', 0xA739: u'av',
           0x1F670: u'et', 0xFB03: u'ffi', 0xFB04: u'ffl', 0xFB02: u'fl',
           0x0153: u'oe', 0xA74F: u'oo', 0x1E9E: u'fs', 0x00DF: u'fz', 0xFB06: u'st',
           0xFB05: u'ft', 0x1D6B: u'ue'}


def preprocess(file_str):
    cleaned = list(map(lambda x: x.replace('-\n', '').replace('\n', ' ').replace(u'', 'fi'), nltk.sent_tokenize(file_str)))
    return '\n'.join(cleaned)


def convert_pdf2txt(inp_dir, outp_dir, verbose = False):
    pdf_files = sorted(
        [(inp_dir, f) for f in listdir(inp_dir) if isfile(join(inp_dir, f)) and f.lower().endswith(".pdf")])
    if verbose:
        print pdf_files

    for f in pdf_files:
        fpath_out = join(outp_dir, f[1][:-4] + '.txt')
        if verbose:
            print inp_dir,'/',f[1],"=>", fpath_out
        ftxt = open(fpath_out, 'w')
        try:
            dirty = ate.pdf_to_text_textract(join(f[0], f[1])).replace("\n", " ")
            dirty = unidecode(unicode(dirty, encoding="utf-8"))
            clean = preprocess(dirty)
            ftxt.write(clean)
        except TypeError as e:
            if verbose:
                print(e)
                print "error reading file " + fpath_out

        # ftxt.write(ate.pdf_to_text_pypdf(join(f[0], f[1])).replace("\n"," "))
        ftxt.close()
    print('Extracted to {}'.format(outp_dir))





