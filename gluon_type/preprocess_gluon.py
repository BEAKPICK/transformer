import warnings
warnings.filterwarnings('ignore')   # if you remove it, warning of the corrupt from index difference may come out.

import mxnet as mx
import gluonnlp as nlp
import hyperparameters_gluon as hparams

ctx = mx.cpu()

def get_length_index_fn():
    global idx
    idx = 0
    def transform(src, tgt):
        global idx
        result = (src, tgt, len(src), len(tgt), idx)
        idx += 1
        return result
    return transform


def preprocess():

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    load pretrained transformer and vocabularies
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    wmt_model_name = 'transformer_en_de_512'
    wmt_model_model, wmt_src_vocab, wmt_tgt_vocab = nlp.model.get_model(wmt_model_name,
                                                                        dataset_name='WMT2014',
                                                                        pretrained=True,
                                                                        ctx=ctx)
    print(len(wmt_src_vocab), len(wmt_tgt_vocab))

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    download byte-pair encoded wmt14 english-german with gluonnlp module
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    wmt_data_test = nlp.data.WMT2014BPE('newstest2014',
                                        src_lang=hparams.src_lang,
                                        tgt_lang=hparams.tgt_lang)
    print('Source language %s, Target language %s' % (hparams.src_lang, hparams.tgt_lang))
    print('Sample BPE tokens: "{}"'.format(wmt_data_test[0]))

    # for downloading non-bpe data, you can use this.
    wmt_test_text = nlp.data.WMT2014('newstest2014',
                                     src_lang=hparams.src_lang,
                                     tgt_lang=hparams.tgt_lang)
    print('Sample raw text: "{}"'.format(wmt_test_text[0]))

    wmt_test_tgt_sentences = wmt_test_text.transform(lambda src, tgt: tgt)
    print('Sample target sentence: "{}"'.format(wmt_test_tgt_sentences[0]))

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    map the sentences' word tokens to ids based on the vocabulary we load
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    import dataprocessor_gluon

    wmt_transform_fn = dataprocessor_gluon.TrainValDataTransform(wmt_src_vocab, wmt_tgt_vocab)
    # now you can custom your model through this preprocessed data
    wmt_dataset_processed = wmt_data_test.transform(wmt_transform_fn, lazy=False)
    print(*wmt_dataset_processed[0], sep='\n')

    wmt_data_test_with_len = wmt_dataset_processed.transform(get_length_index_fn(), lazy=False)

    wmt_test_tgt_sentences = wmt_test_text.transform(lambda src, tgt: tgt)

    return wmt_data_test_with_len, wmt_model_model, wmt_src_vocab, wmt_tgt_vocab, wmt_test_tgt_sentences

if __name__ == '__main__':
    print(preprocess())