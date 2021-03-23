'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
let's start transformer!
1. dataset from wmt 2014 English-German or newstest2013 for dev
2. tokenize them
3. make transformer model
4. train and evaluate model

https://nlp.gluon.ai/examples/machine_translation/transformer.html
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import gluonnlp as nlp
from mxnet import gluon

from nmt import translation, bleu
import preprocess as pp
import hyperparameters as hparams

wmt_data_test_with_len, wmt_transformer_model, wmt_src_vocab, wmt_tgt_vocab, wmt_test_tgt_sentences = pp.preprocess()

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
create the sampler and dataloader
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
wmt_test_batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(pad_val=0),
                                               nlp.data.batchify.Pad(pad_val=0),
                                               nlp.data.batchify.Stack(dtype='float32'),
                                               nlp.data.batchify.Stack(dtype='float32'),
                                               nlp.data.batchify.Stack())
print(wmt_test_batchify_fn)

wmt_bucket_scheme = nlp.data.ExpWidthBucket(bucket_len_step=1.2)

wmt_test_batch_sampler = nlp.data.FixedBucketSampler(lengths=wmt_data_test_with_len.transform(lambda src, tgt, src_len, tgt_len, idx: tgt_len),
                                                     use_average_length=True,
                                                     bucket_scheme=wmt_bucket_scheme,
                                                     batch_size=256)
print(wmt_test_batch_sampler.stats())

wmt_test_data_loader = gluon.data.DataLoader(
    wmt_data_test_with_len,
    batch_sampler=wmt_test_batch_sampler,
    batchify_fn=wmt_test_batchify_fn,
    num_workers=0)
print(len(wmt_test_data_loader))

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
evaluating the transformer
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
wmt_translator = translation.BeamSearchTranslator(
    model=wmt_transformer_model,
    beam_size=hparams.beam_size,
    scorer=nlp.model.BeamSearchScorer(alpha=hparams.lp_alpha, K=hparams.lp_k),
    max_length=200)

import time
from nmt import utils

eval_start_time = time.time()

wmt_test_loss_function = nlp.loss.MaskedSoftmaxCELoss()
wmt_test_loss_function.hybridize()

wmt_detokenizer = nlp.data.SacreMosesDetokenizer()

wmt_test_loss, wmt_test_translation_out = utils.evaluate(wmt_transformer_model,
                                                         wmt_test_data_loader,
                                                         wmt_test_loss_function,
                                                         wmt_translator,
                                                         wmt_tgt_vocab,
                                                         wmt_detokenizer,
                                                         pp.ctx)

wmt_test_bleu_score, _, _, _, _ = bleu.compute_bleu([wmt_test_tgt_sentences],
                                                    wmt_test_translation_out,
                                                    tokenized=False,
                                                    tokenizer=hparams.bleu,
                                                    split_compound_word=False,
                                                    bpe=False)

print('WMT14 EN-DE SOTA model test loss: %.2f; test bleu score: %.2f; time cost %.2fs'
      %(wmt_test_loss, wmt_test_bleu_score * 100, (time.time() - eval_start_time)))


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
print sample translations
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
print('Sample translations:')
num_pairs = 3

for i in range(num_pairs):
    print('EN:')
    print(wmt_test_text[i][0])
    print('DE-Candidate:')
    print(wmt_test_translation_out[i])
    print('DE-Reference:')
    print(wmt_test_tgt_sentences[i])
    print('========')


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
translation inference
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
print('Translate the following English sentence into German:')

sample_src_seq = 'We love language .'

print('[\'' + sample_src_seq + '\']')

sample_tgt_seq = utils.translate(wmt_translator,
                                 sample_src_seq,
                                 wmt_src_vocab,
                                 wmt_tgt_vocab,
                                 wmt_detokenizer,
                                 pp.ctx)

print('The German translation is:')
print(sample_tgt_seq)