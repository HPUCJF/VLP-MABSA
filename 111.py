import json
import gradio as gr
import os

from datetime import datetime
import argparse
import json
from torch import optim
import torch
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
import random
from src.data.collation import Collator
from src.data.dataset import MVSA_Dataset, Twitter_Dataset
from src.data.tokenization_new import ConditionTokenizer
from src.model.config import MultiModalBartConfig
from src.model.MAESC_model import MultiModalBartModel_AESC
from src.model.model import MultiModalBartModelForPretrain
from src.training import fine_tune
from src.utils import Logger, save_training_data, load_training_data, setup_process, cleanup_process
from src.model.metrics import AESCSpanMetric
from src.model.generater import SequenceGeneratorModel
import src.eval_utils as eval_utils
import numpy as np
import torch.backends.cudnn as cudnn
import torch
import gradio as gr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        action='append',
                        nargs=2,
                        default=[('twitter15', './src/data/jsons/twitter15_info.json')],
                        metavar=('DATASET_NAME', 'DATASET_PATH'),

                        help='')
    # required

    parser.add_argument('--checkpoint_dir',

                        default='./',
                        type=str,
                        help='where to save the checkpoint')
    parser.add_argument('--bart_model',
                        default='bart-base',
                        type=str,
                        help='bart pretrain model')
    # path
    parser.add_argument(
        '--log_dir',
        default='15_aesc',
        type=str,
        help='path to output log files, not output to file if not specified')
    parser.add_argument('--model_config',
                        default='config/pretrain_base.json',
                        type=str,
                        help='path to load model config')
    parser.add_argument('--text_only',
                        default=False,
                        type=bool,
                        help='if only input the text')
    parser.add_argument('--checkpoint',
                        default='state_dict/52.92',
                        type=str,
                        help='name or path to load weights')
    parser.add_argument('--lr_decay_every',
                        default=4,
                        type=int,
                        help='lr_decay_every')
    parser.add_argument('--lr_decay_ratio',
                        default=0.8,
                        type=float,
                        help='lr_decay_ratio')
    # training and evaluation
    parser.add_argument('--epochs',
                        default=35,
                        type=int,
                        help='number of training epoch')
    parser.add_argument('--eval_every', default=1, type=int, help='eval_every')
    parser.add_argument('--lr', default=7e-5, type=float, help='learning rate')
    parser.add_argument('--num_beams',
                        default=4,
                        type=int,
                        help='level of beam search on validation')
    parser.add_argument(
        '--continue_training',
        action='store_true',
        help='continue training, load optimizer and epoch from checkpoint')
    parser.add_argument('--warmup', default=0.1, type=float, help='warmup')
    # dropout
    parser.add_argument(
        '--dropout',
        default=None,
        type=float,
        help=
        'dropout rate for the transformer. This overwrites the model config')
    parser.add_argument(
        '--classif_dropout',
        default=None,
        type=float,
        help=
        'dropout rate for the classification layers. This overwrites the model config'
    )
    parser.add_argument(
        '--attention_dropout',
        default=None,
        type=float,
        help=
        'dropout rate for the attention layers. This overwrites the model config'
    )
    parser.add_argument(
        '--activation_dropout',
        default=None,
        type=float,
        help=
        'dropout rate for the activation layers. This overwrites the model config'
    )

    # hardware and performance
    parser.add_argument('--grad_clip', default=5, type=float, help='grad_clip')
    parser.add_argument('--gpu_num',
                        default=1,
                        type=int,
                        help='number of GPUs in total')
    parser.add_argument('--cpu',
                        action='store_true',
                        help='if only use cpu to run the model')
    parser.add_argument('--amp',
                        action='store_true',
                        help='whether or not to use amp')
    parser.add_argument('--master_port',
                        type=str,
                        default='12355',
                        help='master port for DDP')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='training batch size')
    parser.add_argument('--seed', type=int, default=66, help='seed')
    parser.add_argument('--num_workers',
                        type=int,
                        default=0,
                        help='#workers for data loader')
    parser.add_argument('--max_len', type=int, default=10, help='max_len')
    parser.add_argument('--max_len_a',
                        type=float,
                        default=0.6,
                        help='max_len_a')

    parser.add_argument('--bart_init',
                        type=int,
                        default=1,
                        help='use bart_init or not')

    parser.add_argument('--check_info',
                        type=str,
                        default='',
                        help='check path to save')
    parser.add_argument('--is_check',
                        type=int,
                        default=0,
                        help='save the model or not')
    parser.add_argument('--task', type=str, default='', help='task type')
    args = parser.parse_args()

    if args.gpu_num != 1 and args.cpu:
        raise ValueError('--gpu_num are not allowed if --cpu is set to true')

    if args.checkpoint is None and args.model_config is None:
        raise ValueError(
            '--model_config and --checkpoint cannot be empty at the same time')

    return args


def find_sentence_position(text, sentence):
    words = text.split()
    target_words = sentence.split()
    target_length = len(target_words)

    for i in range(len(words) - target_length + 1):
        if words[i:i + target_length] == target_words:
            start_index = i
            end_index = i + target_length
            return start_index, end_index

    return None


def chatbot_interface(sentence, image_id, add_aspect, de_aspect):
    args = parse_args()
    if args.cpu:
        device = 'cpu'
        map_location = device
    else:
        device = torch.device("cuda:0")

    tokenizer = ConditionTokenizer(args=args)
    label_ids = list(tokenizer.mapping2id.values())  # mapping2id{'AESC': id, 'POS': id,'NEU': id,'NEG': id}
    senti_ids = list(tokenizer.senti2id.values())  # senti2id{'POS': id, 'NEU': id, 'NEG': id}
    # print(label_ids,senti_ids)[50276, 50277, 50278, 50279] [50277, 50278, 50279]

    # =========================== bart_config =============================
    if args.model_config is not None:  # config/pretrain_base.json
        bart_config = MultiModalBartConfig.from_dict(
            json.load(open(args.model_config)))  # 设置模型配置，如dropout等
        # print(bart_config.d_model,"**********************************************************************")
    else:
        bart_config = MultiModalBartConfig.from_pretrained(args.checkpoint)

    if args.dropout is not None:
        bart_config.dropout = args.dropout
    if args.attention_dropout is not None:
        bart_config.attention_dropout = args.attention_dropout
    if args.classif_dropout is not None:
        bart_config.classif_dropout = args.classif_dropout
    if args.activation_dropout is not None:
        bart_config.activation_dropout = args.activation_dropout

    # =========================== bart_config =============================

    bos_token_id = 0  # 因为是特殊符号
    eos_token_id = 1

    # =========================== .bin文件的利用，加载模型 ===============================================================

    if args.checkpoint:
        # pretrain_model = MultiModalBartModelForPretrain.from_pretrained(
        #     args.checkpoint,
        #     config=bart_config,
        #     bart_model=args.bart_model,
        #     tokenizer=tokenizer,
        #     label_ids=label_ids,
        #     senti_ids=senti_ids,
        #     args=args,
        #     error_on_mismatch=False)  # 加载预训练模型
        seq2seq_model = MultiModalBartModel_AESC(bart_config, args,
                                                 args.bart_model, tokenizer,
                                                 label_ids)
        # seq2seq_model.encoder.load_state_dict(
        #     pretrain_model.encoder.state_dict())
        # seq2seq_model.decoder.load_state_dict(
        #     pretrain_model.span_decoder.state_dict())
        model = SequenceGeneratorModel(seq2seq_model,
                                       bos_token_id=bos_token_id,
                                       eos_token_id=eos_token_id,
                                       max_length=args.max_len,
                                       max_len_a=args.max_len_a,
                                       num_beams=args.num_beams,
                                       do_sample=False,
                                       repetition_penalty=1,
                                       length_penalty=1.0,
                                       pad_token_id=eos_token_id,
                                       restricter=None)
        model.load_state_dict(torch.load("checkpoint/70.16"))
    # =========================== .bin文件的利用，加载模型 =============================================================================================
    else:
        seq2seq_model = MultiModalBartModel_AESC(bart_config, args,
                                                 args.bart_model, tokenizer,
                                                 label_ids)
        model = SequenceGeneratorModel(seq2seq_model,
                                       bos_token_id=bos_token_id,
                                       eos_token_id=eos_token_id,
                                       max_length=args.max_len,
                                       max_len_a=args.max_len_a,
                                       num_beams=args.num_beams,
                                       do_sample=False,
                                       repetition_penalty=1,
                                       length_penalty=1.0,
                                       pad_token_id=eos_token_id,
                                       restricter=None)
        # model = MultiModalBartModel_AESC(bart_config, args.bart_model,
        #                                  tokenizer, label_ids)
    model.to(device)

    collate_aesc = Collator(tokenizer,
                            mlm_enabled=False,
                            senti_enabled=False,
                            ae_enabled=False,
                            oe_enabled=False,
                            aesc_enabled=True,
                            anp_enabled=False,
                            text_only=args.text_only)



    callback = None
    metric = AESCSpanMetric(eos_token_id,
                            num_labels=len(label_ids),
                            conflict_id=-1)

    result = sentence.split(' ')
    dict = {}
    aspects = []
    words = add_aspect.split(',')
    for word in words:
        start, end = find_sentence_position(sentence, word)
        dict = {"from": start, "to": end, "polarity": "POS", "term": word.split()}
        aspects.append(dict)
    term = add_aspect.split(' ')
    data = {
        "words": result,
        "image_id": image_id,
        "aspects": aspects

    }
    x = []
    x.append(data)

    with open('test.json', 'w') as file:
        json.dump(x, file)
    test_dataset = Twitter_Dataset(args.dataset[0][1], split='test')

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             collate_fn=collate_aesc)
    res_test, pre, spans, labels = eval_utils.eval(args, model, test_loader, metric, device)

    word_bpes = [[tokenizer.begin_text_id]]
    for word in result:
        bpes = tokenizer._base_tokenizer.tokenize(word,
                                             add_prefix_space=True)
        bpes = tokenizer._base_tokenizer.convert_tokens_to_ids(bpes)
        word_bpes.append(bpes)
    word_bpes.append([tokenizer.end_text_id])  # [<<text>>_id , word_token_id , ..... <</text>>_id]

    lens = list(map(len, word_bpes))  # 将名为word_bpes的列表中的每个元素的长度计算出来，并将结果存储在一个新的列表lens中。
    cum_lens = np.cumsum(list(lens)).tolist()
    i = 5
    y = pre[0][0]
    y = y.tolist()
    print(y)
    aspec = []
    while(i<=len(y)):
        s = cum_lens.index(y[i-2]-6)
        end = cum_lens.index(y[i-1]-6)
        print(s,end)
        if(y[i]==3):
            p = "POS"
        if (y[i] == 4):
            p = "NEU"
        if y[i] == 5:
            p = "NEG"
        sub_list = result[s:end+1]  # 注意切片的结束索引是不包含的，所以需要写成7
        asp = ' '.join(map(str, sub_list))
        asp = (asp,p)
        aspec.append(asp)
        print(aspec)
        i = i+3
    print(pre)
    return aspec


iface = gr.Interface(
    fn=chatbot_interface,
    inputs=[gr.Textbox(lines=2, placeholder="Enter your query here..."), gr.Textbox(lines=2, placeholder="image_id"),
            gr.Textbox(lines=2, placeholder="enter the aspects"),
            gr.Textbox(lines=2, placeholder="enter the worry aspects")],
    outputs="text",
    title="Big Model Chatbot",
    description="Ask me anything!",
    examples=[
        [
            "RT @ OU Football : Practice one in the books Want more photos ? Follow the official Oklahoma football Facebook page ! # onlyONE",
            "74960.jpg", "Oklahoma,more photos"],
        ["Embattled Metro Councilman Dan Johnson to debate challenger John Witt , an independent . # Louisville", "553062.jpg","Metro,Dan Johnson,John Witt,# Louisville"],
        ["推荐一本好书"]
    ]
)

iface.launch()
