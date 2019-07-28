# -*- encoding:utf-8 -*-
"""
Author: wangqing
Date: 20190715
Version:2.0
Details：
Version1.0 实现了模型的训练，在1.0 的基础上做了以下改动：
1. 对train函数进行了修改
2. 增加了test函数
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hierarchy_model_new import Summarizer
import random
import os
import distributed
import signal
import torchsnooper


def multi_main(args):
    """ Spawns 1 process per GPU """
    nb_gpu = 2
    # 封装了python自带的multiprocessing模块，若存在shared_memory,则可以将其发送给其他进程
    # 下面的函数实现的是
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args, device_id, error_queue,), daemon=True))
        procs[i].start()
        print(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def run(args, device_id, error_queue):
    """ run process """
    args_gpu_ranks = [0, 1]
    world_size = 2
    try:
        gpu_rank = distributed.multi_init(device_id, world_size, args_gpu_ranks)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args_gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args_gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def save_model(step, state_dict, model_file):
    path = os.path.join(model_file, 'model_step_%d.pt' % step)
    torch.save(state_dict, path)


def train(args):
    """
    1. 首先load data，并将data送入模型中，得到

    2. 计算loss

    3. optimizer

    :param args: 从命令行传入的参数
    :return:
    """
    # torch.backends.cudnn.deterministic = True

    # if device_id >= 0:
    # torch.cuda.set_device(device_id)

    lr = args.learning_rate
    iterations = args.iterations

    # 首先取出训练数据
    train_file = args.train_file
    train_data = torch.load(train_file)

    """
    batch_size应该等于1，因为bert的output的shap为（batch_size, sequence_length, hidden_size)
    我们采用的不同段落进行输入，无法使用一个batch_size中多个segment进行训练。
    因此
    """
    model = Summarizer(args)
    # model.to(args.device)
    # optimizer = optim.Adagrad(model.parameters(), lr=lr)
    optimizer = model.optimizer
    for iter in range(iterations):
        print("This is {} iteration **************".format(iter + 1))
        # random.shuffle(train_data)
        whole_loss = 0
        for i in range(len(train_data)):
            if i != 1:
                continue
            model.loss = 0
            seg_dict = train_data[i]
            contrast_list = model(seg_dict)
            # 取出contrast_list中的每一个item，进行Loss计算，并进行Loss的更新
            for j in range(len(contrast_list)):
                contrast_dict = contrast_list[j]
                title_index = contrast_dict['gen']
                tgt_tensor = contrast_dict['tgt']
                # print(seg_dict['segment'][j]['tgt_txt'])

                # calculate loss and update
                # title_index.requires_grad = True
                print(title_index)
                print(tgt_tensor)
                para_loss = model.loss_func(title_index, tgt_tensor)

                # 计算加和
                model.loss += para_loss
                # optimizer.zero_grad()
                # para_loss.backward()
                # optimizer.step()

                # print(para_loss.grad)

                # model.loss += para_loss

            # optimizer
            optimizer.zero_grad()
            model.loss.backward()
            optimizer.step()

            # print(model.loss)

            print("The loss of {}th data is {}".format(i + 1, model.loss))
            whole_loss += model.loss
            # checkpoint，根据iteration来计算
        print("Finish train whole dataset")
        print("{} iteration loss is {}".format(iter + 1, whole_loss))

        # checkpoint，每1000 iteration保存一次
        iteration = iter + 1
        if (iteration % 100) == 0:
            state_dict = model.state_dict()
            save_model(iteration, state_dict, args.model_file)


def val(args):
    """
    validate model
    :param args:
    :return:
    """
    print("valdidation function")


def test(args):
    """
    具体做法：
    1. load checkpoint
    2. 载入测试数据，为每个段落产生标题
    3. 计算loss
    4. 将标题ID 转换成对应的文字
    但是这个版本
    :param args:
    :return:
    """
    # load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model = Summarizer(args)
    model.load_state_dict(checkpoint)
    model.eval()

    # load test data
    test_data = torch.load(args.test_file)
    with torch.no_grad:
        for i in range(len(test_data)):
            seg_dict = test_data[i]
            contrast_list = model(seg_dict)
            # 取出contrast_list中的每一个item，进行Loss计算
            for j in range(len(contrast_list)):
                contrast_dict = contrast_list[j]
                title_index = contrast_dict['gen']
                tgt_tensor = contrast_dict['tgt']

                model.loss = model.loss_func(title_index, tgt_tensor)
                print("The loss in test phase is {}".format(model.loss))
                # 将生成的title和目标title进行打印
                gen_title = model.convert_idx_to_word(title_index)
                print("Generate title is {}".format(gen_title))
                raw_tgt = seg_dict['segment'][j]['tgt_txt']
                print("The goal title is {}".format(raw_tgt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", default=1, type=int)
    parser.add_argument("-iterations", default=20, type=int)
    parser.add_argument("-train_file", default="./preprocess/multi_heading.pt")
    parser.add_argument("-test_file", default="./preprocess/multi_test.pt")
    parser.add_argument("-vocab_file", default="./preprocess/vocab.pt")
    parser.add_argument("-learning_rate", default=0.1)
    parser.add_argument("-mode", default="train")
    parser.add_argument("-model_file", default="./model_ckpt")
    parser.add_argument("-checkpoint", default="./model_ckpt/model_step_5000.ckpt")
    parser.add_argument("-predict_config", default="./bert_config.json")
    parser.add_argument("-device", default="cuda")
    args = parser.parse_args()

    mode = args.mode
    if mode == "train":
        train(args)
        # multi_main(args)
    elif mode == "val":
        val(args)
    elif mode == "test":
        test(args)
