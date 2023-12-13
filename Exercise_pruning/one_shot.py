import argparse
import json
import time
from datetime import datetime
import warnings
import os
warnings.filterwarnings("ignore")

from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import random
from logger import SummaryLogger

from utils import *
import models
import pruning
import config
from data import DataLoader


# parser = argparse.ArgumentParser(description='Pharaphaser training')
# parser.add_argument('--text', default='log.txt', type=str)
# parser.add_argument('--exp_name', default='cifar10/FT', type=str)
# parser.add_argument('--log_time', default='1', type=str)
# parser.add_argument('--lr', default='0.1', type=float)
# parser.add_argument('--resume_epoch', default='0', type=int)
# parser.add_argument('--epoch', default='163', type=int)
# parser.add_argument('--decay_epoch', default=[82, 123], nargs="*", type=int)
# parser.add_argument('--w_decay', default='5e-4', type=float)
# parser.add_argument('--cu_num', default='0', type=str)
# parser.add_argument('--seed', default='1', type=str)
# parser.add_argument('--load_pretrained_teacher', default='trained/Teacher.pth', type=str)
# parser.add_argument('--load_pretrained_paraphraser', default='trained/Paraphraser.pth', type=str)
# parser.add_argument('--save_model', default='ckpt.t7', type=str)
# parser.add_argument('--rate', type=float, default=0.5, help='The paraphrase rate k')
# parser.add_argument('--beta', type=int, default=500)


# #Other parameters
# DEVICE = torch.device("cuda")
# RESUME_EPOCH = args.resume_epoch
# DECAY_EPOCH = args.decay_epoch
# DECAY_EPOCH = [ep - RESUME_EPOCH for ep in DECAY_EPOCH]
# FINAL_EPOCH = args.epoch
# EXPERIMENT_NAME = args.exp_name
# RATE = args.rate
# BETA = args.beta

cudnn.deterministic = True
cudnn.benchmark = False


#### random Seed ####
num = random.randint(1, 10000)
random.seed(num)
torch.manual_seed(num)
#####################

# Loss and Optimizer
criterion_CE = nn.CrossEntropyLoss()
criterion = nn.L1Loss()
criterion_kl = KLLoss()


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss


def hyperparam():
    args = config.config()
    return args

def main(args):

    global arch_name_teacher, arch_name_student
    if args.cuda and not torch.cuda.is_available():
        raise Exception('No GPU found, please run without --cuda')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cu_num

    # set model name
    arch_name_teacher, arch_name_student = set_arch_name(args, kd=1)

    print('\n=> creating model \'{}\', \'{}\''.format(arch_name_teacher, arch_name_student))
    
    # Load pretrained models
    pruner = pruning.__dict__[args.pruner] # default : dcil
    Teacher, image_size = pruning.models.__dict__[args.arch_teacher](data=args.dataset, num_layers=args.layers,
                                                width_mult=args.width_mult,
                                                depth_mult=args.depth_mult,
                                                model_mult=args.model_mult,
                                                mnn=pruner.mnn)


    Student, image_size = pruning.models.__dict__[args.arch_student](data=args.dataset, num_layers=args.layers,
                                                width_mult=args.width_mult,
                                                depth_mult=args.depth_mult,
                                                model_mult=args.model_mult,
                                                mnn=pruner.mnn)

    assert Teacher is not None, 'Unavailable Teacher model parameters!! exit...\n'
    assert Student is not None, 'Unavailable Student model parameters!! exit...\n'



    optimizer = optim.SGD(Student.parameters(), lr=args.lr if args.warmup_lr_epoch == 0 else args.warmup_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    optimizer_t = optim.SGD(Teacher.parameters(), lr=args.lr if args.warmup_lr_epoch == 0 else args.warmup_lr, momentum=args.momentum, weight_decay=args.weight_deca, nesterov=args.nesterov)

    scheduler = set_scheduler(optimizer, args)
    scheduler_t = set_scheduler(optimizer_t, args)

    # set multi-gpu
    if args.cuda:

        Teacher = Teacher.cuda()
        Student = Student.cuda()
        criterion = criterion.cuda()
        Teacher = nn.DataParallel(Teacher, device_ids=args.gpuids,
                                output_device=args.gpuids[0])
        Student = nn.DataParallel(Student, device_ids=args.gpuids,
                                output_device=args.gpuids[0])
        cudnn.benchmark = True
        
        # for distillation


    # Dataset loading
    print('==> Load data..')
    start_time = time.time()
    train_loader, val_loader = DataLoader(args.batch_size, args.dataset,
                                          args.workers, args.datapath, image_size,
                                          args.cuda)

    elapsed_time = time.time() - start_time
    print('===> Data loading time: {:,}m {:.2f}s'.format(
        int(elapsed_time//60), elapsed_time%60))
    print('===> Data loaded..')

    arch_name_teacher, arch_name_student

    # load a pre-trained model
    if args.load is not None:
        ckpt_file = pathlib.Path('checkpoint') / arch_name / args.dataset / args.load
        assert isfile(ckpt_file), '==> no checkpoint found \"{}\"'.format(args.load)

        print('==> Loading Checkpoint \'{}\''.format(args.load))
        # check pruning or quantization or transfer
        strict = False if args.prune else True
        # load a checkpoint
        checkpoint = load_model(model, ckpt_file, main_gpu=args.gpuids[0], use_cuda=args.cuda, strict=strict)
        print('==> Loaded Checkpoint \'{}\''.format(args.load))

    # for training
    if args.run_type == 'train':
        # init parameters
        start_epoch = 0
        global iterations
        iterations = 0
        best_acc1 = 0.0
        train_time = 0.0
        validate_time = 0.0

        os.makedirs('./results', exist_ok=True)
        file_train_acc = os.path.join('results', '{}.txt'.format('_'.join(['train', arch_name, args.dataset, args.save.split('.pth')[0]])))
        file_test_acc = os.path.join('results', '{}.txt'.format('_'.join(['test', arch_name, args.dataset, args.save.split('.pth')[0]])))

        epochs = args.target_epoch + 75
        # for epoch in range(start_epoch, args.epochs):
        for epoch in range(start_epoch, epochs):

            print('\n==> {}/{} training'.format(
                    arch_name, args.dataset))
            print('==> Epoch: {}, lr = {}'.format(
                epoch, optimizer.param_groups[0]["lr"]))

            # train for one epoch
            print('===> [ Training ]')
            start_time = time.time()
            acc1_train, acc5_train = train(args, train_loader,
                epoch=epoch, model=model,
                criterion=criterion, optimizer=optimizer, scheduler=scheduler)

            elapsed_time = time.time() - start_time
            train_time += elapsed_time
            print('====> {:.2f} seconds to train this epoch\n'.format(
                elapsed_time))

            # evaluate on validation set
            print('===> [ Validation ]')
            start_time = time.time()
            acc1_valid, acc5_valid = validate(args, val_loader,
                epoch=epoch, model=model, criterion=criterion)
            elapsed_time = time.time() - start_time
            validate_time += elapsed_time
            print('====> {:.2f} seconds to validate this epoch'.format(
                elapsed_time))



            tt1, tt = validate_t(args, val_loader,
                epoch=epoch, model=model, criterion=criterion)

            acc1_train = round(acc1_train.item(), 4)
            acc5_train = round(acc5_train.item(), 4)
            acc1_valid = round(acc1_valid.item(), 4)
            acc5_valid = round(acc5_valid.item(), 4)

            open(file_train_acc, 'a').write(str(acc1_train)+'\n')
            open(file_test_acc, 'a').write(str(acc1_valid)+'\n')

            # remember best Acc@1 and save checkpoint and summary csv file
            state = model.state_dict()
            summary = [epoch, acc1_train, acc5_train, acc1_valid, acc5_valid]

            is_best = acc1_valid > best_acc1
            best_acc1 = max(acc1_valid, best_acc1)
            if is_best:
                save_model(arch_name, args.dataset, state, args.save)
            save_summary(arch_name, args.dataset, args.save.split('.pth')[0], summary)

            # for pruning
            if args.prune:
                num_total, num_zero, sparsity = pruning.cal_sparsity(model)
                print('\n====> sparsity: {:.2f}% || num_zero/num_total: {}/{}'.format(sparsity, num_zero, num_total))

            # end of one epoch
            print()

        # calculate the total training time 
        # avg_train_time = train_time / (args.epochs - start_epoch)
        # avg_valid_time = validate_time / (args.epochs - start_epoch)
        avg_train_time = train_time / (epochs - start_epoch)
        avg_valid_time = validate_time / (epochs - start_epoch)
        total_train_time = train_time + validate_time
        print('====> average training time each epoch: {:,}m {:.2f}s'.format(
            int(avg_train_time//60), avg_train_time%60))
        print('====> average validation time each epoch: {:,}m {:.2f}s'.format(
            int(avg_valid_time//60), avg_valid_time%60))
        print('====> training time: {}h {}m {:.2f}s'.format(
            int(train_time//3600), int((train_time%3600)//60), train_time%60))
        print('====> validation time: {}h {}m {:.2f}s'.format(
            int(validate_time//3600), int((validate_time%3600)//60), validate_time%60))
        print('====> total training time: {}h {}m {:.2f}s'.format(
            int(total_train_time//3600), int((total_train_time%3600)//60), total_train_time%60))

        return best_acc1
    
    elif args.run_type == 'evaluate':   # for evaluation
        # for evaluation on validation set
        print('\n===> [ Evaluation ]')
        
        # main evaluation
        start_time = time.time()
        acc1, acc5 = validate(args, val_loader, None, model, criterion)
        elapsed_time = time.time() - start_time
        print('====> {:.2f} seconds to evaluate this model\n'.format(
            elapsed_time))
        
        acc1 = round(acc1.item(), 4)
        acc5 = round(acc5.item(), 4)

        # save the result

        ckpt_name = '{}-{}-{}'.format(arch_name, args.dataset, args.load[:-4])
        save_eval([ckpt_name, acc1, acc5])

        if args.prune:
            _,_,sparsity = pruning.cal_sparsity(model)
            print('Sparsity : {}'.format(sparsity))
        return acc1
    else:
        assert False, 'Unkown --run-type! It should be \{train, evaluate\}.'
    



def eval(net):
    loader = testloader
    flag = 'Test'

    epoch_start_time = time.time()
    net.eval()
    val_loss = 0

    correct = 0


    total = 0
    criterion_CE = nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs= net(inputs)


        loss = criterion_CE(outputs[3], targets)
        val_loss += loss.item()

        _, predicted = torch.max(outputs[3].data, 1)
        total += targets.size(0)

        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    print('%s \t Time Taken: %.2f sec' % (flag, time.time() - epoch_start_time))
    print('Loss: %.3f | Acc net: %.3f%%' % (train_loss / (b_idx + 1), 100. * correct / total))
    return val_loss / (b_idx + 1),  correct / total



    for epoch in range(RESUME_EPOCH, FINAL_EPOCH+1):
        f = open(os.path.join("logs/" + path, 'log.txt'), "a")

        ### Train ###
        train_loss, acc = train(Teacher,Student, epoch)
        scheduler.step()
        scheduler_t.step()

        ### Evaluate  ###
        val_loss, test_acc  = eval(Student)


        f.write('EPOCH {epoch} \t'
                'ACC_net : {acc_net:.4f} \t  \n'.format(epoch=epoch, acc_net=test_acc)
                )
        f.close()

    utils.save_checkpoint({
        'epoch': epoch,
        'state_dict': Student.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, True, 'ckpt/' + path, filename='Model_{}.pth'.format(epoch))

def train(teacher,student, epoch):
    epoch_start_time = time.time()
    print('\n EPOCH: %d' % epoch)

    teacher.eval()
    student.train()

    train_loss = 0
    correct = 0
    total = 0

    global optimizer
    global optimizer_t

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        optimizer_t.zero_grad()

        # DML
        ###################################################################################
        teacher_outputs= teacher(inputs)
        student_outputs = student(inputs)


        s_loss = criterion_CE(student_outputs[3], targets) + criterion_kl(student_outputs[3], teacher_outputs[3])
        t_loss = criterion_CE(teacher_outputs[3], targets) + criterion_kl(teacher_outputs[3], student_outputs[3])
        loss = s_loss + t_loss
        ###################################################################################


        loss.backward()
        optimizer.step()
        optimizer_t.step()

        train_loss += loss.item()

        _, predicted = torch.max(student_outputs[3].data, 1)
        total += targets.size(0)

        correct += predicted.eq(targets.data).cpu().sum().float().item()


        b_idx = batch_idx

    print('Train s1 \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc net: %.3f%%|' % (train_loss / (b_idx + 1), 100. * correct / total))
    return train_loss / (b_idx + 1), correct / total


if __name__ == '__main__':
    start_time = time.time()
    args = hyperparam()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cu_num
    main(args)
    elapsed_time = time.time() - start_time
    print('====> total time: {}h {}m {:.2f}s'.format(
        int(elapsed_time//3600), int((elapsed_time%3600)//60), elapsed_time%60))

