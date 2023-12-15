import time
import warnings
import os
warnings.filterwarnings("ignore")

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import random

from utils import *
import pruning
import config
from data import DataLoader


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
    

cudnn.deterministic = True
cudnn.benchmark = False

#### random Seed ####
num = random.randint(1, 10000)
random.seed(num)
torch.manual_seed(num)
#####################

# Loss and Optimizer
criterion = nn.L1Loss()
criterion_kl = KLLoss()

def hyperparam():
    args = config.config()
    return args

def main(args):
    global arch_name_t, arch_name_s
    
    if args.cuda and not torch.cuda.is_available():
        raise Exception('No GPU found, please run without --cuda')
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cu_num

    # set model name
    arch_name_t, arch_name_s = set_arch_name(args, kd=1)
    print('\n=> creating model \'{}\', \'{}\''.format(arch_name_t, arch_name_s))
    
    # Load pretrained models
    pruner = pruning.__dict__[args.pruner] # default : dcil
    Teacher, image_size = pruning.models.__dict__[args.arch_t](data=args.dataset, num_layers=args.layers_t,
                                                width_mult=args.width_mult_t,
                                                depth_mult=args.depth_mult_t,
                                                model_mult=args.model_mult_t,
                                                mnn=pruner.mnn)


    Student, image_size = pruning.models.__dict__[args.arch_s](data=args.dataset, num_layers=args.layers_s,
                                                width_mult=args.width_mult_s,
                                                depth_mult=args.depth_mult_s,
                                                model_mult=args.model_mult_s,
                                                mnn=pruner.mnn)

    assert Teacher is not None, 'Unavailable Teacher model parameters!! exit...\n'
    assert Student is not None, 'Unavailable Student model parameters!! exit...\n'

    if len(args.load_pretrained) > 2 :
        path = args.load_pretrained
        state = torch.load(path)
        load_pretrained(Teacher, state)
        
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

    # load a pre-trained model
    if args.load is not None:
        checkpoint_t = load_checkpoint(Teacher, arch_name_t, args)
        checkpoint = load_checkpoint(Student, arch_name_s, args)

    # for training
    if args.run_type == 'train':
        # init parameters
        start_epoch = 0
        global iterations
        iterations = 0
        best_acc1 = 0.0        
        best_acc1_t = 0.0
        train_time = 0.0
        validate_time = 0.0

        os.makedirs('./results', exist_ok=True)
        file_train_acc_t = os.path.join('results', '{}.txt'.format('_'.join(['train', arch_name_t, args.dataset, args.save.split('.pth')[0]])))
        file_train_acc = os.path.join('results', '{}.txt'.format('_'.join(['train', arch_name_s, args.dataset, args.save.split('.pth')[0]])))
        file_test_acc_t = os.path.join('results', '{}.txt'.format('_'.join(['test', arch_name_t, args.dataset, args.save.split('.pth')[0]])))
        file_test_acc = os.path.join('results', '{}.txt'.format('_'.join(['test', arch_name_s, args.dataset, args.save.split('.pth')[0]])))

        # pruning
        target_sparsity =  args.target_epoch

        if args.prune_type == 'structured':
            filter_mask = pruning.get_filter_mask(Teacher, target_sparsity, args)
            pruning.filter_prune(Teacher, filter_mask)
        elif args.prune_type == 'unstructured':
            threshold = pruning.get_weight_threshold(Teacher, target_sparsity, args)
            pruning.weight_prune(Teacher, threshold, args)

        epochs = args.target_epoch + 75
        # for epoch in range(start_epoch, args.epochs):
        for epoch in range(start_epoch, epochs):

            print('\n==> s : {}, t : {} / {} training'.format(
                    arch_name_s, arch_name_t, args.dataset))
            print('==> Epoch: {}, lr = {}'.format(
                epoch, optimizer.param_groups[0]["lr"]))

            # train 
            print('===> [ Training ]')
            start_time = time.time()
            acc1_train_t, acc5_train_t, acc1_train, acc5_train = train(args, train_loader,
                epoch=epoch, teacher=Teacher, student=Student, scheduler=scheduler, scheduler_t=scheduler_t)

            elapsed_time = time.time() - start_time
            train_time += elapsed_time
            print('====> {:.2f} seconds to train for Teacher this epoch\n'.format(
                elapsed_time))

            # evaluate on validation set for Teacher
            print('===> [ Validation for Teacher ]')
            start_time = time.time()
            acc1_valid_t, acc5_valid_t = validate(args, val_loader,
                epoch=epoch, model=Teacher, criterion=criterion)
            elapsed_time = time.time() - start_time
            validate_time += elapsed_time
            print('====> {:.2f} seconds to validate for Teacher this epoch'.format(
                elapsed_time))

            # evaluate on validation set for Student
            print('===> [ Validation for Student ]')
            start_time = time.time()
            acc1_valid, acc5_valid = validate(args, val_loader,
                epoch=epoch, model=Student, criterion=criterion)
            elapsed_time = time.time() - start_time
            validate_time += elapsed_time
            print('====> {:.2f} seconds to validate for Student this epoch'.format(
                elapsed_time))

            acc1_train_t = round(acc1_train_t.item(), 4)
            acc5_train_t = round(acc5_train_t.item(), 4)
            acc1_valid_t = round(acc1_valid_t.item(), 4)
            acc5_valid_t = round(acc5_valid_t.item(), 4)

            acc1_train = round(acc1_train.item(), 4)
            acc5_train = round(acc5_train.item(), 4)
            acc1_valid = round(acc1_valid.item(), 4)
            acc5_valid = round(acc5_valid.item(), 4)

            open(file_train_acc, 'a').write(str(acc1_train)+'\n')
            open(file_test_acc, 'a').write(str(acc1_valid)+'\n')
            open(file_train_acc_t, 'a').write(str(acc1_train)+'\n')
            open(file_test_acc_t, 'a').write(str(acc1_valid)+'\n')

            # remember best Acc@1 and save checkpoint and summary csv file
            state_t = Teacher.state_dict()
            summary_t = [epoch, acc1_train_t, acc5_train_t, acc1_valid_t, acc5_valid_t]
            state = Student.state_dict()
            summary = [epoch, acc1_train, acc5_train, acc1_valid, acc5_valid]

            is_best_t = acc1_valid_t > best_acc1_t
            best_acc1_t = max(acc1_valid_t, best_acc1_t)
            if is_best_t:
                save_model(arch_name_t, args.dataset, state_t, args.save)
            save_summary(arch_name_t, args.dataset, args.save.split('.pth')[0], summary_t)
            
            is_best = acc1_valid > best_acc1
            best_acc1 = max(acc1_valid, best_acc1)
            if is_best:
                save_model(arch_name_s, args.dataset, state, args.save)
            save_summary(arch_name_s, args.dataset, args.save.split('.pth')[0], summary)

            # for pruning
            if args.prune:
                num_total, num_zero, sparsity = pruning.cal_sparsity(Teacher)
                print('\n====> teacher sparsity: {:.2f}% || num_zero/num_total: {}/{}'.format(sparsity, num_zero, num_total))
                num_total, num_zero, sparsity = pruning.cal_sparsity(Student)
                print('\n====> student sparsity: {:.2f}% || num_zero/num_total: {}/{}'.format(sparsity, num_zero, num_total))

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

        return best_acc1_t, best_acc1

    elif args.run_type == 'evaluate':   # for evaluation

        # evaluate on validation set for Teacher
        print('===> [ Validation for Teacher ]')
        start_time = time.time()
        acc1_t, acc5_t = validate(args, val_loader, model=Teacher)
        elapsed_time = time.time() - start_time
        validate_time += elapsed_time
        print('====> {:.2f} seconds to validate for Teacher this epoch'.format(
            elapsed_time))

        # evaluate on validation set for Student
        print('===> [ Validation for Student ]')
        start_time = time.time()
        acc1, acc5 = validate(args, val_loader,model=Student)
        elapsed_time = time.time() - start_time
        validate_time += elapsed_time
        print('====> {:.2f} seconds to validate for Student this epoch'.format(
            elapsed_time))

        
        acc1_t = round(acc1_t.item(), 4)
        acc5_t = round(acc5_t.item(), 4)
        acc1 = round(acc1.item(), 4)
        acc5 = round(acc5.item(), 4)

        # save the result
        ckpt_name_t = '{}-{}-{}'.format(arch_name_t, args.dataset, args.load[:-4])
        save_eval([ckpt_name_t, acc1_t, acc5_t])
        ckpt_name = '{}-{}-{}'.format(arch_name_s, args.dataset, args.load[:-4])
        save_eval([ckpt_name, acc1, acc5])

        if args.prune:
            _,_,sparsity_t = pruning.cal_sparsity(Teacher)
            _,_,sparsity = pruning.cal_sparsity(Student)
            print('Teacher Sparsity : {}'.format(sparsity_t))
            print('Student Sparsity : {}'.format(sparsity))
        return acc1
    else:
        assert False, 'Unkown --run-type! It should be \{train, evaluate\}.'



def train(args, train_loader, epoch, teacher, student, scheduler, scheduler_t, **kwargs):
    r"""Train model each epoch
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_t = AverageMeter('Loss', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    top1_t = AverageMeter('Acc@1', ':6.2f')
    top5_t = AverageMeter('Acc@5', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress_t = ProgressMeter(len(train_loader), batch_time, data_time,
                             losses_t, top1_t, top5_t, prefix="Epoch: [{}]".format(epoch))
    progress = ProgressMeter(len(train_loader), batch_time, data_time,
                             losses, top1, top5, prefix="Epoch: [{}]".format(epoch))

    teacher.eval()
    student.train()
    end = time.time()

    loader_len = len(train_loader)

    global optimizer
    global optimizer_t

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        scheduler.step(globals()['iterations'] / loader_len)
        scheduler_t.step(globals()['iterations'] / loader_len)
        data_time.update(time.time() - end)

        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        optimizer_t.zero_grad()

        # DML
        ###################################################################################
        teacher_outputs= teacher(inputs, 3)
        student_outputs = student(inputs, 3)


        s_loss = criterion(student_outputs, targets) + criterion_kl(student_outputs[3], teacher_outputs[3])
        t_loss = criterion(teacher_outputs, targets) + criterion_kl(teacher_outputs[3], student_outputs[3])
        loss = s_loss + t_loss
        ###################################################################################


        # measure accuracy and record loss
        acc1_t, acc5_t = accuracy(teacher_outputs, targets, topk=(1, 5))
        acc1, acc5 = accuracy(student_outputs, targets, topk=(1, 5))
        losses_t.update(t_loss.item(), input.size(0))
        losses.update(s_loss.item(), input.size(0))
        top1_t.update(acc1_t[0], input.size(0))
        top5_t.update(acc5_t[0], input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        if batch_idx % args.print_freq == 0:
            progress_t.print(batch_idx)
            progress.print(batch_idx)


        loss.backward()
        optimizer.step()
        optimizer_t.step()

        # measure elapsed time
        batch_time.update(time.time() - end)


        end = time.time()

        print('====> Acc@1 {top1_t.avg:.3f} Acc@5 {top5_t.avg:.3f}'
              .format(top1=top1_t, top5=top5_t))
        print('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        
    return top1_t.avg, top5_t.avg, top1.avg, top5.avg



def validate(args, val_loader, model):
    r"""Validate model each epoch and evaluation
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()


    with torch.no_grad():
        end = time.time()

        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs= model(inputs, 3)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)

            if batch_idx % args.print_freq == 0:
                progress.print(batch_idx)

            end = time.time()

        print('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg



if __name__ == '__main__':
    start_time = time.time()
    args = hyperparam()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cu_num
    main(args)
    elapsed_time = time.time() - start_time
    print('====> total time: {}h {}m {:.2f}s'.format(
        int(elapsed_time//3600), int((elapsed_time%3600)//60), elapsed_time%60))

