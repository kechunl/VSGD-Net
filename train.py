import time
import os
import numpy as np
import pdb
import sys
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
from util.logger import Logger
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from options.val_options import ValOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util.evaluate import evaluate
from util.visualize_detection import display_instances
import torch.distributed as dist
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))      
    print('Total training: %d epochs' % (opt.niter + opt.niter_decay))  
else:    
    start_epoch, epoch_iter = 1, 0
opt.start_epoch = start_epoch

opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 40

# logger
sys.stdout = Logger(filename=os.path.join(opt.checkpoints_dir, opt.name, 'log.txt'))

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
# Validation dataset
opt_val = ValOptions().parse(save=False, set_GPU=False)
opt_val.nThreads = 1   # val code only supports nThreads = 1
opt_val.batchSize = 1  # val code only supports batchSize = 1
opt_val.serial_batches = True  # no shuffle
opt_val.no_flip = True  # no flip
opt_val.gpu_ids = [opt_val.gpu_ids[0]]

# for visualization
opt_val.no_html = False

val_dataloader = CreateDataLoader(opt_val)
val_dataset_size = len(val_dataloader)
print('validation images = %d' % val_dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt, 'train')
val_visualizer = Visualizer(opt_val, 'val')

if opt.fp16:    
    from apex import amp
    model, [optimizer_G, optimizer_D, optimizer_detect] = amp.initialize(model, [model.optimizer_G, model.optimizer_D, optimizer_detect], opt_level='O1')             
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
else:
    optimizer_G, optimizer_D, optimizer_detect = model.module.optimizer_G, model.module.optimizer_D, model.module.optimizer_detect

total_steps = (start_epoch-1) * dataset_size + epoch_iter  # 0

display_delta = total_steps % opt.display_freq  # 0
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    print('learning rate of detection:', optimizer_detect.param_groups[0]['lr'])
    if opt.isTrain and len(opt.gpu_ids)>0 and not opt.fp16:
        data_loader.sampler.set_epoch(epoch)

    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize * len(opt.gpu_ids)
        epoch_iter += opt.batchSize * len(opt.gpu_ids)

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta # save fake images after training every 100 images

        torch.cuda.empty_cache()

        ############## Forward Pass ######################
        losses, generated, outputs = model(Variable(data['label']), Variable(data['inst']), Variable(data['image']), Variable(data['feat']), data['target'], infer=save_fake)
        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) and not isinstance(x, dict) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))
        # loss_dict = dict(zip(model.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)
        loss_detect = sum(loss for loss in loss_dict['Detect'].values()) + loss_dict['Detect']['loss_mask']
        # loss_GAN_detect = loss_G + loss_detect
        loss_recon = loss_dict['Recon']

        ############### Backward Pass ####################

        if not (opt.freeze_GAN and epoch < opt.niter_fix_GAN):
            # update detection weights (fpn + rpn + roi_heads)
            optimizer_detect.zero_grad() 
            optimizer_G.zero_grad()
            if opt.fp16:
                with amp.scale_loss(loss_detect, optimizer_detect) as scaled_loss: scaled_loss.backward()
                with amp.scale_loss(loss_G, optimizer_G) as scaled_loss: scaled_loss.backward()
            else:
                loss_G_detect = loss_detect + loss_G
                loss_G_detect.backward()
            optimizer_detect.step()
            optimizer_G.step()

            # update discriminator weights
            optimizer_D.zero_grad()
            if opt.fp16:                                
                with amp.scale_loss(loss_D, optimizer_D) as scaled_loss: scaled_loss.backward()                
            else:
                loss_D.backward()        
            optimizer_D.step()      
        else:
            # update detection weights (fpn + rpn + roi_heads)
            optimizer_detect.zero_grad() 
            optimizer_G.zero_grad()
            if opt.fp16:
                with amp.scale_loss(loss_detect, optimizer_detect) as scaled_loss: scaled_loss.backward()
                with amp.scale_loss(loss_G, optimizer_G) as scaled_loss: scaled_loss.backward()
            else:
                # loss_detect.backward()
                loss_G_detect = loss_detect + loss_G
                loss_G_detect.backward()
            optimizer_detect.step()
            optimizer_D.zero_grad()
            loss_D.backward()

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.detach().item() if not isinstance(v, int) else v for k, v in loss_dict.items() if not isinstance(v, dict)}            
            errors.update({k: v.data.detach().item() for k,v in loss_dict['Detect'].items()})
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

            # Debug Nan Loss
            # if any([np.isnan(err) for err in errors.values()]):
            #     f = open(os.path.join(opt.checkpoints_dir, opt.name, dist.get_rank()+'.txt'), "a")
            #     message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, epoch_iter, t)
            #     total = 0
            #     for k, v in errors.items():
            #         if v != 0:
            #             message += '%s: %.3f ' % (k, v)
            #             total += v
            #     message += 'Total: %.3f' % (total)

            #     f.write(message)


        ### display output images
        if save_fake:
            image_drawn1 = display_instances(data['label'][0,:], outputs[0]['boxes'], scores=outputs[0]['scores'], masks=outputs[0]['masks'], mask_thresh=0.5)
            image_drawn1 = image_drawn1.permute(1,2,0).numpy()
            image_drawn2 = display_instances(data['label'][0,:], outputs[0]['boxes'], scores=outputs[0]['scores'], masks=outputs[0]['masks'], mask_thresh=0.1)
            image_drawn2 = image_drawn2.permute(1,2,0).numpy()
            gt_drawn = display_instances(data['label'][0,:], data['target'][0]['boxes'], scores=None, masks=data['target'][0]['masks'], mask_thresh=0.5)
            gt_drawn = gt_drawn.permute(1,2,0).numpy()
            visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                   ('synthesized_image', util.tensor2im(generated.data[0])),
                                   ('real_image', util.tensor2im(data['image'][0])),
                                   ('detection_0.5', image_drawn1),
                                   ('detection_0.1', image_drawn2),
                                   ('gt', gt_drawn)])
            visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest', prefix=opt.netG)            
            # model.save('latest', prefix=opt.netG)
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break
       
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest', prefix=opt.netG)
        model.module.save(epoch, prefix=opt.netG)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    model.module.update_learning_rate(epoch)

    ### evaluate after every epoch
    # TODO: evaluation model on validation set, add reconstruction loss (l1)
    if epoch>0 and dist.get_rank() == 0:
        evaluate(model, val_dataloader, opt_val, coco_only=(epoch % opt.save_epoch_freq != 0), visualizer=val_visualizer, epoch=epoch, total_steps=total_steps)
    dist.barrier()
    model.train()
dist.destroy_process_group()
