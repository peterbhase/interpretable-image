import os
import torch

def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print):
    '''
    model: this is not the multigpu model
    '''
    if accu > target_accu:
        log('\tnew best: %.2f       old best: %.2f' % (accu * 100, target_accu * 100))
        torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '.pth')))
