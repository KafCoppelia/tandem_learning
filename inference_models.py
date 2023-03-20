from torchsummary import summary
from models.cifar10 import CifarNetSmallIF_Inference, CifarNetSmall, CifarNet
from models.cifar10_sj import CifarNetSmall_SJ
import torch
import torch.backends.cudnn as cudnn
from data.data_loader_cifar10 import get_test_loader
import os, sys
import time

home_dir = os.getcwd()
sys.path.insert(0, home_dir)
data_dir = os.path.join(home_dir, 'data/')
test_loader = get_test_loader(data_dir)

'''
    Inference
'''
if __name__ == "__main__":
    Tencode = 1
    neuronParam = {
		'neuronType': 'LIF',
		'vthr': 1,
		'leaky_rate_mem': 0.8
	}
    device = torch.device("cuda")
    
    # model = CifarNetSmallIF_Inference(neuronParam, Tencode, f"./exp/cifar10/lif_small_tencode{Tencode}_ckp.pt", device)
    model = CifarNet()
    # model = CifarNetSmall_SJ(Tencode)
    model = model.to(device)
    
    if device == torch.device('cuda'):
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    
    # state = torch.load(f"./exp/cifar10/sj_small_tencode{Tencode}_ckp.pt", map_location=torch.device("cpu"))
    state = torch.load(f"./exp/cifar10/ann_ckp.pt", map_location=torch.device("cpu"))
    weights = state["model_state_dict"]
    model.load_state_dict(weights)
    
    model.eval()  # Put the model in test mode
    
    
    print(summary(model, input_size=(3, 32, 32)))
    
    # start = time.time()
    
    # correct = 0
    # total = 0
    # for data in test_loader:
    #     inputs, labels = data

    #     # Transfer to GPU
    #     inputs, labels = inputs.type(torch.FloatTensor).to(device), \
    #         labels.type(torch.LongTensor).to(device)

    #     # forward pass
    #     y_pred = model.forward(inputs)
    #     # print(y_pred.shape)
    #     _, predicted = torch.max(y_pred.data, 1)

    #     total += labels.size(0)
    #     correct += (predicted == labels).sum().item()

    # acc = correct/total

    # print('Total time: {:.2f}'.format(time.time() - start))
    # print('Test Accuracy: {:4f}'.format(acc))