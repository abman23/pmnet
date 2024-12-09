import torch
import torch.nn as nn


#RESULT_FOLDER = '/content/drive/MyDrive/Colab Notebooks/Joohan/PMNet_Extension_Result'
RESULT_FOLDER = 'PMNet_Extension_Result'
TENSORBOARD_PREFIX = f'{RESULT_FOLDER}/tensorboard'


def L1_loss(pred, target):
  loss = nn.L1Loss()(pred, target)
  return loss

def MSE(pred, target):
  loss = nn.MSELoss()(pred, target)
  return loss

def RMSE(pred, target, metrics=None):
  loss = (((pred-target)**2).mean())**0.5
  return loss

'''
Building segments loss function: Takes in two np arrays:
When the input is 0, there if the pred is not 0 - then it leads to increase in a count.
This count is averaged acrossed samples.
Need to be between (input,pred) and black pixels in input need to be matched with the black pixels in the pred.
'''

def building_segments(input,pred,target):
  building_prediction = torch.where((input!=0)&(pred!=0),1,0)
  building_input = torch.where((input!=0),1,0)
  building_prediction_sum = building_prediction.sum()
  building_input_sum = building_input.sum()
  avg_across_one_batch = building_prediction_sum/building_input_sum
  '''
  Uncomment the following lines to debug these outputs :
  print("Building_prediction_sum ",building_prediction_sum)
  print("Building input sum ",building_input_sum)
  print("Avg across one batch ",avg_across_one_batch)
  '''
  return avg_across_one_batch

'''
It computes the rmse in the ROI region alone and avoids checking the
building segmentation loss as well that is unrelated to the problem statement.
'''
def roi_rmse_loss(input,pred,target):
  build_count = 0
  input = input[:,0,:,:].unsqueeze(1)
  #print("Input dimension ",input.shape)
  error_tensor = torch.where(input==0,(pred-target)**2,0)
  sum_torch = error_tensor.sum()
  #print("Sum torch ",sum_torch)
  count_non_zero = error_tensor.count_nonzero()
  error_tensor_mean = sum_torch/count_non_zero
  #print("Mean : ",error_tensor_mean)
  #print("Max value: ",error_tensor.max())
  error_float = (error_tensor_mean)**0.5
  #building_sum = torch.sum(building_tensor)
  #print("count_non_zero:",count_non_zero)
  #print("Error float per batch ",error_float)
  return error_float