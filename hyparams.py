from glob import glob
import os

class HyParams:
      
	def __init__(self, **kwargs):
		self.data = {}

		for key, value in kwargs.items():
			self.data[key] = value

#   to get hyparam     
	def __getattr__(self, key):
		if key not in self.data:
			raise AttributeError("'HyParams' object has no attribute %s" % key)
		return self.data[key]

#   to set hyparam  
	def set_hparam(self, key, value):
		self.data[key] = value
		
# Default hyperparameters
hyparams = HyParams(
	
    img_size= 224,       	# input image size
	
	batch_size= 21,     	# batch size
	eval_batch_size= 21,		# eval batch size
	lr = 1e-4,          	# learning rate
    total_epoch = 30,   	# total number of epochs
	
    checkpoint_interval= 3000,   # no of steps before saving checkpoint
	eval_interval= 500,         # no of steps before eval step
    
	seed = 12				# seed for random.
)
	