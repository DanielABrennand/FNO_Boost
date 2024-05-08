
#%% Imports
from simvue import Run
import numpy as np
import torch
import os
import functools
import operator
import h5py
import timeit

#%% Setup and Configuration recording

t1 = timeit.default_timer()

SEED = None
if not SEED:
    SEED = np.random.randint(0,4294967295)
torch.manual_seed(SEED)
np.random.seed(SEED) 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

configuration = {"Case": 'RBB Camera', #Specifying the Camera setup
                 "Pipeline": 'Sequential', #Shot-Agnostic RNN windowed data pipeline. 
                 "Calibration": 'Calcam', #CAD inspired Geometry setup
                 "Epochs": 500, 
                 "Batch Size": 10,
                 "Optimizer": 'Adam',
                 "Learning Rate": 0.005,
                 "Scheduler Step": 100,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'GELU',
                 "Normalisation Strategy": 'Range',
                 "Instance Norm": 'No', #Layerwise Normalisation
                 "Log Normalisation":  'No',
                 "Physics Normalisation": 'No', #Normalising the Variable 
                 "T_in": 10, #Input time steps
                 "T_out": 10, #Max simulation time
                 "Step": 10, #Time steps output in each forward call
                 "Modes":16, #Number of Fourier Modes
                 "Width": 16, #Features of the Convolutional Kernel
                 "Loss Function": 'LP-Loss', #Choice of Loss Fucnction
                 "Resolution":1,
                 "Seed": SEED, #Nmpy and PyTorch randomiser seed
                 "Device": str(DEVICE),
                 "Train Percentage":0.8 #Percentage of data used for testing
                 }

#run_tags = ['FNO', 'Camera', 'rbb', 'Forecasting', 'shot-agnostic']
#run_folder = "/FNO_Boost"

#run = Run(mode='online')
#run.init(folder=run_folder, tags=run_tags, metadata=configuration)


#%% Data Locations
#Loading
DATA_FOLDER = "/home/ir-bren1/rds/rds-ukaea-ap002-mOlK9qn0PlQ/ir-bren1/FNO_Boost_Test"

#Saving
MODEL_SAVE_LOC = "/home/ir-bren1/rds/rds-ukaea-ap002-mOlK9qn0PlQ/ir-bren1/Heuristics/FNO_Boost/"
FIGURE_SAVE_LOC = ""

# %% Normalisationfunctions

class UnitGaussianNormalizer(object):
    # normalization, pointwise gaussian
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class GaussianNormalizer(object):
    # normalization, Gaussian - across the entire dataset
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class RangeNormalizer(object):
    # normalization, scaling by range - pointwise
    def __init__(self, x, low=-1.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x


    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()

class MinMax_Normalizer(object):
    #normalization, rangewise but across the full domain 
    def __init__(self, x, low=-1.0, high=1.0):
        super(MinMax_Normalizer, self).__init__()
        mymin = torch.min(x)
        mymax = torch.max(x)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.reshape(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

    def cuda(self):
        self.a = self.a.cuda()
        self.b = self.b.cuda()

    def cpu(self):
        self.a = self.a.cpu()
        self.b = self.b.cpu()

#%% Loss Functions

class LpLoss(object):
    #loss function with rel/abs Lp loss
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


#%% Model Architecture

class SpectralConv2d(torch.nn.Module):
    #2D Fourier layer. It does FFT, linear transform, and Inverse FFT.  
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()  

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = torch.nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = torch.nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # Complex multiplication
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(torch.nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous T_in timesteps + 2 locations (u(t-T_in, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=x_discretistion, y=y_discretisation, c=T_in)
        output: the solution of the next timestep
        output shape: (batchsize, x=x_discretisation, y=y_discretisatiob, c=step)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = torch.nn.Linear(T_in+2, self.width)
        # input channel is 12: the solution of the previous T_in timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv5 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)


        self.w0 = torch.nn.Conv2d(self.width, self.width, 1)
        self.w1 = torch.nn.Conv2d(self.width, self.width, 1)
        self.w2 = torch.nn.Conv2d(self.width, self.width, 1)
        self.w3 = torch.nn.Conv2d(self.width, self.width, 1)
        self.w4 = torch.nn.Conv2d(self.width, self.width, 1)
        self.w5 = torch.nn.Conv2d(self.width, self.width, 1)

        # self.norm = nn.InstanceNorm2d(self.width)
        self.norm = torch.nn.Identity()

        self.fc1 = torch.nn.Linear(self.width, 128)
        self.fc2 = torch.nn.Linear(128, step)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.norm(self.conv0(self.norm(x)))
        x2 = self.w0(x)
        x = x1+x2
        x = torch.nn.functional.gelu(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x2 = self.w1(x)
        x = x1+x2
        x = torch.nn.functional.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x2 = self.w2(x)
        x = x1+x2
        x = torch.nn.functional.gelu(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x2 = self.w3(x)
        x = x1+x2

        x1 = self.norm(self.conv4(self.norm(x)))
        x2 = self.w4(x)
        x = x1+x2

        x1 = self.norm(self.conv5(self.norm(x)))
        x2 = self.w5(x)
        x = x1+x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        #Using x and y values from the simulation discretisation 
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(x_grid, dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(y_grid, dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += functools.reduce(operator.mul, list(p.size()))

        return c

#%% Data Loading and pre-processing

T_In = configuration["T_in"]
T_Out = configuration["T_out"]

def FilamentIdentification(Frames,frame_range):
    #Returns an array of just the filaments from the input frames by subtracting the average frame
    #Assumes frames are in the form [X,Y,T] and that the return will be smaller in the T dimension by 
    #2 times the frame_range

    #This just makes ure its a numpy array
    Frames = np.array(Frames)

    num_frames = Frames.shape[2]
    num_smoothed_frames = num_frames-(2*frame_range)
    subtracted_frames = np.zeros((Frames.shape[0],Frames.shape[1],num_smoothed_frames))
    for n in range(num_smoothed_frames):
        frm = n+frame_range
        subset_frames = Frames[:,:,(frm-frame_range):(frm+frame_range+1)]

        avg_frame = np.mean(subset_frames,2)
        subtracted_frames[:,:,n] = Frames[:,:,frm] - avg_frame

    return subtracted_frames


Data_File_Names = os.listdir(DATA_FOLDER)
Data_File_Names = [x for x in Data_File_Names if ".h5" in x]

#run.update_metadata({'Files used': Data_File_Names})

Base_Shots_List_Input = []
Base_Shots_List_Output = []
Fil_Shot_List_Output = []
for n,Data_File in enumerate(Data_File_Names):
    print("Importing file {} of {}".format(n,len(Data_File_Names)))
    Full_FilePath = os.path.join(DATA_FOLDER,Data_File)
    F = h5py.File(Full_FilePath)
    Num_Frames = len([x for x in F.keys() if "frame" in x])
    Shot_Frames = []
    #Collate frames in the shot:
    for Frame in range(Num_Frames):
        Shot_Frames.append(torch.from_numpy(np.array(F["frame{}".format(str(Frame).zfill(4))]).astype(np.float32)))
    #Shot_Frames is now [X,Y,T] for all T
    Shot_Frames = torch.stack(Shot_Frames,-1)

    #NOTE BIG THING HERE: using the same smoothing range as the cutoff at the start/end of a frame as
    #it makes the maths nice

    #Calculates the high frequency elements for this shot (Note that this removes the first and last 
    #"Frame_Range" number of frames from the start and end)
    #NOTE remove magic number 5 (should be in config as something like smoothing range)
    Fil_Frames = torch.from_numpy(FilamentIdentification(Shot_Frames,5).astype(np.float32))

    #Remove the first and last few frames
    #NOTE replace magic number 5
    Shot_Frames = Shot_Frames[:,:,5:-5]

    #Now split into in/out sets of equal length:
    Sets_Number = int(np.floor((Shot_Frames.shape[2] - T_Out)/T_In))
    Set_Length = T_In + T_Out

    for Set in range(Sets_Number):
        Base_Set = Shot_Frames[:,:,(Set*T_In):((Set*T_In)+Set_Length)]      
        Fil_Set = Fil_Frames[:,:,(Set*T_In):((Set*T_In)+Set_Length)]

        #Splitting into inputs and outputs (a and u)
        Base_Shots_List_Input.append(Base_Set[:,:,:(Base_Set.shape[2]//2)])
        Base_Shots_List_Output.append(Base_Set[:,:,(Base_Set.shape[2]//2):])
        Fil_Shot_List_Output.append(Fil_Set[:,:,(Base_Set.shape[2]//2):])

Base_Shots_Tensor_Input = torch.stack(Base_Shots_List_Input) #Now in form [Index,X,Y,T]
Base_Shots_Tensor_Output = torch.stack(Base_Shots_List_Output) #Now in form [Index,X,Y,T]
Fil_Shots_Tensor_Output = torch.stack(Fil_Shot_List_Output) #Now in form [Index,X,Y,T]

#Possibly do a shuffle here on axis 0

#Splitting into test/train
Total_Batches = Base_Shots_Tensor_Input.shape[0]
Train_Size = int(Total_Batches*configuration["Train Percentage"])
Test_Size = Total_Batches - Train_Size


Base_Shots_Tensor_Input_Train = Base_Shots_Tensor_Input[:Train_Size,:,:,:]
Base_Shots_Tensor_Input_Test = Base_Shots_Tensor_Input[Train_Size:,:,:,:]

Base_Shots_Tensor_Output_Train = Base_Shots_Tensor_Output[:Train_Size,:,:,:]
Base_Shots_Tensor_Output_Test = Base_Shots_Tensor_Output[Train_Size:,:,:,:]

Fil_Shots_Tensor_Output_Train = Fil_Shots_Tensor_Output[:Train_Size,:,:,:]
Fil_Shots_Tensor_Output_Test = Fil_Shots_Tensor_Output[Train_Size:,:,:,:]

#Normalising
#All normalisations preserve data structure
norm_strategy = configuration['Normalisation Strategy']

if norm_strategy == 'Min-Max':
    input_normalizer = MinMax_Normalizer(Base_Shots_Tensor_Input_Train)
    base_output_normalizer = MinMax_Normalizer(Base_Shots_Tensor_Output_Train)
    filament_output_normalizer = MinMax_Normalizer(Fil_Shots_Tensor_Output_Train)

if norm_strategy == 'Range':
    input_normalizer = RangeNormalizer(Base_Shots_Tensor_Input_Train)
    base_output_normalizer = RangeNormalizer(Base_Shots_Tensor_Output_Train)
    filament_output_normalizer = RangeNormalizer(Fil_Shots_Tensor_Output_Train)

if norm_strategy == 'Gaussian':
    input_normalizer = GaussianNormalizer(Base_Shots_Tensor_Input_Train)
    base_output_normalizer = GaussianNormalizer(Base_Shots_Tensor_Output_Train)
    filament_output_normalizer = GaussianNormalizer(Fil_Shots_Tensor_Output_Train)

Base_Shots_Tensor_Input_Train = input_normalizer.encode(Base_Shots_Tensor_Input_Train)
Base_Shots_Tensor_Input_Test = input_normalizer.encode(Base_Shots_Tensor_Input_Test)

Base_Shots_Tensor_Output_Train = base_output_normalizer.encode(Base_Shots_Tensor_Output_Train)
Base_Shots_Tensor_Output_Test = base_output_normalizer.encode(Base_Shots_Tensor_Output_Test)

Fil_Shots_Tensor_Output_Train = filament_output_normalizer.encode(Fil_Shots_Tensor_Output_Train)
Fil_Shots_Tensor_Output_Test = filament_output_normalizer.encode(Fil_Shots_Tensor_Output_Test)


#Combining Output tensors for the two models
Combined_Tensor_Output_Train = torch.stack((Base_Shots_Tensor_Output_Train,Fil_Shots_Tensor_Output_Train),1) #In the form [Index,Base or Filament,X,Y,T]
Combined_Tensor_Output_Test = torch.stack((Base_Shots_Tensor_Output_Test,Fil_Shots_Tensor_Output_Test),1) #In the form [Index,Base or Filament,X,Y,T]

#Extracting hyperparameters from the config dict
modes = configuration['Modes']
width = configuration['Width']
step = configuration['Step']
batch_size = configuration['Batch Size']
T_in = configuration['T_in']
T_Out = configuration['T_out']

#Creating data loaders
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Base_Shots_Tensor_Input_Train, Combined_Tensor_Output_Train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Base_Shots_Tensor_Input_Test, Combined_Tensor_Output_Test), batch_size=batch_size, shuffle=True)

# Using arbitrary R and Z positions sampled uniformly within a specified domain range. 
# pad the location (x,y)
x_grid = np.linspace(-1.5, 1.5, 448)
y_grid = np.linspace(-2.0, 2.0, 640)


#%% Training setup

#Instantiating the Models. 
basemodel = FNO2d(modes, modes, width)
filamentmodel = FNO2d(modes, modes, width)

basemodel.to(DEVICE)
filamentmodel.to(DEVICE)

#Setting up optimizers and schedulers
baseoptimizer = torch.optim.Adam(basemodel.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
basescheduler = torch.optim.lr_scheduler.StepLR(baseoptimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
filamentoptimizer = torch.optim.Adam(filamentmodel.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
filamentscheduler = torch.optim.lr_scheduler.StepLR(filamentoptimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])

myloss = LpLoss(size_average=False)
epochs = configuration['Epochs']

t2 = timeit.default_timer()
print("Setup complete in: {}".format(t2-t1))

#%%Training Loop
for ep in range(epochs): #Training Loop - Epochwise
    basemodel.train()
    filamentmodel.train()

    t1 = timeit.default_timer()

    base_train_l2= 0
    base_test_l2 = 0
    fil_train_l2= 0
    fil_test_l2 = 0
    for xx, yy in train_loader: #Training Loop - Batchwise
        baseoptimizer.zero_grad()
        filamentoptimizer.zero_grad()
        xx = xx.to(DEVICE)
        yy = yy.to(DEVICE)

        #Base model
        baseout = basemodel(xx)        
        baseloss = myloss(baseout.reshape(batch_size, -1), yy[:,0,:,:,:].reshape(batch_size, -1)) 
        base_train_l2 += baseloss

        baseloss.backward()
        baseoptimizer.step()

        #Filament model
        filout = filamentmodel(xx)
        filloss = myloss(filout.reshape(batch_size, -1), yy[:,1,:,:,:].reshape(batch_size, -1)) 
        fil_train_l2 += filloss

        filloss.backward()
        filamentoptimizer.step()

    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(DEVICE)
            yy = yy.to(DEVICE)

            #Base model
            baseout = basemodel(xx)        
            baseloss = myloss(baseout.reshape(batch_size, -1), yy[:,0,:,:,:].reshape(batch_size, -1)) 
            base_test_l2 += baseloss

            #Filament model
            filout = filamentmodel(xx)
            filloss = myloss(filout.reshape(batch_size, -1), yy[:,1,:,:,:].reshape(batch_size, -1)) 
            fil_test_l2 += filloss

    basescheduler.step()
    filamentscheduler.step()

    base_train_loss = base_train_l2 / Train_Size
    base_test_loss = base_test_l2 / Test_Size

    fil_train_loss = fil_train_l2 / Train_Size
    fil_test_loss = fil_test_l2 / Test_Size

    t2 = timeit.default_timer()

    print("Epoch: {}, Time: {}, Base Train Loss:{}, Base Test Loss: {}, Filament Train Loss: {}, Filament Test Loss: {}".format(ep,(t2-t1),base_train_loss,base_test_loss,fil_train_loss,fil_test_loss))
    #PR_File.write("Epoch: {}, Time: {}, Base Train Loss:{}, Base Test Loss: {}, Filament Train Loss: {}, Filament Test Loss: {}\n".format(ep,(t2-t1),base_train_loss,base_test_loss,fil_train_loss,fil_test_loss))
    #un.log_metrics({'Base Train Loss': base_train_loss,' Base Test Loss': base_test_loss,'Filament Train Loss':fil_train_loss,'Filament Test Loss':fil_test_loss})

#Saving the models

torch.save(basemodel.state_dict(), "{}Test1-BaseModel.pth".format(MODEL_SAVE_LOC))
torch.save(basemodel.state_dict(), "{}Test2-FilamentModel.pth".format(MODEL_SAVE_LOC))

#run.close()


