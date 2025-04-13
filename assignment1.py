# First of all, we need to import the data as wells as the required packages
import numpy as np
import matplotlib.pyplot as mpt
import tqdm as tqdm
import typing as tp
import os
train_data=np.load('Assignment1-Dataset/train_data.npy')
train_label=np.load('Assignment1-Dataset/train_label.npy')
test_data=np.load('Assignment1-Dataset/test_data.npy')
test_label=np.load('Assignment1-Dataset/test_label.npy')
# train_data=np.load('Test-Dataset/X_train.npy')
# train_label=np.load('Test-Dataset/y_train.npy')
# test_data=np.load('Test-Dataset/X_test.npy')
# test_label=np.load('Test-Dataset/y_test.npy')
print(f"train_data:{train_data.shape}\ntrain_label:{train_label.shape}\ntest_data:{test_data.shape}\ntest_label:{test_label.shape}")

# %matplotlib inline
# The most important component is the DATA, we need to analyze and preprocess it! 
# However, the feature dimension is 128, QUITE HUGE! We need to use certain techniques to lower the dimension.
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Normalize the data
std=StandardScaler()
print(std.fit(train_data))
# print(f"mean is {std.mean_} and variance is {std.var_}")
# We definitely need to normalize the data then
train_data=std.transform(train_data)
# print(train_data)

# # Visualize the data for better understanding
# pca=PCA(n_components=128,whiten=True,svd_solver='full')
# pca_train_data=pca.fit_transform(train_data)
# # print(pca_train_data.shape)
# # Apply UMAP to pac transformed data
# import umap as up
# umap_reducer = up.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
# train_data_umap = umap_reducer.fit_transform(pca_train_data)

# mpt.figure(figsize=(8,6))
# scatter=mpt.scatter(x=train_data_umap[:,0],y=train_data_umap[:,1],cmap='viridis',c=train_label)
# mpt.colorbar(scatter,label='Class')
# mpt.xlabel('feature1')
# mpt.ylabel('feature2')
# mpt.savefig('results/data')

# Activate function class, to better organize different activate functions
import math
class Activation:
    def __init__(self,activation:tp.Optional[str],alpha=0.01):
        #alpha is used for leaky relu
        self.fn=activation
        self.alpha=alpha
    
    def forward(self,z):
        if self.fn=='sigmoid':
            return self._sigmoid(z)
        elif self.fn=='relu':
            return self._relu(z)
        elif self.fn=='softmax':
            return self._softmax(z)
        elif self.fn=='tanh':
            return self._tanh(z)
        elif self.fn=='leaky_relu':
            return self._leaky_relu(z)
        elif self.fn=='gelu':
            return self._gelu(z)
        
    def derivative(self,z):
        if self.fn=='sigmoid':
            return self._sigmoid_derivative(z)
        elif self.fn=='relu':
            return self._relu_derivative(z)
        elif self.fn=='softmax':
            return self._softmax_derivative(z)
        elif self.fn=='tanh':
            return self._tanh_derivative(z)
        elif self.fn=='leaky_relu':
            return self._leaky_relu_derivative(z)
        elif self.fn=='gelu':
            return self._gelu_derivative(z)
    @staticmethod  
    def _sigmoid(z):
        return 1/(1+np.exp(-z))
    
    def _sigmoid_derivative(self,z):
        return self._sigmoid(z)*(1-self._sigmoid(z))
    @staticmethod
    def _tanh(z):
        # return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        # Overflow might occur using manul written function, use numpy instead
        return np.tanh(z)
    @staticmethod
    def _tanh_derivative(self,z):
        return 1-self._tanh(z)**2
    @staticmethod
    def _relu(z):
        # ReLU can mitigate vanishing gradient, but there will be "dead" neurons
        return np.maximum(0,z)
    
    def _relu_derivative(self,z):
        return (z>0).astype(float)
    
    def _leaky_relu(self,z):
        # LeakyReLU can recover some certain "dead" neurons
        return z if z>0 else self.alpha*z
    
    def _leaky_relu_derivative(self,z):
        return 1.0 if z>0 else self.alpha
        
    
    def _gelu(self,z):
        # GeLU is smoother than ReLU and can avoid gradient vanishing
        return 0.5*z*(1+self._tanh(math.sqrt(2/math.pi)*(z+0.044715*z**3)))
    
    def _gelu_derivative(self,z):
        left=1+self._tanh(math.sqrt(2/math.pi)*(z+0.044715*z**3))
        right=z*(1-self._tanh(math.sqrt(2/math.pi)*(z+0.044715*z**3))**2)*(math.sqrt(2/math.pi)*(1+3*0.044715*z**2))
        return 2*left*right
        
    @staticmethod
    def _softmax(z):
        # To address the underflow and 0/0=NAN issue(when the value is very small it would be treaed as 0)
        # "Stability Trick is used" --- new z = old z - max (z) to ensure the denominator is at least 1
        exp_z=np.exp(z-np.max(z,axis=1,keepdims=True))
        return exp_z/np.sum(exp_z,axis=1,keepdims=True)
    
    def _softmax_derivative(self,z):
        return 1
    
    # Normalization class, to normalize the data for better training performance
from numpy import ndarray
from abc import ABC
class Normalizer(ABC):
    def __init__(self,dim:int,eps:float=1e-5,momentum:float=0.9,learning_rate=0.01):
        self.dim=dim
        self.eps=eps
        self.momentum=momentum
        self.lr=learning_rate
    
    @classmethod
    def forward(self,x:np.ndarray,training:bool=True) -> tuple[np.ndarray,tuple[np.ndarray]] | np.ndarray:
        pass
    
    @classmethod
    def backward(self,do:np.ndarray,cache:tuple[np.ndarray]) -> np.ndarray:
        pass
    
    @classmethod
    def get_params(self):
        pass
    
    @classmethod
    def zero_grad(self):
        pass
    

class BatchNorm(Normalizer):
    def __init__(self, dim:int, eps:float=1e-5, momentum:float=0.9, learning_rate:float=0.01):
        super().__init__(dim=dim,eps=eps,momentum=momentum,learning_rate=learning_rate)
        # Trainable params: 
        # Gamma is mean shift, while beta is variance shift
        self.gamma=np.ones((1,self.dim))
        self.beta=np.zeros((1,self.dim))
        
        # mean and variance for each column(feature)
        self.running_mean=np.zeros((1,self.dim))
        self.running_std=np.zeros((1,self.dim))
        # saved_mean and saved_std are only used in inference step
        self.saved_mean=np.zeros((1,self.dim))
        self.saved_std=np.zeros((1,self.dim))
        
        
        
    def forward(self,x:np.ndarray,training:bool=True) -> tuple[np.ndarray,tuple[np.ndarray]] | np.ndarray:
        
        # When the model is training, mean and standard variance will change accordingly to each batch.
        if training:
        # Calculate the mean and standard variance(eps will be added to avoid 0 in denominator) for this batch
            self.running_mean=np.sum(x,axis=0)/x.shape[0]
            var=np.sum((x-self.running_mean)**2,axis=0)/x.shape[0]
            self.running_std=np.sqrt(var+self.eps)
        
        # Moving Average --> Store the state of mean and std values for inference, EMA(Exponential Moving Average) is used
            self.saved_mean=self.momentum*self.saved_mean+(1-self.momentum)*self.running_mean
            self.saved_std=self.momentum*self.saved_std+(1-self.momentum)*self.running_std
        
        # Normalize
            x_mu=x-self.running_mean
            x_normalize=x_mu/self.running_std
        
        # Scale and Shift
            bn=self.gamma*x_normalize+self.beta
            cache=(x_normalize,x_mu,var)
            return bn,cache
        else:
        # When the model is inferencing, just use the stored variance
            x_normalize=(x-self.saved_mean)/self.saved_std
            bn=self.gamma*x_normalize+self.beta
            
            return bn
        
    
    def backward(self,do:ndarray,cache:tuple[ndarray]) -> ndarray:
        # BatchNorm should be used for mini-batch training, remember to sum up the whole batch when calculating gradient
        x_normalize=cache[0]
        batch_size=x_normalize.shape[0]
        dbeta=np.sum(do,axis=0) #(D,)
        dgamma=np.sum((do*x_normalize),axis=0) #(D,)
        dx_hat=do*self.gamma #(N,D)
        ivar=1./self.running_std
        dx_mu1=dx_hat*ivar #(N,D)
        x_mu=cache[1]
        divar=np.sum(dx_hat*x_mu,axis=0) #(D,)
        dsqrt_var=divar*(-1)/self.running_std**2 #(D,)
        var=cache[2]
        dvar=1/(2*np.sqrt(var+self.eps))*dsqrt_var #(D,)
        dsq=dvar*np.ones((batch_size,self.dim))/batch_size #(N,D)
        dx_mu2=2*x_mu*dsq #(N,D)
        dx1=dx_mu1+dx_mu2 #(N,D)
        dmu=(-1)*np.sum(dx_mu1+dx_mu2,axis=0) #(D,)
        dx2=np.ones((batch_size,self.dim))/batch_size*dmu #(N,D)
        dx=dx1+dx2
        
        # Update gamma and beta
        self.gamma-=self.lr*dgamma
        self.beta-=self.lr*dbeta
        
        return dx
    
    def get_params(self):
        return {
        'gamma':self.gamma,
        'beta':self.beta,
        'running_mean':self.running_mean,
        'running_std':self.running_std,
        'saved_mean':self.saved_mean,
        'saved_std':self.saved_std,
    }
    
    def shape(self):
        return self.dim
    
class Config:
    def __init__(self):
        pass
    
    def to_dict(self):
        return str(self.__dict__)
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
                
# OptimizerConfig, to configure the hyperparameters of chosen optimizer
class OptimizerConfig(Config):
    def __init__(self,name='SGD',lr=0.001,momentum=0.9,
                 weight_decay=1e-4,beta1=0.9,beta2=0.99,
                 epsilon=1e-8):
        super().__init__()
        self.lr=lr
        self.name=name
        self.momentum=momentum
        self.wd=weight_decay
        self.beta1=beta1
        self.beta2=beta2
        self.el=epsilon

# LayerConfig, to configure each layer of the MLP
class LayerConfig(Config):
    def __init__(self,in_dim,out_dim,
                 activation:tp.Optional[Activation]=Activation('relu'),
                 dropout:tp.Optional[float]=0., # represent the dropout rate of neurons in each layer
                 w_initializer:tp.Optional[str]='xavier',
                 b_initializer:tp.Optional[str]='uniform',
                 optimizer:tp.Optional[OptimizerConfig]=OptimizerConfig()):
        super().__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.activation=activation
        self.dropout=dropout
        self.w_initializer=w_initializer
        self.b_initializer=b_initializer
        self.optimizer=optimizer
        
# MLP Config, to configure each layers
# class MLPConfig(Config):
#     def __init__(self,epochs)

# Optimizer Class, an abstract class where specific optimizers will be further implemented
class Optimizer(ABC):
    def __init__(self,params,learning_rate:float=1e-3,weight_decay:float=0.0):
        self.params=params
        self.lr=learning_rate
        self.weight_decay=weight_decay
    
    @classmethod
    def step(self):
        pass
    
    @classmethod
    def zero_grad(self): # This is optinal
        for p in self.params:
            p['grad'].fill(0.0)
    

class SGD(Optimizer): # mini-batch training is already implemented
    def __init__(self,params,learning_rate:float=1e-3,weight_decay:float=0.0):
        super().__init__(params=params,learning_rate=learning_rate,weight_decay=weight_decay)
    
    def step(self):
        for p in self.params:
            grad=p['grad']
            if self.weight_decay>0:
                grad=grad+self.weight_decay*p['param']
            p['param']-=self.lr*grad
            
class SGD_Momentum(Optimizer):
    def __init__(self,params,learning_rate:float=1e-3,weight_decay:float=0.0,momentum=0.9):
        super().__init__(params=params,learning_rate=learning_rate,weight_decay=weight_decay)
        self.momentum=momentum
        self.velocity=[np.zeros_like(p['param']) for p in self.params]
        
    def step(self):
        for i,p in enumerate(self.params):
            grad=p['grad']
            if self.weight_decay>0:
                grad=grad+self.weight_decay*p['param']
            self.velocity[i]=self.momentum*self.velocity[i]+self.lr*grad
            p['param']-=self.velocity[i]
            

class Adam(Optimizer):
    def __init__(self,params,learning_rate:float=1e-3,weight_decay:float=0.0,beta1=0.9,beta2=0.999,epsilon=1e-8):
        super().__init__(params=params,learning_rate=learning_rate,weight_decay=weight_decay)
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon
        self.m=[np.zeros_like(p['param']) for p in self.params] # First moment estimation
        self.v=[np.zeros_like(p['param']) for p in self.params] # Second moment estimation
        self.m_hat=[np.zeros_like(p['param']) for p in self.params] # update m accordingly
        self.v_hat=[np.zeros_like(p['param']) for p in self.params] # update v accordingly
        self.t=np.zeros((len(self.params),)) # Timestamp to keep an record
        
    def step(self):
        for i,p in enumerate(self.params):
            grad=p['grad']
            if self.weight_decay>0:
                grad=grad+self.weight_decay*p['param']
            self.t[i]+=1
            self.m[i]=self.beta1*self.m[i]+(1-self.beta1)*grad
            self.v[i]=self.beta2*self.v[i]+(1-self.beta2)*grad**2
            self.m_hat[i]=self.m[i]/(1-self.beta1**self.t[i])
            self.v_hat[i]=self.v[i]/(1-self.beta2**self.t[i])
            p['param']-=self.lr*self.m_hat[i]/(self.v_hat[i]**0.5+self.epsilon)
            
            
# Layer Class, to personalize every layer
class Layer:
    def __init__(self,layer_config: LayerConfig):
        # Choose how to initialize w and b according to config
        self.in_dim=layer_config.in_dim
        self.out_dim=layer_config.out_dim
        self.activate_fn:Activation=layer_config.activation
        self.layer_config=layer_config
        
        #init weights and bias
        self.W=self._init_weights()
        self.b=self._init_bias()
        self.dW=np.zeros((self.in_dim,self.out_dim))
        self.db=np.zeros((1,self.out_dim))
        
            
    def get_params(self):
        # Used to expose parameters for the Optimizer to update
        return [
            {'param':self.W,'grad':self.dW},
            {'param':self.b,'grad':self.db}
        ]
        
            
    def _init_weights(self):
        method=self.layer_config.w_initializer
        if method == "xavier":
            # Mainly for Tanh, Sigmoid
            scale = np.sqrt(2 / (self.in_dim + self.out_dim))
            return np.random.randn(self.in_dim, self.out_dim) * scale
        elif method == "he":
            # Mainly for ReLU, LeakyReLU and GELU
            scale = np.sqrt(2 / self.in_dim)
            return np.random.randn(self.in_dim, self.out_dim) * scale
        elif method == "uniform":
            return np.random.uniform(-1, 1, size=(self.in_dim, self.out_dim))
        
    def _init_bias(self):
        method=self.layer_config.b_initializer
        if method=="zero":
            return np.zeros((1,self.out_dim))
        elif method=="constant":
            # a small constant to avoid neurons' death
            return np.full((1,self.out_dim),fill_value=0.01)
        elif method=='uniform':
            return np.random.uniform(-0.1,0.1,size=(1,self.out_dim))
    
    def shape(self) -> tuple:
        return (self.in_dim,self.out_dim)
    
    def forward(self,x):
        # x: [batch_size,in_dim], W: [in_dim,out_dim], b:[1,out_dim]
        cache={} # cache is used to store information for the backpropogation later
        z=np.dot(x,self.W)+self.b # numpy broadcast ensure there is no shape mismatch
        cache['z']=z
        a=self.activate_fn.forward(z)
        
        self.input=x
        self.output=a
            
        
        return a,cache
    
    def backward(self,x,do,cache:dict):
        # Back propogation according to different chosen Opimizer
        batch_size=x.shape[0]
        dz=do*self.activate_fn.derivative(cache['z']) #(batch_size,out_dim)
        self.db=np.sum(dz,axis=0) #(1,out_dim)
        self.dW=np.dot(x.T,dz)#(in_dim,out_dim)
        dx=np.dot(dz,self.W.T) #(batch_size,in_dim)
        
        # Call the optimizer
        optimizer_config=self.layer_config.optimizer
        if optimizer_config.name.lower()=='SGD'.lower():
            optimizer=SGD(params=self.get_params(),
                          learning_rate=optimizer_config.lr,
                          weight_decay=optimizer_config.wd)
        elif optimizer_config.name.lower()=='SGD_Momentum'.lower():
            optimizer=SGD_Momentum(params=self.get_params(),
                                   learning_rate=optimizer_config.lr,
                                   weight_decay=optimizer_config.wd,
                                   momentum=optimizer_config.momentum)
        elif optimizer_config.name.lower()=='Adam'.lower():
            optimizer=Adam(params=self.get_params(),
                           learning_rate=optimizer_config.lr,
                           weight_decay=optimizer_config.wd,
                           beta1=optimizer_config.beta1,
                           beta2=optimizer_config.beta2,
                           epsilon=optimizer_config.el)
        optimizer.step()
        return dx

from copy import deepcopy
from tqdm import tqdm
class MLP: # build uoon layers
    def __init__(self,layers,epochs,batch_size=16):
        self.layers=layers
        self.epoches=epochs
        self.batch_size=batch_size
        
        
    def forward(self,X,is_training=True):
        caches=[]
        output=deepcopy(X)
        for i,layer in enumerate(self.layers):
            if isinstance(layer,Layer):
                output,cache=layer.forward(output)
                dropout_rate=layer.layer_config.dropout
                if dropout_rate>0 and is_training and i!=len(self.layers)-1:
                    # apply dropout to this layer if it's training and not the output layer
                    mask=np.random.binomial(1,1-dropout_rate,size=output.shape)/(1-dropout_rate) # make sure divide by (1-dropout_rate) to maintain expectation
                    output*=mask
                    cache['mask']=mask
                    # print(f"i:{i} dropout:{dropout_rate}")
                caches.append(cache) # store in cache for back propogation later
            elif isinstance(layer,BatchNorm):
                if is_training:
                    output,cache=layer.forward(output,True)
                    caches.append(cache)
                else:
                    output=layer.forward(output,False)
                    
        return output,caches
                
                    
    
    def compute_loss(self,y_pred,y_true):
        batch_size=y_pred.shape[0] # y_pred is one-hot encoded
        if len(y_true.shape)>1:
            y_true=y_true.flatten()
        entropy_loss=np.sum(-np.log(y_pred[np.arange(batch_size),y_true]+1e-9))/batch_size
        return entropy_loss
    
    def calculate_f1_score(self,y_pred,y_true):
        if len(y_true.shape)>1:
            n_classes=y_true.shape[1]
            y_true=np.argmax(y_true,axis=1)
        else:
            n_classes=np.max(y_true)+1
        
        if len(y_pred.shape)>1:
            y_pred=np.argmax(y_pred,axis=1)
            
        precision=np.zeros(n_classes)
        recall=np.zeros(n_classes)
        f1_scores=np.zeros(n_classes)
        for i in range(n_classes):
            tp=np.sum((y_pred==i)&(y_true==i))
            fp=np.sum((y_pred==i)&(y_true!=i))
            fn=np.sum((y_pred!=i)&(y_true==i))
            precision[i]=tp/(tp+fp) if (tp+fp)>0 else 0
            recall[i]=tp/(tp+fn) if (tp+fn)>0 else 0
            f1_scores[i]=2*precision[i]*recall[i]/(precision[i]+recall[i]) if (precision[i]+recall[i])>0 else 0
        
        return np.mean(f1_scores)
        
    
    def backward(self,y_pred,y_true,caches):
        # Running backpropogation, collecting parameters from all Linear Layers
        batch_size=y_pred.shape[0]
        y_one_hot=np.zeros_like(y_pred)
        if len(y_true.shape)>1:
            y_true=y_true.flatten()
        y_one_hot[np.arange(batch_size),y_true]=1
        do=(y_pred-y_one_hot)/batch_size #loss should be divided by batch size
        for i in range(len(self.layers)-1,-1,-1):
            layer=self.layers[i]
            # print(f"i:{i}, layershape:{layer.shape()}")
            if isinstance(layer,Layer):
                if 'mask' in caches[i]: #apply dropout
                    mask=caches[i]['mask']
                    # print(f"do shape:{do.shape},mask shape:{mask.shape}")
                    do*=mask
                do=layer.backward(layer.input,do,caches[i])
            elif isinstance(layer,BatchNorm):
                do=layer.backward(do,caches[i])
    
    def train(self,X,y):
        num_samples=X.shape[0]
        num_batch=num_samples//self.batch_size
        
        #Initialize evaluation curve to store evaluation scores
        loss_curve=[]
        accuracy_curve=[]
        f1_curve=[]
        for epoch in tqdm(range(self.epoches),desc="Training Progess"):
            indices=np.random.permutation(num_samples)
            X_shuffled=X[indices]
            y_shuffled=y[indices]
            epoch_loss=0
            epoch_correct=0
            epoch_total=0
            all_y_true=[]
            all_y_pred=[]
            
            for i in range(num_batch):
                start=i*self.batch_size
                end=start+self.batch_size
                X_batch=X_shuffled[start:end]
                y_batch=y_shuffled[start:end]
                
                y_pred,caches=self.forward(X_batch,True)
                batch_loss=self.compute_loss(y_pred,y_batch)
                epoch_loss+=batch_loss
                predicted_classes=np.argmax(y_pred,axis=1)
                epoch_correct+=np.sum(predicted_classes==y_batch.flatten())
                epoch_total+=len(y_batch)
                all_y_true.append(y_batch)
                all_y_pred.append(predicted_classes)
                
                self.backward(y_pred,y_batch,caches)
            
            all_y_true=np.concatenate(all_y_true)
            all_y_pred=np.concatenate(all_y_pred)
            epoch_f1=self.calculate_f1_score(all_y_pred,all_y_true)
            epoch_loss/=num_batch
            epoch_accuracy=epoch_correct/epoch_total
            
            #Plotting
            loss_curve.append(epoch_loss)
            accuracy_curve.append(epoch_accuracy)
            f1_curve.append(epoch_f1)
            
            if epoch%5==0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
                print(f"Epoch {epoch}, Accuracy: {epoch_accuracy:.4f}")
                print(f"Epoch {epoch}, F1 Score: {epoch_f1:.4f}")
            
        return loss_curve,accuracy_curve,f1_curve
        
        
    def predict(self,X):
        y_pred,_=self.forward(X,is_training=False)
        return np.argmax(y_pred,axis=1)
    
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='MLP Model Training and Evaluation')
    
    # Data preprocessing parameters
    parser.add_argument('--pca_components', type=int, default=128, help='Number of PCA components')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    
    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'sgd_momentum', 'adam'], help='Optimizer type')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (for sgd_momentum)')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam optimizer')
    
    # Network structure parameters
    parser.add_argument('--hidden_layers', type=str, default='128,64,32', help='Hidden layer sizes, comma separated')
    parser.add_argument('--activation', type=str, default='gelu', 
                        choices=['relu', 'sigmoid', 'tanh', 'gelu', 'leaky_relu'], help='Activation function')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--use_batchnorm', action='store_true', help='Whether to use BatchNorm')
    parser.add_argument('--w_init', type=str, default='xavier', 
                        choices=['xavier', 'he', 'uniform'], help='Weight initialization method')
    parser.add_argument('--b_init', type=str, default='uniform', 
                        choices=['zero', 'constant', 'uniform'], help='Bias initialization method')
    
    args = parser.parse_args()
    
    train_data=np.load('Assignment1-Dataset/train_data.npy')
    train_label=np.load('Assignment1-Dataset/train_label.npy')
    test_data=np.load('Assignment1-Dataset/test_data.npy')
    test_label=np.load('Assignment1-Dataset/test_label.npy')
    
    # Set optimizer configuration
    optimizer_config = OptimizerConfig(
        name=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2
    )
    
    # Data preprocessing
    global pca_train_data, pca_test_data
    pca = PCA(n_components=args.pca_components, whiten=True, svd_solver='full')
    pca_train_data = pca.fit_transform(train_data)
    # std_test=StandardScaler()
    # test_data=std_test.fit_transform(test_data)
    pca_test_data = pca.transform(test_data) 
    
    # Build network layers
    hidden_sizes = [int(size) for size in args.hidden_layers.split(',')]
    layers = []
    
    # Input layer to first hidden layer
    layer_config = LayerConfig(
        in_dim=args.pca_components,
        out_dim=hidden_sizes[0],
        activation=Activation(args.activation),
        dropout=args.dropout,
        w_initializer=args.w_init,
        b_initializer=args.b_init,
        optimizer=optimizer_config
    )
    layers.append(Layer(layer_config))
    
    if args.use_batchnorm:
        layers.append(BatchNorm(hidden_sizes[0]))
    
    # Between hidden layers
    for i in range(len(hidden_sizes)-1):
        layer_config = LayerConfig(
            in_dim=hidden_sizes[i],
            out_dim=hidden_sizes[i+1],
            activation=Activation(args.activation),
            dropout=args.dropout,
            w_initializer=args.w_init,
            b_initializer=args.b_init,
            optimizer=optimizer_config
        )
        layers.append(Layer(layer_config))
        
        if args.use_batchnorm:
            layers.append(BatchNorm(hidden_sizes[i+1]))
    
    # Last hidden layer to output layer
    output_layer_config = LayerConfig(
        in_dim=hidden_sizes[-1],
        out_dim=10,  # 10 classes
        activation=Activation('softmax'),
        dropout=0.0,
        w_initializer=args.w_init,
        b_initializer=args.b_init,
        optimizer=optimizer_config
    )
    layers.append(Layer(output_layer_config))
    
    # Create and train model
    model = MLP(layers=layers, epochs=args.epochs, batch_size=args.batch_size)
    loss_curve, accuracy_curve, f1_curve = model.train(pca_train_data, train_label)
    
    # Visualize training process
    mpt.figure(figsize=(15, 5))
    mpt.subplot(1, 3, 1)
    mpt.plot(loss_curve, 'b-')
    mpt.title("Training Loss")
    mpt.xlabel('Epochs')
    mpt.ylabel('Loss')
    mpt.grid(True)
    
    mpt.subplot(1, 3, 2)
    mpt.plot(accuracy_curve, 'r-')
    mpt.title("Accuracy Rate")
    mpt.xlabel('Epochs')
    mpt.ylabel('Accuracy')
    mpt.grid(True)
    
    mpt.subplot(1, 3, 3)
    mpt.plot(f1_curve, 'g-')
    mpt.title("F1 Score")
    mpt.xlabel('Epochs')
    mpt.ylabel('F1 Score')
    mpt.grid(True)
    
    mpt.tight_layout()
    os.makedirs('results', exist_ok=True)
    
    # Save file with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mpt.savefig(f'results/training_metrics_{timestamp}.png', dpi=300)
    
    # Evaluate on test set
    prediction = model.predict(pca_test_data)
    accuracy = np.sum(prediction == test_label.flatten()) / prediction.shape[0]
    f1 = model.calculate_f1_score(prediction, test_label)
    
    print(f"Performance on test set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save training parameters to log
    with open(f'results/training_log_{timestamp}.txt', 'w') as f:
        f.write(f"Parameter configuration:\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
        f.write(f"\nTest set performance:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

if __name__=='__main__':
    main()


