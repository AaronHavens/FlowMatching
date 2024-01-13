from data_utils import *
from vector_fields import *
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm, trange
import seaborn as sns
import imageio.v2 as imageio
import os

# Apply the default theme
sns.set_theme()

def main():
    N = 10000
    state_dim = 2
    batch_size = 256
    epochs = 2000
    data_loader = get_circle_dataset(N, batch_size)

    vf = simpleVF(state_dim)

    loss_fn = nn.MSELoss()
    
    optimizer = torch.optim.Adam(vf.parameters(), lr=1e-3)

    def train(epoch):
        vf.train = True
        running_loss = 0
        for i, data in enumerate(data_loader):
            # Every data instance is an input + label pair
            Xi, Yi = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            Yi_hat = vf(Xi)

            # Compute the loss and its gradients
            loss = loss_fn(Yi_hat, Yi)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()

        return running_loss/i

    t = trange(epochs, desc='loss:', leave=True)
    for j in t:
        epoch_loss = train(j)
        t.set_description("loss: {}".format(epoch_loss))


    def simulate(x0, steps=50):
        def dxdt(y, t):
            y_ = torch.Tensor([[y[0], y[1], t]])
            return vf(y_).detach().numpy()[0]

        sol = odeint(dxdt, x0, np.linspace(0,1,steps))

        return sol


    def eval():
        vf.train=False
        nx, ny = (20, 20)
        steps = 50
        x = np.linspace(-1.0, 1.0, nx)
        y = np.linspace(-1.0, 1.0, ny)

        xv, yv = np.meshgrid(x, y, indexing='ij')
        trajs = np.zeros((nx,ny,steps,2))
        for i in range(nx):
            for j in range(ny):
                Xij = [xv[i,j], yv[i,j]]
                solij = simulate(Xij)
                #plt.plot(solij[:,0], solij[:,1],c='b')
                trajs[i,j,:,:] = solij[:,:]
        
        if not os.path.exists('./snapshots'):
            os.makedirs('./snapshots')

        for i in range(steps):
            fig, ax = plt.subplots(figsize=(10,10))
            ax.axis('equal')
            ax.set_xlim([-1.5,1.5])
            ax.set_ylim([-1.5,1.5])
            ax.scatter(trajs[:,:,i,0], trajs[:,:,i,1],c='m')
            ax.set_title('time: t={}'.format(i/steps))
            plt.savefig('./snapshots/step_{}.png'.format(i), bbox_inches='tight')
            plt.close()

        images = []

        for j in range(steps):
            filename = './snapshots/step_{}.png'.format(j)
            images.append(imageio.imread(filename))
        imageio.mimsave('./assets/circle_flow.gif', images,)
                
    def eval_vector_field():
        vf.train=False
        nx, ny = (300, 300)
        x = np.linspace(-1.5, 1.5, nx)
        y = np.linspace(-1.5, 1.5, ny)

        xv, yv = np.meshgrid(x, y, indexing='xy')
        UV = np.zeros((nx,ny,2))
        for i in range(nx):
            for j in range(ny):
                Xij = torch.Tensor([[xv[i,j], yv[i,j],1]])
                UV[i,j,:] = vf(Xij).detach().numpy()
       
        plt.figure() 
        plt.xlim([-1.5,1.5])
        plt.ylim([-1.5,1.5])
        plt.streamplot(xv, yv, UV[:,:,0], UV[:,:,1], density=2.0, linewidth=None, color='#A23BEC') 
        plt.show()

    eval()




if __name__ == "__main__":
    main()
