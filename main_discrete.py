from data_utils import *
from vector_fields import *
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm, trange
import seaborn as sns
import imageio.v2 as imageio
import os
from time import time
from collections import deque

# Apply the default theme
sns.set_theme()

def cbf_loss(x, Yi_hat, Yi, MSE):
    loss = MSE(Yi_hat, Yi)


    return loss


def main():
    state_dim = 2
    batch_size = 256
    iterations = 500000
    train_mode = False
    time_steps = 20
    scale = 2.0
    model_load_name = 'circle_discrete'
    model_save_name = 'circle_discrete'
    vis_save_name = 'circle_discrete_odeint_step'

    vf = DiscreteSimpleVF(state_dim, time_steps=time_steps)

    interp = LinearInterpolant()
    #interp = AddInterpolant(state_dim)
    mse_loss_fn = nn.MSELoss()
    
    optimizer = torch.optim.Adam(list(vf.parameters())+list(interp.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations)

    def train(iteration):
        vf.train = True
        interp.train = True
        running_loss = 0
        optimizer.zero_grad()
        Xi, Yi, t_index = generate_circle_flow_DT(interp, batch_size=batch_size, time_steps=time_steps)

        #Xi, Yi, t_index = generate_grid_flow(interp, batch_size, scale=scale)
        Yi_hat = vf(Xi, t_index)

        loss = cbf_loss(Xi, Yi_hat, Yi, mse_loss_fn)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        return running_loss/batch_size

    if train_mode:
        q = deque(maxlen=2000)
        t = trange(iterations, desc='loss:', leave=True)
        for j in t:
            epoch_loss = train(j)
            q.append(epoch_loss)
            t.set_description("loss: {:.3E}, lr: {:.3E}".format(np.mean(q), scheduler.get_last_lr()[0]))
            scheduler.step()

        torch.save(vf.state_dict(), './checkpoints/{}.pt'.format(model_save_name))
    else:
        vf.load_state_dict(torch.load('./checkpoints/{}.pt'.format(model_load_name)))


    def eval():
        vf.train=False
        steps = time_steps
        N = 50000
        print('evaluating ...')
        t1 = time()
        t = np.linspace(0,1,steps)
        Xi = torch.randn(size=(N,2))
        trajs = np.zeros((N,steps+1,2))
        trajs[:, 0, :] = Xi.detach().numpy()
        for j in range(steps):
            Xi = vf.odeint_integrate(Xi, j)
            trajs[:,j+1,:] = Xi.detach().numpy()
        t2 = time()
        print('eval time: {} per/sample'.format((t2-t1)/N))
        
        if not os.path.exists('./snapshots'):
            os.makedirs('./snapshots')
        print('plotting ...')
        for i in tqdm(range(steps+1)):
            fig, ax = plt.subplots(figsize=(6,6))#,dpi = 128)
            ax.axis('equal')
            ax.set_xlim([-1.5,1.5])
            ax.set_ylim([-1.5,1.5])
            major_ticks = np.arange(-scale*0.5, scale*0.5 + 0.2*scale, 0.2*scale)
            ax.set_xticks(major_ticks)
            ax.set_yticks(major_ticks)

            # And a corresponding grid
            #ax.grid(which='both')
            #ax.grid(color='black', linestyle='-', linewidth=2)


            #ax.plot([-0.5,-0.5],[-1.5, 1.5],c='r', linestyle='-')
            #ax.plot([0.5,0.5],[-1.5, 1.5],c='r', linestyle='-', label='constraint |x| <= 1/2')
            #circle_cons_1 = plt.Circle((0.,0.5), 0.25, color='red', fill=False, linestyle='-', label='constraint x^2 + (y +/- 0.5)^2 >= 0.25^2')
            #circle_cons_2 = plt.Circle((0.,-0.5), 0.25, color='red', fill=False, linestyle='-')

            circle = plt.Circle((0., 0.), 1.0, color='black', fill=False, label='target')
            #square = plt.Rectangle((-1.0, -1.0), 2, 2, edgecolor='black', facecolor='none', label='target')
            ax.add_patch(circle)
            #ax.add_patch(circle_cons_1)
            #ax.add_patch(circle_cons_2)

            ax.scatter(trajs[:,i,0], trajs[:,i,1],c='m', s=5)
            #ax.legend(loc=1)
            ax.set_title('time: t={}'.format((i)/(steps)))
            plt.savefig('./snapshots/step_{}.png'.format(i))#, bbox_inches='tight')
            plt.close()

        images = []

        for j in range(steps+1):
            filename = './snapshots/step_{}.png'.format(j)
            images.append(imageio.imread(filename))
            #print(images[j].shape)
        imageio.mimsave('./assets/{}.gif'.format(vis_save_name), images, loop=20)

    eval()




if __name__ == "__main__":
    main()
