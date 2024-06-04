import torch
from models.vit import ViT
from controllable_utils import *
from data_utils import *
from tqdm import tqdm, trange
import os
import imageio.v2 as imageio


def mse_loss(Yi_hat, Yi, MSE):
    loss = MSE(Yi_hat, Yi)

    return loss


def main():
    state_dim = 2
    batch_size = 1
    iterations = 50000
    train_mode = False

    state_dim = 2
    num_vector_fields = 2
    time_steps = 50
    depth = 3
    heads = 1
    mlp_dim = 32
    scale = 2.0
    vf = ViT(
        state_dim=state_dim,
        num_vector_fields=num_vector_fields, 
        time_steps=time_steps, 
        depth=depth, 
        heads=heads, 
        mlp_dim=mlp_dim, 
        dim_head = 32, 
        dropout = 0., 
        emb_dropout = 0.)


    interp = LinearInterpolantDT()
    #interp = AddInterpolant(state_dim)
    mse_loss_fn = nn.MSELoss()
    #print(vf)
    #print(list(vf.parameters()))
    optimizer = torch.optim.Adam(list(vf.parameters())+list(interp.parameters()), lr=3e-4)

    def train(iteration):
        vf.train = True
        running_loss = 0
        optimizer.zero_grad()
        Xi, Yi = generate_circle_flow_DT(interp, batch_size, time_steps=time_steps)
        #print('target',Yi)
        Yi_hat = vf(Xi)
        #print('predicted',Yi_hat)        
        loss = mse_loss(Yi_hat, Yi, mse_loss_fn)
        loss.backward()
        #print(vf.F.grad)
        optimizer.step()

        running_loss += loss.item()

        return running_loss/batch_size

    if train_mode:
        t = trange(iterations, desc='loss:', leave=True)
        for j in t:
            epoch_loss = train(j)
            t.set_description("loss: {}".format(epoch_loss))

        torch.save(vf.state_dict(), './checkpoints/circle_controllable.pt')
    else:
        vf.load_state_dict(torch.load('./checkpoints/circle_controllable.pt'))


    def eval():
        vf.train=False
        steps = time_steps
        N = 1000
        print('evaluating ...')
        t = np.linspace(0,1,steps)
        Xi = torch.rand(size=(N,2))*0.4 - 0.2
        V = vf(Xi)*1/(steps+1)
        V = torch.concat((Xi[:,None,:], V),axis=1)
        #print(V)
        trajs = torch.cumsum(V, dim=1).detach().numpy()
        #print(trajs)
        
        if not os.path.exists('./snapshots'):
            os.makedirs('./snapshots')
        print('plotting ...')
        for i in tqdm(range(steps)):
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
            ax.set_title('time: t={}'.format((i+1)/steps))
            plt.savefig('./snapshots/step_{}.png'.format(i))#, bbox_inches='tight')
            plt.close()

        images = []

        for j in range(steps):
            filename = './snapshots/step_{}.png'.format(j)
            images.append(imageio.imread(filename))
            #print(images[j].shape)
        imageio.mimsave('./assets/circle_controllable.gif', images, loop=20)

    eval()
                
    




if __name__ == "__main__":
    main()

