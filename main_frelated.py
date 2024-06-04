from data_utils import *
from vector_fields import *
from f_related_utils import *
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tqdm import tqdm, trange
import seaborn as sns
import imageio.v2 as imageio
import os

# Apply the default theme
sns.set_theme()

def cbf_loss(x, Yi_hat, Yi, MSE):
    loss = MSE(Yi_hat, Yi)

    return loss


def main():
    state_dim = 2
    batch_size = 64
    iterations = 50000
    train_mode = True
    scale = 2.0

    #F = Diffeomorpism(state_dim)
    vf = F_Related_VF(state_dim, n_layers=3, hid_dim=8)
    #vf = LipschitzVF(state_dim)
    #interp = BezierInterpolant3(state_dim)
    interp = LinearInterpolant()
    #interp = AddInterpolant(state_dim)
    mse_loss_fn = nn.MSELoss()
    #print(vf)
    #print(list(vf.parameters()))
    optimizer = torch.optim.Adam(list(vf.parameters())+list(interp.parameters()), lr=3e-4)

    def train(iteration):
        vf.train = True
        interp.train = True
        running_loss = 0
        optimizer.zero_grad()
        Xi, Yi = generate_circle_flow(interp,batch_size, embedding=None, use_t=True)

        #Xi, Yi = generate_grid_flow(interp, batch_size, scale=scale)
        Yi_hat = vf(Xi)

        loss = cbf_loss(Xi, Yi_hat, Yi, mse_loss_fn)
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

        torch.save(vf.state_dict(), './checkpoints/circle_F_related.pt')
    else:
        vf.load_state_dict(torch.load('./checkpoints/circle_F_related.pt'))

    def simulate(x0, steps=50):
        def dxdt(y, t):
            y_ = torch.Tensor([[y[0], y[1]]])
            return vf(y_).detach().numpy()[0]

        sol = odeint(dxdt, x0, np.linspace(0,1,steps))

        return sol


    def eval():
        vf.train=False
        steps = 50
        N = 1000
        trajs = np.zeros((N,steps,2))
        print('evaluating ...')
        t = np.linspace(0,1,steps)
        #Xi = torch.tensor(np.random.uniform(-0.2,0.2,size=(N,2))).float()
        Xi = torch.randn(size=(N,2))*0.2
        zeros = torch.zeros(N,1)
        Xi = torch.cat((Xi,zeros),axis=1)

        #for i in tqdm(range(N)):
            #Xi = np.random.normal(0,1,size=(2,))
            #x0 = torch.rand(size=(batch_size,2))*0.4 - 0.2
            #Xi = torch.tensor(np.random.uniform(-0.2,0.2,size=(1,2))).float()
        for j in range(steps):
            trajs[:,j,:] = vf.time_t_flow(Xi, t[j]).detach().numpy()
            #soli = simulate(Xi, steps=steps)
            #plt.plot(solij[:,0], solij[:,1],c='b')
            #trajs[i,:,:] = soli[:,:]
        
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

            #circle = plt.Circle((0., 0.), 1.0, color='black', fill=False, label='target')
            square = plt.Rectangle((-1.0, -1.0), 2, 2, edgecolor='black', facecolor='none', label='target')
            ax.add_patch(square)
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
        imageio.mimsave('./assets/circle_F_related.gif', images, loop=20)
                
    def eval_vector_field():
        vf.train=False
        nx, ny = (50, 50)
        x = np.linspace(-1.5, 1.5, nx)
        y = np.linspace(-1.5, 1.5, ny)
        steps = 100
        xv, yv = np.meshgrid(x, y, indexing='xy')
        ts = np.linspace(0,1,steps)
        for k, t in enumerate(ts):
            print(k)
            UV = np.zeros((nx,ny,2))
            for i in range(nx):
                for j in range(ny):
                    Xij = torch.Tensor([[xv[i,j], yv[i,j],t]])
                    UV[i,j,:] = vf(Xij).detach().numpy()
            fig, ax = plt.subplots(figsize=(6,6))#,dpi = 128)
            ax.axis('equal')
            ax.set_xlim([-1.5,1.5])
            ax.set_ylim([-1.5,1.5])
            #circle_cons = plt.Circle((0.,0.5), 0.25, color='red', fill=False, linestyle='-', label='constraint x^2 + (y-0.5)^2 >= 0.25^2')
            circle = plt.Circle((0., 0.), 1.0, color='black', fill=False, label='target')
            ax.add_patch(circle)
            ax.add_patch(circle_cons)
            ax.streamplot(xv, yv, UV[:,:,0], UV[:,:,1], density=2.0, linewidth=None, color='#A23BEC') 
            plt.savefig('./snapshots/vf_step_{}.png'.format(k))#, bbox_inches='tight')
            plt.close()

        images = []

        for k in range(steps):
            filename = './snapshots/vf_step_{}.png'.format(k)
            images.append(imageio.imread(filename))
            #print(images[j].shape)
        imageio.mimsave('./assets/vf_circle_flow_hole_const.gif', images,)

        plt.show()

    #eval_vector_field()
    eval()




if __name__ == "__main__":
    main()
