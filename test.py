import sys
import argparse
from visdom import Visdom
import matplotlib.pyplot as plt


import numpy as np

import torch
from torch.autograd import Variable

from dataset.dataset import *
from utility.utils import *
from model import *

from domains.gridworld import *
from generators.obstacle_gen import *

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def main(config, n_domains=10, max_obs=30, 
            max_obs_size=None, n_traj=10, n_actions=8):
    record = False
    # Correct vs total:
    correct, total = 0.0, 0.0
    # Automatic swith of GPU mode if available
    use_GPU = torch.cuda.is_available()
    vin = torch.load(config.weights)
    if use_GPU:
         vin = vin.cuda()

    for dom in range(n_domains):
        # Randomly select goal position
        goal = [np.random.randint(config.imsize),
                np.random.randint(config.imsize)]
        # Generate obstacle map
        obs = obstacles([config.imsize, config.imsize], goal, max_obs_size)
        # Add obstacles to map
        n_obs = obs.add_n_rand_obs(max_obs)
        # Add border to map
        border_res = obs.add_border() 
        # Ensure we have valid map
        if n_obs == 0 or not border_res:
            continue
        # Get final map
        im = obs.get_final()

        # Generate gridworld from obstacle map
        G = gridworld(im, goal[0], goal[1])

        g_dense = G.W
        g_masked = np.ma.masked_values(g_dense, 0)
        g_sparse =  csr_matrix(g_dense)
        # Get value prior
        value_prior = G.get_reward_prior()
        # Sample random trajectories to our goal
        value, stateO, states_xy, states_one_hot = sample_trajectory(G, n_traj)
        stateO1=[]
   
        
#        print("states")
#        print(states_xy)
        for i in range(n_traj):
            if len(states_xy[i]) > 1:
 #               print("begin==================================")
 #               print("==========================================")
 #               print("======================================")
 #               print("state")
 #               print(states_xy[i])
 #               print("O")
 #               print(stateO[i])
                # Get number of steps to goal


                L = len(states_xy[i]) * 2
                # Allocate space for predicted steps
                pred_traj = np.zeros((L, 2))


                o_traj = np.zeros((L, 2))
                # Set starting position
                pred_traj[0,:] = states_xy[i][0,:]
                o_traj[0, :] = stateO[i][0,:]
                o = G.map_ind_to_state(o_traj[0, 0], o_traj[0, 1])
                
                pathO = []
                for j in range(1, L):
                    # Transform current state data 
                    state_data = pred_traj[j-1, :]
                    state_data = state_data.astype(np.int)
                    stateO1.append(state_data)
                    s = G.map_ind_to_state(pred_traj[j-1, 0], pred_traj[j-1, 1])
#                    stateO_data = o_traj[j-1, :]
                    if  j <= len(stateO[i]):
                        stateO_data = stateO[i][j-1, :]
                        stateO_data = stateO_data.astype(np.int)
  #                  print("state_data")
  #                  print(state_data)
#                    print("g_sparse")
#                    print(g_sparse)
#                    print("s")
#                    print(s)
                    d, pred1 = dijkstra(g_sparse, s, return_predecessors =True)
                    path1 = trace_path(pred1, s, o)
                    path1 = np.flip(path1, 0)
 #                   print("path1")
 #                   print(path1)
                        
                    pathO.append(o)





                






                     
                    
                   # if j <= len(stateO[i]):
                    #    stateO_data = stateO[i][j-1]
                    
                        
                    #stateO_data = stateO_data.astype(np.int)
   #                 print("odata")
   #                 print(stateO_data)
                    # Transform domain to Networks expected input shape
                    im_data = G.image.astype(np.int)
    #                print ("im_data")
    #                print (im_data)
                    im_data = 1 - im_data
                    im_data = im_data.reshape(1, 1, config.imsize, config.imsize)
                    # Transfrom value prior to Networks expected input shape
#                    value_data = value_prior.astype(np.int)
 #                   print("pvalue")
#                    print(value_data)
                     
                 #   value_data = G.get_reward_prior()
                 #   r0, c0 = G.get_coords(o)
                 #   r, c = G.neighbors(r0, c0)

                 #   value_data[r, c] = -3
                 #   value_data[r0, c0] = -10
                 #   value_data =value_data.astype(np.int)
                    if j <= len(value[i]):
                        value_data = value[i][j-1].astype(np.int)
     #               print ("value")
      #              print(value_data)
                    value_data = value_data.reshape(1, 1, config.imsize, config.imsize)
                    # Get inputs as expected by network
                    X_in = torch.from_numpy(np.append(im_data, value_data, axis=1)).float()
                    S1_in = torch.from_numpy(state_data[0].reshape([1, 1])).float()
                    S2_in = torch.from_numpy(state_data[1].reshape([1, 1])).float()
                    O1_in = torch.from_numpy(stateO_data[0].reshape([1, 1])).float()
                    O2_in = torch.from_numpy(stateO_data[1].reshape([1, 1])).float()
                    
                    
       #             print("current position")
       #             print(S1_in, S2_in)
                    # Send Tensors to GPU if available
                    if use_GPU:
                        X_in = X_in.cuda()
                        S1_in = S1_in.cuda()
                        S2_in = S2_in.cuda()
                        O1_in = O1_in.cuda()
                        O2_in = O2_in.cuda()                     
                    # Wrap to autograd.Variable
                        X_in, S1_in, S2_in, O1_in, O2_in = Variable(X_in), Variable(S1_in), Variable(S2_in), Variable(O1_in), Variable(O2_in)
                    # Forward pass in our neural net
        #            print("input")
        #            print(X_in)
                    if record == True:
                        _, predictions = vin(X_in, S1_in, S2_in, O1_in, O2_in, config, record_images=False)
                        imgs = np.concatenate([vin.grid_image] + [vin.reward_image] + vin.value_images)
                        np.savez_compressed('learned_rewards_values1_{:d}x{:d}'.format(config.imsize, config.imsize), imgs)
                        record = False
                    #print("from ")
                    #print(X_in.shape)
                    _, predictions = vin(X_in, S1_in, S2_in, O1_in, O2_in, config)                        
                    _, indices = torch.max(predictions.cpu(), 1, keepdim=True)
                    a = indices.data.numpy()[0][0]
                    # Transform prediction to indices

                    ns = G.sample_next_state(s, a)
                    nr, nc = G.get_coords(ns)
                    pred_traj[j, 0] = nr
                    pred_traj[j, 1] = nc

                    
                    
 #                   if len(path1) >= 2:
 #                       print("update o")
 #                       print(path1[1])
 #                       o = path1[1]
 #                   elif j < L-2 and len(path1) == 1:



                         #pathO[j-1:] = [o] * (1)
#                        states_xy[i] = states_xy[i][:j-1,:]
#                        break

#                   elif j < L-2 and len(path1) > 1:
 #                       r0, c0 = G.get_coords(path1[0])
  #                      r, c = G.get_coords(path1[1])
   #                     st = G.escape(r,c,r0,c0)
    #                    nr, nc = G.get_coords(st)
     #                   states_xy[i][j,:] = np.hstack((nr, nc))
      #                  print("Escape")

                    orow, oc = G.get_coords(o)
                    o_traj[j, 0] = orow
                    o_traj[j, 1] = oc
                    
                    if nr == goal[0] and nc == goal[1]:
                        # We hit goal so fill remaining steps
                        pred_traj[j+1:,0] = nr
                        pred_traj[j+1:,1] = nc
                        break
                # Plot optimal and predicted path (also start, end)
                if pred_traj[-1, 0] != goal[0] and pred_traj[-1, 1] != goal[1]:
                    correct += 1
                total += 1
#                print("===================================")
#                print("state")
#                print(states_xy[i])
#                print("stateO")
#                print(stateO[i])
#                print("O")
#                print(o_traj)
                if config.plot == True:
                    visualize(G.image.T, states_xy[i], stateO[i], pred_traj, goal)
        sys.stdout.write("\r" + str(int((float(dom)/n_domains) * 100.0)) + "%")
        sys.stdout.flush()
    sys.stdout.write("\n")
    print('Rollout Accuracy: {:.2f}%'.format(100*(correct/total)))


def draw(dom, states_xy, stateO, pred_traj):
    vis = Visdom()
#    vis.surf(X=dom, opts=dict(colormap='Hot'))
#    vis.line(X= states_xy[:,0], Y= states_xy[:,1])
    x1 = states_xy[:,0].tolist
    y1 = states_xy[:,1].tolist
    trace = dict(x=x1, y=y1, mode="markers+lines", type='custom',
             marker={'color': 'red', 'symbol': 104, 'size': "10"},
             text=["one", "two", "three"], name='1st Trace')
    layout = dict(title="First Plot", xaxis={'title': 'x1'}, yaxis={'title': 'x2'})

    vis._send({'data': [trace], 'layout': layout, 'win': 'mywin'})
    
def visualize(dom, states_xy, stateO, pred_traj, goal):
    fig, ax = plt.subplots()
    implot = plt.imshow(dom, cmap="Greys_r")
    ax.plot(states_xy[:,0], states_xy[:,1], c='b', label='Optimal Path')
    ax.plot(pred_traj[:,0], pred_traj[:,1], '-X', c='r', label='Predicted Path')
    ax.plot(stateO[:,0], stateO[:,1], 'c--', label='Oponent Path')
    ax.plot(states_xy[0,0], states_xy[0,1], '-o', label='Start')
    ax.plot(stateO[0,0], stateO[0,1], '-p', label='O Start')
    ax.plot(goal[0], goal[1], '-s', label='Goal')
    legend = ax.legend(loc='upper right', shadow=False)
    for label in legend.get_texts():
        label.set_fontsize('x-small') # the legend text size
    for label in legend.get_lines():
        label.set_linewidth(0.5)  # the legend line width
    #plt.draw()
    vis = Visdom()
    vis.matplot(plt)
    
 #   plt.waitforbuttonpress(0)
    plt.close(fig)


if __name__ == '__main__':
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', 
                        type=str, 
                        default='trained/vin_8x8.pth', 
                        help='Path to trained weights')
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--imsize', 
                        type=int, 
                        default=8, 
                        help='Size of image')
    parser.add_argument('--k', 
                        type=int, 
                        default=10, 
                        help='Number of Value Iterations')
    parser.add_argument('--l_i', 
                        type=int, 
                        default=2, 
                        help='Number of channels in input layer')
    parser.add_argument('--l_h', 
                        type=int, 
                        default=150, 
                        help='Number of channels in first hidden layer')
    parser.add_argument('--l_q', 
                        type=int, 
                        default=10, 
                        help='Number of channels in q layer (~actions) in VI-module')
    config = parser.parse_args()
    # Compute Paths generated by network and plot
    main(config)
