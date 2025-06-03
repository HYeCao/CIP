import os
import pickle
import numpy as np
import pandas as pd
import time
from causallearnmain.causallearn.search.FCMBased import lingam
import sys
from utilis.config import ARGConfig
import torch
import torch.nn.functional as F
import ipdb

def get_sa2r_weight(env, memory, agent, sample_size=5000, causal_method='DirectLiNGAM'):
    
    states, actions, rewards, next_states, dones = memory.sample(sample_size)
    rewards = np.squeeze(rewards[:sample_size]) 
    rewards = np.reshape(rewards, (sample_size, 1))
    X_ori = np.hstack((states[:sample_size,:], actions[:sample_size,:], rewards)) 
    X = pd.DataFrame(X_ori, columns=list(range(np.shape(X_ori)[1])))
    
    if causal_method == 'DirectLiNGAM':
        start_time = time.time()  
        model = lingam.DirectLiNGAM()
        model.fit(X)
        end_time = time.time()
        model._running_time = end_time - start_time
        weight_r = model.adjacency_matrix_[-1, np.shape(states)[1]:(np.shape(states)[1]+np.shape(actions)[1])]\

        weight_s_r = model.adjacency_matrix_[-1, 0:np.shape(states)[1]]

    #softmax weight_r
    weight = F.softmax(torch.Tensor(weight_r),0)
    weight = weight.numpy()   
    #* multiply by action size
    weight = weight * weight.shape[0]

    return weight, model._running_time


def get_sa2r_weight_cf(env, memory, agent, sample_size=5000, causal_method='DirectLiNGAM'):
    # Sample data from memory
    states, actions, rewards, next_states, dones = memory.sample(sample_size)

    # Prepare data for causal model
    rewards = np.squeeze(rewards[:sample_size])
    rewards = np.reshape(rewards, (sample_size, 1))
    X_ori = np.hstack((states[:sample_size, :], actions[:sample_size, :], rewards))
    X = pd.DataFrame(X_ori, columns=list(range(np.shape(X_ori)[1])))

    Z_ori = np.hstack((states[:sample_size, :], actions[:sample_size, :], next_states[:sample_size, :]))
    Z = pd.DataFrame(Z_ori, columns=list(range(np.shape(Z_ori)[1])))

    if causal_method == 'DirectLiNGAM':
        start_time = time.time()
        model = lingam.DirectLiNGAM()
        model.fit(X)
        end_time = time.time()
        model._running_time = end_time - start_time

        # Extract weights for causal effect
        weight_r = model.adjacency_matrix_[-1, np.shape(states)[1]:(np.shape(states)[1] + np.shape(actions)[1])]
        weight_s_r = model.adjacency_matrix_[-1, 0:np.shape(states)[1]]

        model2 = lingam.DirectLiNGAM()
        model2.fit(X)
        weight_s_s = model.adjacency_matrix_[-1, 0:np.shape(next_states)[1]]

    # Softmax weight_r
    weight = F.softmax(torch.Tensor(weight_r), 0)
    weight = weight.numpy()
    weight = weight * weight.shape[0]


    weight_s = F.softmax(torch.Tensor(weight_s_s), 0)
    weight_s = weight_s.numpy()
    weight_s = weight_s * weight_s.shape[0]


    def counterfactual_data_augmentation(states, actions, rewards, next_states, dones, weight_s):
        # Resample two batches of states
        batch_size = states.shape[0] // 2
        states_batch1 = states[:batch_size]
        states_batch2 = states[batch_size:2 * batch_size]

        # Corresponding actions, rewards, next states, and dones
        actions_batch1 = actions[:batch_size]
        actions_batch2 = actions[batch_size:2 * batch_size]
        rewards_batch1 = rewards[:batch_size]
        rewards_batch2 = rewards[batch_size:2 * batch_size]
        next_states_batch1 = next_states[:batch_size]
        next_states_batch2 = next_states[batch_size:2 * batch_size]
        dones_batch1 = dones[:batch_size]
        dones_batch2 = dones[batch_size:2 * batch_size]

        min_indices_batch1 = np.argmin(weight_s * states_batch1, axis=1)
        min_indices_batch2 = np.argmin(weight_s * states_batch2, axis=1)

        augmented_states_batch1 = np.copy(states_batch1)
        augmented_states_batch2 = np.copy(states_batch2)
        augmented_next_states_batch1 = np.copy(next_states_batch1)
        augmented_next_states_batch2 = np.copy(next_states_batch2)

        for i in range(batch_size):
            augmented_states_batch1[i, min_indices_batch1[i]], augmented_states_batch2[i, min_indices_batch2[i]] = (
                states_batch2[i, min_indices_batch2[i]],
                states_batch1[i, min_indices_batch1[i]]
            )

            augmented_next_states_batch1[i, min_indices_batch1[i]], augmented_next_states_batch2[
                i, min_indices_batch2[i]] = (
                next_states_batch2[i, min_indices_batch2[i]],
                next_states_batch1[i, min_indices_batch1[i]]
            )

        augmented_states = np.vstack((augmented_states_batch1, augmented_states_batch2))
        augmented_next_states = np.vstack((augmented_next_states_batch1, augmented_next_states_batch2))
        augmented_actions = np.vstack((actions_batch1, actions_batch2))
        augmented_rewards = np.vstack((rewards_batch1, rewards_batch2))
        augmented_dones = np.hstack((dones_batch1, dones_batch2))

        return augmented_states, augmented_actions, augmented_rewards, augmented_next_states, augmented_dones

    # Apply counterfactual data augmentation
    augmented_states, augmented_actions, augmented_rewards, augmented_next_states, augmented_dones = counterfactual_data_augmentation(
        states, actions, rewards, next_states, dones, weight_s
    )

    return (weight, model._running_time, weight_s, augmented_states,
            augmented_actions, augmented_rewards, augmented_next_states,augmented_dones)
