from operator import add, and_
import numpy as np
import utils
import config as conf
from functools import reduce
import math
import matplotlib.pyplot as plt

def get_means_over_window(vals, window_size):
    means = []
    for i in range(len(vals)):
        if i < window_size - 1:
            means.append(np.mean(vals[0:(i+1)]))
        else:
            means.append(np.mean(vals[(i - window_size + 1):(i+1)]))
    return means


def plot_vals(vals, title, xlabel, ylabel, filename, plot_mean=False, mean_window=None):
    x_vals = [i for i in range(len(vals))]
    plt.figure()
    plt.plot(x_vals, vals)
    if plot_mean and mean_window is not None:
        plt.plot(x_vals, get_means_over_window(vals, mean_window), 'r-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig('plots/' + str(filename))


def plot_vals_with_mean(vals, mean_vals, plot_name, title, xlabel, ylabel):
    x_vals = [i for i in range(len(vals))]
    plt.figure()
    plt.plot(x_vals, vals, 'c-', x_vals, mean_vals, 'r-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig('plots/' + str(plot_name))


def plot_training_episode_avg_loss(training_losses, model_name):
    x_vals = [i for i in range(len(training_losses))]
    plt.figure()
    plt.plot(x_vals, training_losses)
    plt.title('Mean Loss per Training Episode')
    plt.xlabel('episode')
    plt.ylabel('loss')
    plt.savefig('plots/' + str(model_name) + '_training_loss_plot.png')


def plot_episode_rewards_with_mean(episode_rewards, mean_rewards, model_name):
    plot_vals_with_mean(episode_rewards, mean_rewards,
                       str(model_name) + '_reward_plot.png',
                       'Reward per Training Episode', 'episode', 'reward')


def plot_episode_scores_with_mean(episode_scores, mean_scores, model_name):
    plot_vals_with_mean(episode_scores, mean_scores,
                    str(model_name) + '_score_plot.png',
                    'Score per Training Episode', 'episode', 'score')


def plot_episode_steps_survived_with_mean(episode_steps_survived, mean_steps_survived, model_name):
    plot_vals_with_mean(episode_steps_survived, mean_steps_survived,
                        str(model_name) + '_steps_survived_plot.png',
                        'Steps Survived per Training Episode', 'epsiode', 'steps survived')


def get_epsilon_decay_factor(e_max, e_min, e_decay_window):
    return math.exp(math.log(e_min / e_max) / e_decay_window)



def select_model_actions(models, state):
    model_actions = []
    for model in models:
        model_actions.append(model.get_action(state))
    return model_actions


def optimize_models(models):
    for model in models:
        model.optimize()


def update_models_memory(models, state, actions, next_state, rewards, dones):
    for (model, action, reward, done) in zip(models, actions, rewards, dones):
        model.remember(state, action, next_state, reward, done)


def train_models(env, models, episodes=10, steps=2500, print_every=200, model_name="train_drl", mean_reward_window=10, target_update=10):
    print("\nTRAIN MODE")

    training_losses = []
    training_rewards = []
    mean_rewards = []

    model = models[0]
    
    for episode in range(episodes):
        print('=== Starting Episode %s ===' % episode)

        # done = False  # whether game is done or not (terminal state)
        # reset the environment to fresh starting state with game agents initialized for models
        episode_rewards = [0 for _ in models]
        episode_loss = []

        for model in models:
            model.done = False
            model.eval = False

        env.reset(models)
        state = env.get_state()  # get the first state

        for step in range(steps):  # cap the num of game ticks
            actions = select_model_actions(models, state)

            # environment determines new state, reward, whether terminal, based on actions taken by all models
            rewards, dones = env.update_game_state(models, actions)
            next_state = env.get_state()

            episode_rewards = list(
                map(add, episode_rewards, rewards))  # update rewards
            update_models_memory(models, state, actions, next_state,
                                 rewards, dones)  # update replay memory

            # optimize models
            loss = model.optimize()
            if loss is not None:
                episode_loss.append(loss)

            # check for termination of our player #TODO
            if dones[0]:
                break
            # terminate if all other players are dead
            if (len(dones) > 1):
                if reduce(and_, dones[1:]):
                    break

            state = next_state  # update the state

            if step % print_every == 0 and step != 0:
                print("----STEP %s rewards----" % step)
                for idx, model in enumerate(models):
                    print("Model %s: %s" % (model.id, episode_rewards[idx]))
        # print("------EPISODE %s rewards------" % episode)
        # for idx, model in enumerate(models):
        #     print("Model %s: %s" % (model.id, episode_rewards[idx]))

        # decay epsilon
        if model.learning_start:
            epsilon = models[0].decay_epsilon()
            # print("epsilon after decay: ", epsilon)

        #sync target net with policy
        if episode % target_update == 0:
            model.sync_target_net()

        training_losses.append(np.mean(episode_loss))
        training_rewards.append(episode_rewards[0])
        mean_reward = np.mean(training_rewards[-mean_reward_window:])
        mean_rewards.append(mean_reward)

        print('Mean Episode Loss: {:.4f} | Episode Reward: {:.4f} | Mean Reward: {:.4f}'.format(np.mean(episode_loss), episode_rewards[0], mean_reward))

    plot_training_episode_avg_loss(training_losses, model_name)
    plot_episode_rewards_and_mean(training_rewards, mean_rewards, model_name)
    plt.show()

def test_models(env, models, steps=2500, print_every=200):
    print("\nTEST MODE")
    episode_rewards = [0 for _ in models]
    for model in models:
        model.done = False
        model.eval = True

    env.reset(models)
    state = env.get_state()  # get the first state

    for step in range(steps):  # cap the num of game ticks
        actions = select_model_actions(models, state)

        # environment determines new state, reward, whether terminal, based on actions taken by all models
        rewards, dones = env.update_game_state(models, actions)

        # TODO: update dones for other models, persist (otherwise negative rewards)

        next_state = env.get_state()

        episode_rewards = list(
            map(add, episode_rewards, rewards))  # update rewards

        # check for termination of our player #TODO
        if dones[0]:
            break

        state = next_state  # update the state

        if step % print_every == 0:
            print("----STEP %s rewards----" % step)
            for idx, model in enumerate(models):
                print("Model %s: %s" % (model.id, episode_rewards[idx]))
    print("------TEST rewards------")
    for idx, model in enumerate(models):
        print("Model %s: %s" % (model.id, episode_rewards[idx]))
