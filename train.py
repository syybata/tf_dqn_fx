import numpy as np
import tensorflow as tf

from catch_ball import CatchBall
from dqn_agent import DQNAgent


if __name__ == "__main__":
    # parameters
    n_epochs = 1000

    # environment, agent
    env = CatchBall()
    agent = DQNAgent(env.enable_actions, env.name)

    # variables
    win = 0

    ### TENSORBOARD
    Q_max_ph = tf.placeholder(tf.float32)
    loss_ph = tf.placeholder(tf.float32)
    
    # for mining
    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list="1"
        )
    )
    sess=tf.Session(config=config)
    tf.global_variables_initializer().run(session=sess)
    
    tf.summary.scalar('Q_max', Q_max_ph)
    tf.summary.scalar('loss', loss_ph)
    summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('shiba_train', sess.graph)


    for e in range(n_epochs):
        # reset
        frame = 0
        loss = 0.0
        Q_max = 0.0
        env.reset()
        state_t_1, reward_t, terminal = env.observe()

        while not terminal:
            state_t = state_t_1

            # execute action in environment
            action_t = agent.select_action(state_t, agent.exploration)
            env.execute_action(action_t)

            # observe environment
            state_t_1, reward_t, terminal = env.observe()

            # store experience
            agent.store_experience(state_t, action_t, reward_t, state_t_1, terminal)

            # experience replay
            agent.experience_replay()

            # for log
            frame += 1
            loss += agent.current_loss
            Q_max += np.max(agent.Q_values(state_t))
            #if reward_t == 1:
            if reward_t > 0:
                win += 1


        print("epoch: {:03d}/{:03d}, win: {:03d}, loss: {:.4f}, Q_max: {:.4f}, reward: {:.2f}".format(
            e, n_epochs - 1, win, loss / frame, Q_max / frame, reward_t))

        ### TENSORBOARD
        summ = sess.run(summaries, {loss_ph: loss/frame, Q_max_ph: Q_max/frame})
        train_writer.add_summary(summ, e)


    # save model
    agent.save_model()

