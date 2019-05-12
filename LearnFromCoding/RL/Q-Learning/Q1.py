import pandas as pd
import numpy as np
import time
import random


'''
-o---T 宝藏探险者
T 就是宝藏的位置, o 是探索者的位置
例子的环境是一个一维世界，在世界的右边有宝藏，探索者只要得到宝藏尝到了甜头
然后以后就记住了得到宝藏的方法，这就是他用强化学习所学习到的行为。
Q-learning 是一种记录行为值 (Q value) 的方法
每种在一定状态的行为都会有一个值 Q(s, a)
就是说 行为 a 在 s 状态的值是 Q(s, a)
s 在上面的探索者游戏中，就是 o 所在的地点了
而每一个地点探索者都能做出两个行为 left/right，这就是探索者的所有可行的 a 啦。
'''

# Create all the possible states and actions
states = range(5)
actions = ['left', 'right']

# Create our enviroment
def update_env(state):

    global states
    env = list('----T')

    if state in states:
        if state != states[-1]:
            env[state] = 'o'
        print('{}'.format(''.join(env)))
    else:
        print('Unknown Area you walked in')


def get_next_state(state, action):

    # In current state, we take the one action
    global actions
    global states

    # After taking actions, we can get new next state
    # The action to this affect the next state

    # if this action is to go left
    if action == 'left' and state != states[0]:
        next_state = state - 1

    # if this action is to go right
    elif action == 'right' and state != states[-1]:
        next_state = state + 1
    
    # No action did, so stay in here
    else:
        next_state = state


    return next_state


def get_valid_actions(state):
    # when we at zero state, we cannot go left
    # when we find Gold, we cannot go right

    global actions # ['left', 'right']
    
    valid_actions = set(actions)

    if state == states[-1]:             
        valid_actions -= set(['right'])
    if state == states[0]:  
        valid_actions -= set(['left'])
    
    return list(valid_actions)

reward = [0,0.001,0.001,0.1,1]

q_table = pd.DataFrame(data=[[0 for _ in actions] for _ in states],
                       index=states, columns=actions)


def RL():

    # 为Q learning 设置参数

    epsilon = 0.9   # 贪婪度 greedy


    alpha = 0.1     # 学习率


    gamma = 0.8     # 奖励递减值

    global rewards
    global states
    global actions

    for i in range(4):

        # 随机初始化一个开始的状态
        # 在第一个位置上
        # current_state = 1
        current_state = random.choice(states)

        # 在我们的环境中把主人公放上
        update_env(current_state)

        # 记录一下一共尝试了多少步
        total_steps = 0

        # 给定一个终止学习的标准
        while current_state != states[-1]:

            # 可视化我们得Q-table
            print('\nq_table:')
            print(q_table)

            # 选择动作 贪婪
            if (random.uniform(0,1) > epsilon) or ((q_table.ix[current_state] == 0).all()):
                current_action = random.choice(get_valid_actions(current_state))
            else:
                current_action = q_table.ix[current_state].idxmax()

            # 目前的状态->动作行为发生->得到新的一个状态
            # 这里的话普通编程就能实现，没有`学习`的过程
            # 通过动作在目前的状态上获得下一个状态
            # 因为根据Qtable我们已经获得了value，强化学习就是通过这个value进行学习
            next_state = get_next_state(current_state, current_action)

            # 来到新的一个状态之后，我们要获得新的状态目前的Q值
            next_state_q_values = q_table.ix[next_state, get_valid_actions(next_state)]

            # 紧接着重新更新这个值,因为这个值代表了对动作的一个反馈，我们执行了动作来到了这个地方，就要更新一下给一个反馈，看这个动作是否是对的或者是好的
            q_table.ix[current_state, current_action] += alpha * (reward[next_state] + gamma * next_state_q_values.max() - q_table.ix[current_state, current_action])

            # 同时更新到最新的状态,算法从这个状态再开始下一个动作
            current_state = next_state


            update_env(current_state) # enviroment
            total_steps += 1          # enviroment
        # 在当前的状态下，我们选择Q值最大的动作，然后执行这个动作，进入到下一个新的状态，直到满足结束的条件：到达目的地
        print('\rEpisode {}: total_steps = {}'.format(i, total_steps), end='') # enviroment
        time.sleep(1)                                                          # enviroment
        
    print('\nq_table:')
    print(q_table)

        




if __name__ == "__main__":
    RL()