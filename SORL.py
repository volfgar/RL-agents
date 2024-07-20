import gym
import numpy as np
import time
import pandas as pd


"""
SARSA on policy learning python implementation.
This is a python implementation of the SARSA algorithm in the Sutton and Barto's book on
RL. It's called SARSA because - (state, action, reward, state, action). The only difference
between SARSA and Qlearning is that SARSA takes the next action based on the current policy
while qlearning takes the action with maximum utility of next state.
Using the simplest gym environment for brevity: https://gym.openai.com/envs/FrozenLake-v0/
"""
'''
https://habr.com/ru/post/475236/ - ICE problem with py code examples
https://habr.com/ru/company/piter/blog/434738/  Q-learning on Taxi example

udemy cources:
https://www.udemy.com/course/deep-reinforcement-learning-in-python/
https://www.udemy.com/course/artificial-intelligence-reinforcement-learning-in-python/

Deep reinforcement Learning,  habr paper
https://habr.com/ru/post/439674/

!!!ATTENTION!!! anthology RL-methods Overview 
http://louiskirsch.com/maps/reinforcement-learning

'''

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def init_q(s, a, type="ones"):
    """
    @param s the number of states
    @param a the number of actions
    @param type random, ones or zeros for the initialization
    """
    if type == "ones":
        return np.ones((s, a))
    elif type == "random":
        return np.random.random((s, a))
    elif type == "zeros":
        return np.zeros((s, a))

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## SARSA: My Step function for  RL model
def MyStep(_a,_s,_aTrue,_sNew):    
    _done = True
    if _a != _aTrue:
        _sNew,_reward, _done = _sNew, -1, 0
    else: 
        _sNew,_reward, _done = (_sNew-1), 1, 1    
    return _sNew, _reward, _done, {"oldState": _s, "action":_a, "True action":_aTrue,"newState":_sNew,   "reward": _reward, "done":_done }

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def epsilon_greedy(Q, epsilon, n_actions, s, train=False):
    """
    @param Q Q values state x action -> value
    @param epsilon for exploration
    @param s number of states
    @param train if true then no random actions selected
    """
    _prob = np.random.rand()
    if train or _prob< epsilon:
        action = np.argmax(Q[s, :])
    else:
        action = np.random.randint(0, n_actions)
    return action, _prob
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def mainRL(_S, _A , topic,  _complLeads, _epsilon=0.95, _episodes=10000,  _Q = None, _ConvCount=None, _PrepQ=None, _DrawDynamicChart=None, _test=False, _render=False, _Learn=True):
    ## Main procedure
    '''
    @param Q Q values state x action -> value
    '''
    alpha = 0.4 ## это фактор обучения. Чем он выше, тем сильнее агент доверяет новой информации
    gamma = 0.999 ## это фактор дисконтирования. Чем он меньше, тем меньше агент задумывается о выгоде от будущих своих действий
    epsilon = _epsilon## вероятность выбора лучшего действия в epsilon-жадной стратегии
    episodes = _episodes #10000#len(_complLeads)
    episodesChunkShowQ = 100
    max_steps = 1
    sNameDummy = 'Прочее'   ## Тема по умолчанию, используется, если в _S не найдена тема рассматриваемого обращения
    aNameDummy = _A.iloc[0]['Наименование пакета']         ## Имя действия по умолчанию, используется, если в _A не найдено действие рассматриваемого обращения
    #n_tests = 20
    try:
        n_actions = len(_A)
        timestep_reward = []
        Q = init_q(len(_S), len(_A), 'ones') if _Q is None else _Q ## Требуется ли обучение с нуля или берем уже обученную матрицу
        #print(Q)
        convHist = pd.DataFrame()
        Qhist = pd.DataFrame()

        for index, episode in _complLeads.iloc[:episodes].iterrows():
                sName = episode['Тема']
                aName = episode['Наименование пакета']
                s =  _S[_S['Тема'] == sName].index if  len(_S[_S['Тема'] == sName].index) > 0 else  _S[_S['Тема'] == sNameDummy].index
                if len(s) == 0: 
                    raise ValueError('Wrong State: '+sName+' with state id: '+str(s))
                if _render:
                    print(f"Episode MSISDN: {episode['MSISDN']}, state: {s[0]}, state:{sName}, episode:{index}")
                total_reward = 0
                a,prob_ = epsilon_greedy(Q, epsilon, n_actions, s)
                t = 0        
                done = False
                while t < max_steps:
                    #if _render:
                     #   env.render()
                    t += 1
                    # Get real - true action, which been applied
                    aTrue =  _A[_A['Наименование пакета'] == aName].index if  len(_A[_A['Наименование пакета'] == aName].index) > 0 else  _A[_A['Наименование пакета']  == aNameDummy].index                    
                    #reward = -1 if a != aTrue else 10 
                    s_, reward, done, info = MyStep(a,s[0],aTrue[0],len(_S)-1)

                    if _render:
                        print(info)
                    total_reward += reward
                    a_, probNext = epsilon_greedy(Q, epsilon, n_actions, s_)
                    _complLeads.loc[index, 'pred'] = done
                    _complLeads.loc[index, 'trueA'] = aTrue[0]
                    _complLeads.loc[index, 'A'] = a
                    _complLeads.loc[index, 'S'] = s                    
                    _complLeads.loc[index, 'reward'] = reward
                    _complLeads.loc[index, 'nextS'] = s_            
                    _complLeads.loc[index, 'nextA'] = a_                
                    _complLeads.loc[index, 'prob'] = prob_
                    _complLeads.loc[index, 'packUsed'] = _A.loc[a]['Наименование пакета']
                    _complLeads.loc[index, 'maxA'] = np.argmax(Q[s, :]) 
            
                    if _Learn:
                        if done:
                            Q[s, a] += alpha * ( reward  - Q[s, a] )
                        else:
                            Q[s, a] += alpha * ( reward + (gamma * Q[s_, a_] ) - Q[s, a] )
                    _complLeads.loc[index, 'Q'] = Q[s, a]
                    s, a = s_, a_
                    if _render:
                        print(f"This episode took {t} timesteps and reward {total_reward}")
                    timestep_reward.append(total_reward)
                    break
                if _render:
                    print(f"Here are the Q values:\n{Q}\nTesting now:")
                if(_test) & (np.mod(index,episodesChunkShowQ) == 0) & (index > 0):
                    #print(index)
                    #pdQ.iloc[:,:] = Q
                    #print(_ConvCount(_complLeads.loc[:index]))
                    # Add to pd per 100 episodes or timeSlice!!!!     
                    conv = _ConvCount(_complLeads[:index], index, episodesChunkShowQ)  
                    pdQ = _PrepQ(Q, _S, _A, index)
                    #res=pd.concat([pdQ,conv], axis=1).copy()
                    res = conv.copy()
                    convHist = pd.concat([convHist, res])
                    convHist['Learning rate'] = 0
                    #topic =  'Недовольство\Тарифные планы\Для массового рынка\Недовольство текущими условиями тарифного плана'
                    #topic =  'Недовольство\Интернет\Мобильный Интернет\Низкая скорость'
                    #topic =  'Недовольство\Интернет\Мобильный Интернет\Низкая скорость'
                    
                    #pack = None
                    #pack = 'Скидка 15% по тарифу на 3 месяца'
                    _DrawDynamicChart(convHist, topic,)
                    #pdQ = pd.Data
    except ValueError as e:
        print('Exception in mainRL!')
        print(e)
    return _complLeads, convHist, Q


#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def sarsa(alpha, gamma, epsilon, episodes, max_steps, n_tests, render = False, test=False):
    """
    @param alpha learning rate
    @param gamma decay factor
    @param epsilon for exploration
    @param max_steps for max step in each episode
    @param n_tests number of test episodes
    """
    env = gym.make('Taxi-v3')
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = init_q(n_states, n_actions, type="ones")
    timestep_reward = []
    for episode in range(episodes):
        print(f"Episode: {episode}")
        total_reward = 0
        s = env.reset()
        a = epsilon_greedy(Q, epsilon, n_actions, s)
        t = 0        
        done = False
        while t < max_steps:
            if render:
                env.render()
            t += 1
            #s_, reward, done, info = env.step(a)
            


            total_reward += reward
            a_ = epsilon_greedy(Q, epsilon, n_actions, s_)
            if done:
                Q[s, a] += alpha * ( reward  - Q[s, a] )
            else:
                Q[s, a] += alpha * ( reward + (gamma * Q[s_, a_] ) - Q[s, a] )
            s, a = s_, a_
            if done:
                if render:
                    print(f"This episode took {t} timesteps and reward {total_reward}")
                timestep_reward.append(total_reward)
                break
    if render:
        print(f"Here are the Q values:\n{Q}\nTesting now:")
    if test:
        test_agent(Q, env, n_tests, n_actions)
    return timestep_reward

def test_agent(Q, env, n_tests, n_actions, delay=0.1):
    for test in range(n_tests):
        print(f"Test #{test}")
        s = env.reset()
        done = False
        epsilon = 0
        total_reward = 0
        while True:
            time.sleep(delay)
            env.render()
            a = epsilon_greedy(Q, epsilon, n_actions, s, train=True)
            print(f"Chose action {a} for state {s}")
            s, reward, done, info = env.step(a)
            total_reward += reward
            if done:
                print(f"Episode reward: {total_reward}")
                time.sleep(1)
                break
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
if __name__ =="__main__":
    alpha = 0.4
    gamma = 0.999
    epsilon = 0.9
    episodes = 3000
    max_steps = 1
    n_tests = 20
    timestep_reward = sarsa(alpha, gamma, epsilon, episodes, max_steps, n_tests, test=True)
    print(timestep_reward)

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
'''
   The Complaint - MarketOffer Problem
    by TD SG186072
    
    Description:
        Should choose true Market Offer as a response for a customer complaint
    Observations:
        Customer  profile + complaint type as a Customer Jorney (CJ) trigger
    Actions:
        There are  some deterministic offers:
    - 0: 5GB
    - 1: 
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger
    
    Rewards:
        A customer has taken an offer +20 
        Otherwise -1

    Rendering:

    state space is represented by:
        (types of complaints * customer clusters, represents customer profile)
'''

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ATTENTION!!! For test issues only Not applicable now!
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#from gym.envs.registration import registry, register, make, spec

## Check list of registered envs
#env_dict = gym.envs.registration.registry.env_specs.copy()
#env_dict['foo-v0']

#register(
#    id='GR-v0',
#    entry_point='gym.envs.toy_text:FooEnv',
#)


#from setuptools import setup
#setup(name='gym_foo',
#      version='0.0.1',
#      install_requires=['gym']#And any other dependencies required
#)
#import gym
#from gym.envs.toy_text.foo import FooEnv
##from gym.envs.toy_text.taxi import TaxiEnv

##//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## UnRegistration env
#del gym.envs.registration.registry.env_specs['GR-v0']

#env = gym.make('Taxi-v3')

##//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#from gym import envs
#env = gym.make('GR-v0')


##//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#import gym
#import gym_foo
#import sys
#del sys.modules["gym_foo"]
#del gym_foo
#env = gym.make('foo-v0')

#n_states, n_actions = env.observation_space.n, env.action_space.n



