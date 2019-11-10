# Parallelize OpenAI Gym environment

- simply creating separate instances of act agent and sending them to a process using `concurrent` or `multiprocess` doesn't work. 
- Checkout these resources, apparently they work
    - openAI baseline subProcVecEnv
    - [Parallel versions of classic control environment doesn't work !](https://github.com/openai/gym/issues/165)
    - [Daniels DQN implementation](https://github.com/DanielDworakowski/DQN-Algorithms)