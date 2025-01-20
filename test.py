from d3rlpy.datasets import get_d4rl
from d3rlpy.algos.qlearning import ACLConfig
from d3rlpy.metrics import EnvironmentEvaluator

# get dataset
dataset, env = get_d4rl('hopper-medium-v2')

# get acl model
acl = ACLConfig(compile_graph=True).create('cpu')
acl.build_with_dataset(dataset)

# set environment in scorer function
env_evaluator = EnvironmentEvaluator(env)

# train
acl.fit(
    dataset,
    n_steps=10000,
    evaluators={
        'environment': env_evaluator,
    },
)
