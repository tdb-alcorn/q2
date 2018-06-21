#!/usr/bin/env bash

### Listing ###

q2 list {environments,agents,regimens,objectives}
# Each one of these commands should show some objects available.


### Training ###

q2 train random --regimen online --env gym.CartPole-v1 --render --episodes 10
# You should see CartPole being played.


### Generating ###

q2 generate agent foo
# You should see an agent called Foo at ./agents/foo.py
q2 list agents
# `foo`` should show up in the results.
q2 train foo --env gym.CartPole-v1 --render --episodes 10
# You should see CartPole being played.