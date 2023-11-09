# Bachlor_Arbeit
## Overview
The code is the foundation for the computational assessment in the undergraduate dissertation titled "Fast Failover Routing with Packet-Header Rewriting on Tree Algorithms." This repository contains all the algorithms that have been implemented. In addition, we also provide the configurations that contain the set of parameters used in the evaluations.

## Dependencies and requirements
We implemented the proposed algorithm in Python (3.11.4) using the NetworkX (3.1) library.

## Introduction
The algorithm presented is a synthesis of ideas from the papers "Improving the Resilience of Fast Failover Routing: TREE (Tree Routing to Extend Edge disjoint paths)" and "On the Resiliency of Static Forwarding Tables," proposing a fast rerouting algorithm with packet-header rewriting capabilities based on the Multiple-tree approach.

The pseudo-code of this algorithm is stored in the "RoutingAlgorithm.txt" file, which outlines the main concepts of the algorithm. The implementation code is contained within the "thesis_Algorithm.py" file. To evaluate the algorithm's performance in real-world network graphs, we utilize topological data provided by TopologyZoo, conveniently placed in the graphml folder for easy access.

Additional files are related to the content concerning tree algorithms.
