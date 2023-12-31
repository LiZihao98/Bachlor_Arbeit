# Bachlor_Arbeit
## Overview
The code is the foundation for the computational assessment in the undergraduate dissertation titled "Fast Failover Routing with Packet-Header Rewriting on Tree Algorithms." This repository contains all the algorithms that have been implemented. In addition, we also provide the configurations that contain the set of parameters used in the evaluations.

## Dependencies and requirements
We implemented the proposed algorithm in Python (3.11.4) using the NetworkX (3.1) library.

## Introduction
The algorithm presented is a synthesis of ideas from the papers "Improving the Resilience of Fast Failover Routing: TREE (Tree Routing to Extend Edge disjoint paths)" and "On the Resiliency of Static Forwarding Tables," proposing a fast rerouting algorithm with packet-header rewriting capabilities based on the Multiple-tree approach.

The pseudo-code of this algorithm is stored in the "RoutingAlgorithm.txt" file, which outlines the main concepts of the algorithm. The implementation code is contained within the "thesis_Algorithm.py" file. To evaluate the algorithm's performance in real-world network graphs, we utilize topological data provided by TopologyZoo, conveniently placed in the graphml folder for easy access.

Additional files are related to the content concerning tree algorithms.

## Experiments and Tests
For experiments and tests, random graphs can be generated (e.g., using the relevant methods provided by NetworkX) or topologies from the Topology Zoo can be utilized, with corresponding files found in the graphml folder. To conduct an experiment/test, simply run the thesis_Algorithm.py file, executing it by calling the main function and providing all necessary parameters.

Experiments can also be conducted using the test method starting at line 1220 in the thesis_Algorithm.py file. By default, this tests the real-world network topology of Cogentco from the Topology Zoo under a clustered fault mode. To test other topologies, first modify the address where the summary file is saved at line 1223. Then, when calling main (at line 1240), change the "graphMLpath" parameter. To test under a random fault mode, replace "clustered" with "random" at line 1240 and remove the failurePercentPar and failureDropPar, replacing them with the setting of edgeFailFactorPar, as shown in the comment at line 1241.

To experiment with synthetic random graphs, you need to first modify the address for saving the summary file. This involves commenting out line 1223 and using line 1224 instead. Then, when calling the main function, simply remove the graphMLpath parameter set.

### Example
_If you want to test artificial random graphs just delete graphMLpath = graphMLpaths[1]_

**Testing Random Failure Models**
```
p = 0.15
no_nodes = 10
repeats = 200
edgeFailFactorPar = 1
while edgeFailFactorPar <= 25:
            csv_dict = {}
            avg_hops, avg_hops_tree, avg_hops_edp, resilience, resilience_tree, resilience_edp = main(p, no_nodes, repeats, "random", edgeFailFactorPar = edgeFailFactorPar, graphMLpath = graphMLpaths[1])
            ...
            edgeFailFactorPar += 1
```
**Testing Clustered Failure Models**
```
p = 0.15
no_nodes = 10
repeats = 200
failpercent = 0.1
while failpercent <= 0.9:
            csv_dict = {}
            avg_hops, avg_hops_tree, avg_hops_edp, resilience, resilience_tree, resilience_edp = main(p, no_nodes, repeats, "clustered", failurePercentPar = failpercent, failureDropPar = 0.3, graphMLpath = graphMLpaths[1])
            ...
            failpercent += 0.1
            failpercent = round(failpercent, 1)
```
## Output
The results are stored in summary.csv file, which is located in the defined output or output_graphm
