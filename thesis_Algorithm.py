import networkx as nx
from datetime import datetime
from timeit import default_timer as timer
import csv
import random
import math
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import sys
sys.setrecursionlimit(3000)
########### Tree Algorithm ##############################
def kResilienceTrees(s, d, fails, g, version="multiple", file_name=None, draw=False, unranked=True, treeChoice="shortest"):

    sp_length = getTrueShortestPathLength(g.copy(), s, d, fails)

    # we will use a copy of the input graph
    g_copy = copy.deepcopy(g)

    # At the beginning, all the nodes and the edges of the graph are numbered with '0'
    # i.e. they don't belong to any structure yet
    nx.set_edge_attributes(g_copy,"0","attr")
    nx.set_node_attributes(g_copy,"0","attr")
    # The source and the destination node have their own separate attributes ('s' and 'd' respectively)
    g_copy.nodes[s]["attr"] = "s"
    g_copy.nodes[s]["label"] = "s"
    g_copy.nodes[d]["attr"] = "d"
    g_copy.nodes[d]["label"] = "d"
    try:
        startEDP = timer()

        edge_disjoint_paths = list(nx.edge_disjoint_paths(g_copy,s,d))
    except:
        # if an error is thrown, it means that the source node and the destination node belong
        # to different graph components; the destination node cannot be reached from the source node anymore
        return (True,0,0,[])

    #sort paths long to short
    edge_disjoint_paths.sort(key=lambda x: len(x), reverse=False)

    #give numbers to paths, the higher the shorter
    no_path = 1
    for path in edge_disjoint_paths:
        for i in range(0,len(path)-1):
            if g_copy.nodes[path[i+1]]["attr"] != "d":
                g_copy.nodes[path[i+1]]["attr"] = str(no_path)
            g_copy[path[i]][path[i+1]]["attr"] = str(no_path)

        no_path += 1

    endEDP = timer()

    timeEDP = endEDP - startEDP
    for fail in fails:
        g_copy[fail[0]][fail[1]]["failed"] = True


    # if draw:
    #     Colorizer.colorizeGraph(g_copy.copy(), paths=len(edge_disjoint_paths), file_name=file_name + "-" + version + "-paths")

    startTreeBuilding = timer()

    if version == "multiple":
        makeMultipleTrees(g_copy, edge_disjoint_paths)
    else:
        makeOneTree(g_copy, edge_disjoint_paths, reverse=True) #reverse here, so we make tree from LONGEST path

    endTreeBuilding = timer()
    timeTree = endTreeBuilding - startTreeBuilding

    d_incidents = set()
    # remove incident edges of D from all structures
    for d_edge in g_copy.edges(d):
        d_incidents.add(d_edge[1])
        g_copy[d_edge[0]][d_edge[1]]['attr'] = "-1"

    s_incidents = set()
    for s_edge in g_copy.edges(s):
        s_incidents.add(s_edge[1])

    # if draw:
    #     Colorizer.colorizeGraph(g_copy.copy(), paths=len(edge_disjoint_paths), file_name=file_name + "-" + version + "-noPP-" + version)

    startTreeProcessing = timer()

    trees_changed, overallNodeAdditions = postProcessTree(g_copy, s, d_incidents, s_incidents, edge_disjoint_paths, version=version)

    endTreeProcessing = timer()
    timeProcessing = endTreeProcessing - startTreeProcessing

    overallTime = (timeEDP + timeTree + timeProcessing) * 1000

    """print("Version:",version)
    print("Nodes:",g.number_of_nodes())
    print("Edges:",g.number_of_edges())
    print("Time:",overallTime)
    print("Time EDP:",timeEDP * 1000)
    print("--------------------------------------------")"""

    # if draw:
    #     Colorizer.colorizeGraph(g_copy.copy(), paths=len(edge_disjoint_paths), file_name=file_name + "-" + version + "-PP-" + version)


    #rankings = rankTree(g_copy, s, d, d_incidents, trees_changed)
    rankings = rankTree(g_copy, s, d, d_incidents) #causes crash when trees neighbors S as well

    hops, routed_paths, tree_order_ranked = routeTrees(g_copy.copy(), s, d, d_incidents, fails, rankings, treeChoice=treeChoice)

    if unranked:
        hops_unranked, routed_paths_unranked, tree_order_unranked = routeTrees(g_copy.copy(), s, d, d_incidents, fails, rankings, unranked=True)
    else:
        hops_unranked = -1

    # if draw:
    #     Colorizer.colorizeGraph(g_copy.copy(), paths=len(edge_disjoint_paths), hops=hops, file_name=file_name + "-PProuted-" + version, routed_paths=routed_paths, sp_length=sp_length)

    return hops, sp_length, overallNodeAdditions, hops_unranked, g_copy, routed_paths, rankings, tree_order_ranked, timeEDP * 1000, overallTime

#makes as one tree as large as possible from graph g
def makeOneTree(g_copy, edge_disjoint_paths, reverse=False):

    if reverse:
        edge_disjoint_paths.reverse()
        no_tree = len(edge_disjoint_paths)
    else:
        no_tree = 1

    for path in edge_disjoint_paths:
        nodes_added = 0
        for i in range(1, len(path) - 1):
            nodes = [path[i]]  # obtain a list with the nodes of the i-th path
            it = 0
            while (it < len(nodes)):
                # obtain a list with all the incident edges of nodes from the i-th path
                list_of_incident_edges = list(g_copy.edges(nodes[it]))
                # obtain a generator of the previous edges, which provides edges with the '0' attribute
                # (meaning that they are not used by any structure yet)
                edge_candidates_gen = (edge for edge in list_of_incident_edges if
                                       g_copy.get_edge_data(edge[0], edge[1]).get("attr") == "0")
                for edge in edge_candidates_gen:
                    if g_copy.nodes[edge[1]]["attr"] == "0":
                        g_copy[edge[0]][edge[1]]["attr"] = str(no_tree)
                        g_copy.nodes[edge[1]]["attr"] = str(no_tree)

                        nodes.append(edge[1])
                        nodes_added += 1

                    # we also give an attribute to the incident edges of the destination node
                    # however, tree leaves of the tree are considered to be the neighbors of the destination node
                    if g_copy.nodes[edge[1]]["attr"] == "d":
                        g_copy[edge[0]][edge[1]]["attr"] = str(no_tree)

                it += 1

        no_tree = no_tree + 1 if not reverse else no_tree - 1
        if DEBUG:
            print("added", nodes_added, "nodes to tree", no_tree)

#makes as many trees as possible from graph g
def makeMultipleTrees(g_copy, edge_disjoint_paths):
    no_tree = 1
    for path in edge_disjoint_paths:
        nodes_added = 0
        for i in range(1, len(path) - 1):
            nodes = [path[i]]  # obtain a list with the nodes of the i-th path
            it = 0
            while (it < len(nodes)):
                list_of_incident_edges = list(g_copy.edges(nodes[it]))

                edge_candidates_gen = (edge for edge in list_of_incident_edges if
                                       g_copy.get_edge_data(edge[0], edge[1]).get("attr") == "0")
                for edge in edge_candidates_gen:

                    node_candidate_incident_attrs = [g_copy[e[0]][e[1]]["attr"] for e in g_copy.edges(edge[1])]

                    if str(no_tree) not in node_candidate_incident_attrs and g_copy[edge[0]][edge[1]]["attr"] == "0" and \
                            g_copy.nodes[edge[1]]["attr"] != "s" and g_copy.nodes[edge[1]]["attr"] != "d":
                        g_copy[edge[0]][edge[1]]["attr"] = str(no_tree)
                        g_copy.nodes[edge[1]]["attr"] = str(no_tree)

                        nodes.append(edge[1])
                        nodes_added += 1

                    # we also give an attribute to the incident edges of the destination node
                    # however, tree leaves of the tree are considered to be the neighbors of the destination node
                    if g_copy.nodes[edge[1]]["attr"] == "d":
                        g_copy[edge[0]][edge[1]]["attr"] = str(no_tree)

                it += 1
        if DEBUG:
            print("added", nodes_added, "nodes to tree", no_tree)
        no_tree += 1

#route along tree structures
def routeTrees(g, s, d, d_incidents, fails, rankings, unranked=False, treeChoice="shortest"):
    hops = 0

    trees_attributes = []
    for node1,node2,data in g.edges(data=True):
        if data['attr'] not in trees_attributes and int(data['attr']) > 0:
            trees_attributes.append(data['attr'])

    #trees_attributes = [str(el2) for el2 in sorted([int(el1) for el1 in trees_attributes], reverse=True)]
    trees_attributes = getTreeOrder(rankings, treeChoice=treeChoice)

    routed_paths = []

    found = False

    for attr in trees_attributes:
        if found:
            break
        T = nx.Graph() # we reconstruct the tree
        for node1,node2,data in g.edges(data=True):
            if data['attr'] == attr:
                T.add_edge(node1,node2)

        if s not in list(T.nodes):
            continue

        dfs_edge_order_list = list(nx.dfs_labeled_edges(T, s))
        for n1,n2,label in dfs_edge_order_list:
            if label == "nontree" or n1 == n2: # we also remove self-loops
                dfs_edge_order_list.remove((n1,n2,label))

        hops_current_tree = 0

        if not unranked and attr in rankings:
            dfs_edge_order_list = rankDfs(T, s, d, dfs_edge_order_list, rankings[attr])

        final_dfs = removeFails(dfs_edge_order_list, fails)
        for n1,n2,label in final_dfs:
            routed_paths.append((n1,n2))
            hops_current_tree += 1

            if n1 in d_incidents and (str(n1),str(d)) not in fails and (str(d),str(n1)) not in fails and (int(n1),int(d)) not in fails and (int(d),int(n1)) not in fails:
                routed_paths.append((n1, d))
                hops_current_tree += 1
                found = True
                break
            elif n2 in d_incidents and (str(n2), str(d)) not in fails and (str(d), str(n2)) not in fails and (int(n2), int(d)) not in fails and (int(d), int(n2)) not in fails:
                routed_paths.append((n2, d))
                hops_current_tree += 1
                found = True
                break

        hops += hops_current_tree
    if DEBUG:
        if unranked:
            print("UNRANKED: TOOK", hops, "HOPS")
        else:
            print("RANKED: TOOK", hops, "HOPS")

    return hops if found else -1, routed_paths, trees_attributes


#prune created tree branches that do not lead to d
def postProcessTree(g, s, d_incidents, s_incidents, edge_disjoint_paths, version=None, tree_attr=None):

    # we will test if we can still reach the destination node; for this we will analyze each subgraph (tree) until we will find the destination node
    trees_attributes = []
    trees_changed = set()
    if tree_attr is None:
        for node1, node2, data in g.edges(data=True):
            if data['attr'] not in trees_attributes:
                if int(data['attr']) > 0:
                    trees_attributes.append(data['attr'])
    else:
        trees_attributes.append(tree_attr)

    overallNodeAdditions = 0
    for attr in trees_attributes:
        T = nx.Graph()  # we reconstruct the tree

        for node1, node2, data in g.edges(data=True):
            if data['attr'] == attr:
                T.add_edge(node1, node2)
        if s in list(T.nodes):

            dfs_edge_order_list = list(nx.dfs_labeled_edges(T, s))
            for n1,n2,label in dfs_edge_order_list:
                if label == "nontree" or n1 == n2: # we also remove self-loops
                    dfs_edge_order_list.remove((n1,n2,label))
                    #edges_to_remove.add((node1, node2))

            #print(dfs_edge_order_list)

            good_branch_nodes = set()
            visited_nodes = set()
            visited_nodes.add(dfs_edge_order_list[0][0])
            delete_mode = False

            for i in range(len(dfs_edge_order_list)):

                n1,n2,label = dfs_edge_order_list[i]

                if label == "forward":
                    visited_nodes.add(n2)
                elif label == "reverse":
                    visited_nodes.remove(n2)

                if label == "forward" or n2 in good_branch_nodes:
                    delete_mode = False

                if delete_mode:

                    if DEBUG:
                        print("edge {},{} set to 0".format(n1, n2))

                    g[n1][n2]["attr"] = "0"
                    g.nodes[n2]["attr"] = "0"

                if i < len(dfs_edge_order_list) - 1:
                    n1_next, n2_next, label_next = dfs_edge_order_list[i+1]
                    if label == "forward" and label_next == "reverse" and str(n2) not in d_incidents and int(n2) not in d_incidents: #and n2 not in s_incidents:
                        delete_mode = True
                    elif str(n2) in d_incidents or int(n2) in d_incidents: #or n2 in s_incidents:
                        [good_branch_nodes.add(el) for el in visited_nodes]
                        if edge_disjoint_paths is not None:
                            cnt = 0
                            for path in edge_disjoint_paths:
                                if n2 not in path:
                                    cnt += 1
                                else:
                                    break
                            if cnt == len(edge_disjoint_paths):
                                overallNodeAdditions += 1
                                trees_changed.add(attr)
                                if DEBUG:
                                    print("found node", n2, "that was not originally present in edge disjoint paths")

    return trees_changed, overallNodeAdditions

#order trees by their shortest distance from root node to D
def getTreeOrder(rankings, treeChoice="shortest"):

    order = []
    for key in rankings:
        if len(rankings[key][0]) == 0:
            order.append((key, 1)) #check again
        else:
            if treeChoice == "shortest":
                tmpval = np.min(list(rankings[key][0].values())[0])
            elif treeChoice == "average":
                tmpval = np.mean(list(rankings[key][0].values())[0])
            elif treeChoice == "edgeCount":
                tmpval = len(list(rankings[key][0].values()))

            order.append((key, tmpval))

    order.sort(key=lambda x: x[1])
    final_order = [str(el[0]) for el in order]

    return final_order

#calculate fastest direction to d for every tree
def rankTree(g, s, d, d_incidents, trees_changed=None):

    #if trees_changed not given, just take all attrs in graph (even the one that are not trees, but still paths)
    if trees_changed is None:
        trees_changed = []
        for node1, node2, data in g.edges(data=True):
            if data['attr'] not in trees_changed and int(data['attr']) > 0:
                trees_changed.append(data['attr'])

    trees_ranked = {}
    for attr in trees_changed:
        T = nx.Graph()  # we reconstruct the tree

        for node1, node2, data in g.edges(data=True):
            if data is not None and 'attr' in data and data['attr'] == attr:
                T.add_edge(node1, node2)


        if s not in list(T.nodes):
            continue

        dfs_edge_order_list = list(nx.dfs_labeled_edges(T, s))
        for n1,n2,label in dfs_edge_order_list:
            if label == "nontree" or n1 == n2: # we also remove self-loops
                dfs_edge_order_list.remove((n1,n2,label))

        branching_dict = {}
        direction_dict = {}
        best_dict = {}
        travelled_nodes = set()
        for i in range(len(dfs_edge_order_list)):
            n1, n2, label = dfs_edge_order_list[i]

            #if we can reach d from here and its the best route so far, lock temporary distances
            if label == "forward" and n2 in d_incidents:
                for key in branching_dict:
                    if key not in best_dict:
                        best_dict[key] = []
                        direction_dict[key] = []

                    if key in travelled_nodes:
                        best_dict[key].append(branching_dict[key] + 1) #plus one bc we need one more hop to d
                        direction_dict[key].append(n2)

            #add current node as travelled
            travelled_nodes.add(n2)

            #remove current node from travelled if we move backwards from a node already visited
            if label == "reverse" and n2 in travelled_nodes:
                travelled_nodes.remove(n2)

            if label == "forward":
                for node in travelled_nodes:
                    if node in branching_dict:
                        branching_dict[node] += 1
                    else:
                        branching_dict[node] = 1

            if label == "reverse":
                for node in travelled_nodes:
                    branching_dict[node] -= 1

        trees_ranked[attr] = (best_dict, direction_dict)
    return trees_ranked

#prepare final depth first traversal according to ranking (fastest route to d first)
def rankDfs(T, s, d, dfs, ranking):
    successor_dict = nx.dfs_successors(T, s)

    ranking_processed = [{}, {}]

    def getAllSuccessors(node, succs, set):
        if node in succs:
            for succ in succs[node]:
                set.add(succ)
                getAllSuccessors(succ, succs, set)

    for node in ranking[0]:
        if node not in successor_dict:
            continue

        #write direct neighbor to ranking dict instead of very last node on path -> enable local routing
        neighs = successor_dict[node]
        for neigh in neighs:
            all_succs = set()
            getAllSuccessors(neigh, successor_dict, all_succs)
            for succ in all_succs:
                if succ in ranking[1][node]:
                    ranking[1][node][ranking[1][node].index(succ)] = neigh

        #if multiple paths go over same direct neighbor, take shortest one
        ranking_processed[0][node] = []
        ranking_processed[1][node] = []
        for distinct in np.unique(ranking[1][node]):
            indices = [i for i, x in enumerate(ranking[1][node]) if x == distinct]

            tmpmin = np.min([ranking[0][node][idx] for idx in indices])
            ranking_processed[0][node].append(tmpmin)
            ranking_processed[1][node].append(distinct)

        #sort neighbor rankings so that shortest route is first
        hops = ranking_processed[0][node].copy()
        hops_sorted = sorted(hops)
        dirs_sorted = []
        for i in range(len(hops_sorted)):
            dirs_sorted.append(ranking_processed[1][node][hops.index(hops_sorted[i])])
            hops[hops.index(hops_sorted[i])] = -1

        ranking_processed[0][node] = hops_sorted
        ranking_processed[1][node] = dirs_sorted

    for key in ranking_processed[0]:
        i = 0
        swaps_idx = [-1] * len(ranking_processed[0][key])
        for n1, n2, label in dfs:
            if n1 == key and label == "forward":
                swaps_idx[ranking_processed[1][key].index(n2)] = i
            if n1 == key and label == "reverse":
                swaps_idx[ranking_processed[1][key].index(n2)] = (swaps_idx[ranking_processed[1][key].index(n2)], i + 1)

            i += 1

        range_start = min(swaps_idx, key=lambda x: x[0])[0]
        range_end = max(swaps_idx, key=lambda x: x[1])[1]
        #dfs_reordered = [None] * len(dfs)
        dfs_reordered = []
        for si in swaps_idx:
            dfs_reordered.extend(dfs[si[0] : si[1]])

        dfs[range_start : range_end] = dfs_reordered


    return dfs

#remove failed edges from depth first traversal
def removeFails(dfs, fails):

    #fails = [(14,16)]
    for fail in fails:
        idx = 0
        start_idx = -1
        end_idx = -1
        search_mode = False
        failed_edge = (-1,-1)
        for n1,n2,label in dfs:
            if ((n1 == fail[0] and n2 == fail[1]) or (n1 == fail[1] and n2 == fail[0])) and label == "forward":
                search_mode = True
                failed_edge = fail
                start_idx = idx

            if search_mode and ((n1 == failed_edge[0] and n2 == failed_edge[1]) or (n1 == failed_edge[1] and n2 == failed_edge[0])) and label == "reverse":
                end_idx = idx
                break #new

            idx += 1
        if start_idx > -1 and end_idx > -1:
            for i in range(end_idx, start_idx-1, -1):
                del dfs[i]

    return dfs

def getTrueShortestPathLength(g, s, d, fails):
    g.remove_edges_from(fails)
    try:
        spLength = nx.shortest_path_length(g,s,d)
    except nx.NetworkXNoPath:
        spLength = -1
    return spLength
########### EDP with Header Rewriting ###################
def makeEDP(g, shortest_path, d):
    node_list = list(g.nodes)
    tree_list = {}
    tree_list['main'] = shortest_path
    for s in shortest_path:
        if s != d:
            try:
                edge_disjoint_paths = list(nx.edge_disjoint_paths(g, s, d))
            except Exception as e:
                edge_disjoint_paths = []
            edge_disjoint_paths.sort(key=lambda x: len(x), reverse=False) 
            tree = {}
            no_path = 1
            for edp in edge_disjoint_paths:
                path = []
                for i in range(0, len(edp)):
                    if i+1 < len(edp):
                        path.append((edp[i], edp[i+1], 'forward'))
                tree[str(no_path)] = path
                no_path += 1
            tree_list[s] = tree
    return tree_list

def getNextEdge(attr_path, edge, H, route_list, fails):
    if attr_path != None:
        if edge == None:
            edge = attr_path[0]
            if (edge[0], edge[1]) not in fails and (edge[1], edge[0]) not in fails:
                route_list.append(edge)
            return attr_path[0]
        else:
            try:
                if H == 0:
                    next_index = attr_path.index(edge) + 1
                    next_edge = attr_path[next_index]
                    if (next_edge[0], next_edge[1]) not in fails and (next_edge[1], next_edge[0]) not in fails:
                        route_list.append(next_edge)
                    return next_edge
                else:
                    next_index = attr_path.index(edge) - 1
                    if next_index < 0:
                        return None
                    next_edge = attr_path[next_index]
                    route_list.append((next_edge[1], next_edge[0], 'reverse'))
                    return next_edge
            except Exception as e:
                return None

def routeEDP(H, tree, edge, fails, tree_list, d, route_list):
    if edge[0] == None:
        return
    if (edge[0], edge[1]) in fails or (edge[1], edge[0]) in fails:
        if tree == None:
            route_list.pop()
            current_tree = getCurrentTreeofNode(edge[0], tree_list)
            if current_tree != {}:
                attr_path = getNextAttrPath(current_tree, attr=0) # get first attribute of current_tree
                next_edge = getNextEdge(attr_path, None, 0, route_list, fails)
                routeEDP(0, current_tree, next_edge, fails, tree_list, d, route_list)
            else:
                # If no tree exists with the current node as root -> go back
                shortest_path = tree_list['main']
                next_edge = (getNextNodeOnMainPath(edge[0], shortest_path, 1), edge[0], 'forward')
                if next_edge[0] != None:
                    route_list.append((next_edge[1], next_edge[0], 'reverse'))
                routeEDP(1, None, next_edge, fails, tree_list, d, route_list)
        else: # currently it is routing on the tree
            attr = getAttrGivenEdge(tree, edge)
            back_edge = getNextEdge(tree[attr], edge, 1, route_list, fails)
            if back_edge != None: # but hits a failed link, but can go back
                routeEDP(1, tree, back_edge, fails, tree_list, d, route_list)
            else: # there is no way can go backwards, it means currently it is back to main path
                next_attr_path = getNextAttrPath(tree, attr) 
                if next_attr_path != None: # but there is also other attribute path has not been tried
                    next_edge = getNextEdge(next_attr_path, None, 0, route_list, fails) # get the first edge of next attr_path
                    routeEDP(0, tree, next_edge, fails, tree_list, d, route_list)
                else: # all path has been tried and failed. -> go back on the main path
                    shortest_path = tree_list['main']
                    next_edge = (getNextNodeOnMainPath(edge[0], shortest_path, 1), edge[0], 'forward')
                    if next_edge[0] != None:
                        route_list.append((next_edge[1], next_edge[0], 'reverse'))
                    routeEDP(1, None, next_edge, fails, tree_list, d, route_list)
    elif tree == None:
        if H == 0:
            if edge[1] == d:
                # successful
                return
            else:
                # continue and go forward on the main path.
                shortest_path = tree_list['main']
                next_edge = (edge[1], getNextNodeOnMainPath(edge[1], shortest_path, 0), 'forward')
                route_list.append(next_edge)
                routeEDP(0, None, next_edge, fails, tree_list, d, route_list)
        else:
            # go back on the main path
            current_tree = getCurrentTreeofNode(edge[0], tree_list)
            # current_tree = removeFailed(current_tree, fails, d_neighbor, d)
            attr_path = getNextAttrPath(current_tree, attr=0) # get first attribute of current_tree
            next_edge = getNextEdge(attr_path, None, 0, route_list, fails)
            routeEDP(0, current_tree, next_edge, fails, tree_list, d, route_list)
    else:
        attr = getAttrGivenEdge(tree, edge)
        if H == 0:
            if edge[1] == d:
                return
            else:
                next_edge = getNextEdge(tree[attr], edge, 0, route_list, fails)
                routeEDP(0, tree, next_edge, fails, tree_list, d, route_list)
        else:
            back_edge = getNextEdge(tree[attr], edge, 1, route_list, fails)
            if back_edge != None:
                    routeEDP(1, tree, back_edge, fails, tree_list, d, route_list)
            else:
                next_attr_path = getNextAttrPath(tree, attr)
                if next_attr_path != None: # has next atrribute
                    next_edge = getNextEdge(next_attr_path, None, 0, route_list, fails) # get the first edge of next attr_path
                    routeEDP(0, tree, next_edge, fails, tree_list, d, route_list)
                else:
                    shortest_path = tree_list['main']
                    # all path has been tried and failed. -> go back on the main path
                    next_edge = (getNextNodeOnMainPath(edge[0], shortest_path, 1), edge[0], 'forward')
                    if next_edge[0] != None:
                        route_list.append((next_edge[1], next_edge[0], 'reverse'))
                    routeEDP(1, None, next_edge, fails, tree_list, d, route_list)
    
                
########### Tree with Header Rewriting ##################
def makeMultipleTreesM(g_copy, edge_disjoint_paths):
    no_tree = 1  
    for path in edge_disjoint_paths: 
        nodes_added = 0 
        for i in range(1, len(path) - 1):
            nodes = [path[i]] 
            it = 0
            while (it < len(nodes)):
                list_of_incident_edges = list(g_copy.edges(nodes[it]))  

                edge_candidates_gen = (edge for edge in list_of_incident_edges if
                                       g_copy.get_edge_data(edge[0], edge[1]).get("attr") == "0")
                for edge in edge_candidates_gen:

                    node_candidate_incident_attrs = [g_copy[e[0]][e[1]]["attr"] for e in g_copy.edges(edge[1])]

                    if str(no_tree) not in node_candidate_incident_attrs and g_copy[edge[0]][edge[1]]["attr"] == "0" and \
                            g_copy.nodes[edge[1]]["attr"] != "s" and g_copy.nodes[edge[1]]["attr"] != "d":
                        g_copy[edge[0]][edge[1]]["attr"] = str(no_tree)  
                        g_copy.nodes[edge[1]]["attr"] = str(no_tree)  

                        nodes.append(edge[1]) 
                        nodes_added += 1

                    if g_copy.nodes[edge[1]]["attr"] == "d":
                        g_copy[edge[0]][edge[1]]["attr"] = str(no_tree)

                it += 1

        if DEBUG:
            print("added", nodes_added, "nodes to tree", no_tree)
        no_tree += 1  

def postProcessTreeM(g, s, d_incidents, attribute_list, edge_disjoint_paths, version=None, tree_attr=None):
    trees_attributes = []  
    trees_changed = set() 
    if tree_attr is None:
        for node1, node2, data in g.edges(data=True):
            if data['attr'] not in trees_attributes:
                if int(data['attr']) > 0:
                    trees_attributes.append(data['attr'])
    else:
        trees_attributes.append(tree_attr)
    overallNodeAdditions = 0  
    for attr in trees_attributes:
        T = nx.Graph()  
        for node1, node2, data in g.edges(data=True):
            if data['attr'] == attr:
                T.add_edge(node1, node2)

        if s in list(T.nodes):
            dfs_edge_order_list = list(nx.dfs_labeled_edges(T, s))  
            for n1, n2, label in dfs_edge_order_list.copy():
                if label == "nontree" or n1 == n2:  
                    dfs_edge_order_list.remove((n1, n2, label))
                    attribute_list[attr].remove((n1, n2, label))
            good_branch_nodes = set() 
            visited_nodes = set()  
            visited_nodes.add(dfs_edge_order_list[0][0])  
            delete_mode = False  
            for i in range(len(dfs_edge_order_list)):
                n1, n2, label = dfs_edge_order_list[i]
                if label == "forward":
                    visited_nodes.add(n2)
                elif label == "reverse":
                    visited_nodes.remove(n2)
                if label == "forward" or n2 in good_branch_nodes:
                    delete_mode = False
                if delete_mode:
                    attribute_list[attr].remove((n1, n2, "forward"))
                    attribute_list[attr].remove((n1, n2, "reverse"))
                    g[n1][n2]["attr"] = "0"
                    g.nodes[n2]["attr"] = "0" 
                if i < len(dfs_edge_order_list) - 1:
                    n1_next, n2_next, label_next = dfs_edge_order_list[i + 1]
                    if label == "forward" and label_next == "reverse" and str(n2) not in d_incidents and int(n2) not in d_incidents:
                        delete_mode = True
                    elif str(n2) in d_incidents or int(n2) in d_incidents:
                        [good_branch_nodes.update(visited_nodes)]
                        if edge_disjoint_paths is not None:
                            cnt = 0
                            for path in edge_disjoint_paths:
                                if n2 not in path:
                                    cnt += 1
                                else:
                                    break
                            if cnt == len(edge_disjoint_paths):
                                overallNodeAdditions += 1
                                trees_changed.add(attr)
    return trees_changed, overallNodeAdditions

# Create a tree with each node as the root, save it in a dictionary
# key = the root node and 
# value is sub-dictionary, save the path of each attr
def makeTree(g, shortest_path, d):
    node_list = list(g.nodes)
    tree_list = {}
    tree_list['main'] = shortest_path
    for s in shortest_path:
        if s != d:
            G_copy = g.copy()
            for n in node_list:
                G_copy.nodes[n]["attr"] = "0"
            G_copy.nodes[s]["attr"] = "s"
            G_copy.nodes[d]["attr"] = "d"
            try:
                edge_disjoint_paths = list(nx.edge_disjoint_paths(G_copy, s, d))
            except Exception as e:
                edge_disjoint_paths = []
            edge_disjoint_paths.sort(key=lambda x: len(x), reverse=False)
            no_path = 1
            for path in edge_disjoint_paths:
                for i in range(0, len(path) - 1):
                    if G_copy.nodes[path[i + 1]]["attr"] != "d":
                        G_copy.nodes[path[i + 1]]["attr"] = str(no_path)
                    G_copy[path[i]][path[i + 1]]["attr"] = str(no_path)
                no_path += 1
            makeMultipleTreesM(G_copy, edge_disjoint_paths)
            d_incidents = set()
            for d_edge in G_copy.edges(d):
                d_incidents.add(d_edge[1])
                G_copy[d_edge[0]][d_edge[1]]["attr"] = "-1"
            tree = {}
            # get all attribute of current graph
            trees_attributes = [] 
            for node1, node2, data in G_copy.edges(data=True):
                if data['attr'] not in trees_attributes:
                    if int(data['attr']) > 0:
                        trees_attributes.append(data['attr'])
            # save each attribute path in tree[attr]       
            for attr in trees_attributes:
                T = nx.Graph() 
                for node1, node2, data in G_copy.edges(data=True):
                    if data['attr'] == attr:
                        T.add_edge(node1, node2)
                tree[attr] = list(nx.dfs_labeled_edges(T, s))
            # delete branches that cannot reach the destination
            postProcessTreeM(G_copy, s, d_incidents, tree, edge_disjoint_paths, version=None, tree_attr=None)
            for attr in tree:
                for node1, node2, label in tree[attr].copy():
                    if label == "reverse":
                        tree[attr].remove((node1, node2, "reverse"))
            # Sort by the number of edges in the tree with different attributes
            sort_tree = sorted(tree.items(), key=lambda x:len(x[1]))
            tree_list[s] = dict(sort_tree)
    return tree_list
# Get the tree with the node "n" as the root
def getCurrentTreeofNode(n, tree_list):
    if n in tree_list.keys():
        return tree_list[n]
# get the attribute path of tree.
def getNextAttrPath(tree, attr):
# When getting the first attr, attr input 0, 
# if you want to get the next one just enter the attribute value, it will get the next value of the input attribute value
    if tree != None:
        if attr == 0:
            index = list(tree.keys())[attr]
            return tree[index]
        else:
            keys = list(tree.keys())
            attr = str(attr)
            try:
                next_index = keys.index(attr) + 1
                return tree[keys[next_index]]
            except IndexError:  # If the given key is the last
                return None
            except ValueError:  # If the given key is not in the dictionary
                return None
# get the next arc of "arc"
def getNextArc(attr_path, arc, H, route_list, fails):
# If the input arc is None, return to the first, if the arc is not empty, return to the next arc, H = 1 backward, H = 0 forward
# If the next one does not exist, return None.
    if attr_path != None:
        if arc == None:
            edge = attr_path[0]
            if (edge[0], edge[1]) not in fails and (edge[1], edge[0]) not in fails:
                route_list.append(edge)
            return attr_path[0]
        else:
            try:
                if H == 0:
                    for n_arc in attr_path:
                        # forward -> get the next arc started at node "arc[1]"
                        if n_arc[0] == arc[1] and n_arc[2] == "forward":
                            if (n_arc[0], n_arc[1]) not in fails and (n_arc[1], n_arc[0]) not in fails:
                                route_list.append(n_arc)
                            return n_arc
                    return None
                if H == 1:
                    # find the arc with an index less than the input index and also incoming of arc[0]
                    next_index = attr_path.index(arc) - 1
                    while next_index >= 0:
                        if attr_path[next_index][1] == arc[0]:
                            break
                        next_index -= 1
                if next_index < 0:
                    return None
                edge = attr_path[next_index] # backward -> the next arc is "edge"
                # and if this edge is not failed, add into the "route_list"
                if (edge[0], edge[1]) not in fails and (edge[1], edge[0]) not in fails:
                    if H == 1:
                        temp = (edge[1], edge[0], 'reverse')
                        edge = temp
                    route_list.append(edge)
                return attr_path[next_index]
            except IndexError: 
                return None
            except ValueError:
                return None
# Get the previous node of node "n" in the main path
def getNextNodeOnMainPath(n, shortest_path, H):
    if H == 1:
        if shortest_path.index(n)-1 >= 0:
            return shortest_path[shortest_path.index(n)-1]
    else:
        if shortest_path.index(n)+1 < len(shortest_path):
            return shortest_path[shortest_path.index(n)+1]
# Mark the edges prior to the 'edge' on the attribute path 'attr' in the current tree as failed.
def changeLabel(tree, attr, edge):
    tree_list = []
    isFailed = "failed"
    for e in tree[attr]:
        tree_list.append((e[0], e[1], isFailed))
        if e == edge:
            isFailed = "forward"
    tree[attr] = tree_list
# get the arc starting from node "n" and not failed in "path"
def hasOtherArcStartAt(path, n, route_list, fails):
    for edge in path:
        if edge[0] == n and edge[2] == "forward":
            if (edge[0], edge[1]) not in fails and (edge[1], edge[0]) not in fails:
                route_list.append(edge)
            return edge
    return None
def getAttrGivenEdge(tree, edge):
    for attr in tree.keys():
        if edge in tree[attr]:
            return attr

def route(H, tree, edge, fails, tree_list, d, d_neighbor, route_list):
    if edge[0] == None:
        # all nodes in the shortest path have been tried.
        return
    if (edge[0], edge[1]) in fails or (edge[1], edge[0]) in fails:
        if tree == None:
            # It is routing on the main path at this time.
            # and hit a failed link, then route this package on the tree
            route_list.pop()
            current_tree = getCurrentTreeofNode(edge[0], tree_list)
            if current_tree != {}:
                attr_path = getNextAttrPath(current_tree, attr=0) # get first attribute of current_tree
                next_edge = getNextArc(attr_path, None, 0, route_list, fails)
                route(0, current_tree, next_edge, fails, tree_list, d, d_neighbor, route_list)
            else:
                # If no tree exists with the current node as root -> go back
                shortest_path = tree_list['main']
                next_edge = (getNextNodeOnMainPath(edge[0], shortest_path, 1), edge[0], 'forward')
                if next_edge[0] != None:
                    route_list.append((next_edge[1], next_edge[0], 'reverse'))
                route(1, None, next_edge, fails, tree_list, d, d_neighbor, route_list)
        else: # currently it is routing on the tree, but hits a failed link
            attr = getAttrGivenEdge(tree, edge) # get the attributes of the current edge in the current tree
            changeLabel(tree, attr, edge) # the edges prior to the 'edge' on the attribute path 'attr' in the current tree as failed.
            next_edge = hasOtherArcStartAt(tree[attr], edge[0], route_list, fails) # check if there are other not failed(not tried) edges starting from the current node
            if next_edge != None: # yes -> keep routing forwards
                route(0, tree, next_edge, fails, tree_list, d, d_neighbor, route_list)
            else: # no
                back_edge = getNextArc(tree[attr], (edge[0], edge[1], "failed"), 1, route_list, fails) # check if there is a back path
                if back_edge != None: # yes -> go back
                    route(1, tree, back_edge, fails, tree_list, d, d_neighbor, route_list)
                else: # no -> it means currently the package is return to main path
                    next_attr_path = getNextAttrPath(tree, attr) # check if there is other attribute path has not been tried
                    if next_attr_path != None: # yes -> traverse this attribute path
                        next_edge = getNextArc(next_attr_path, None, 0, route_list, fails) # get the first edge of next attr_path
                        route(0, tree, next_edge, fails, tree_list, d, d_neighbor, route_list)
                    else: # no -> all path has been tried and failed. -> go back on the main path
                        shortest_path = tree_list['main']
                        next_edge = (getNextNodeOnMainPath(edge[0], shortest_path, 1), edge[0], 'forward')
                        if next_edge[0] != None:
                            route_list.append((next_edge[1], next_edge[0], 'reverse'))
                        route(1, None, next_edge, fails, tree_list, d, d_neighbor, route_list)
    elif tree == None:
        if H == 0:
            if edge[1] == d:
                # successful
                return
            else:
                # continue and go forward on the main path.
                shortest_path = tree_list['main']
                next_edge = (edge[1], getNextNodeOnMainPath(edge[1], shortest_path, 0), 'forward')
                route_list.append(next_edge)
                route(0, None, next_edge, fails, tree_list, d, d_neighbor, route_list)
        else:
            # go back on the main path
            current_tree = getCurrentTreeofNode(edge[0], tree_list)
            attr_path = getNextAttrPath(current_tree, attr=0) # get first attribute of current_tree
            next_edge = getNextArc(attr_path, None, 0, route_list, fails)
            route(0, current_tree, next_edge, fails, tree_list, d, d_neighbor, route_list)
    else:
        attr = getAttrGivenEdge(tree, edge)
        if edge[1] in d_neighbor and H == 0:
            # routing on the tree, and the current node is the neighbor of the destination.
            if (edge[1], d) in fails or (d, edge[1]) in fails: # but the link connected to the destination is failed
                changeLabel(tree, attr, edge)
                next_edge = hasOtherArcStartAt(tree[attr], edge[1], route_list, fails) # check if there are other not failed(not tried) links starting from the current node
                if next_edge != None: # yes -> go forwards
                    route(0, tree, next_edge, fails, tree_list, d, d_neighbor, route_list)
                else: # no -> go back
                    route_list.append((edge[1], edge[0], 'reverse'))
                    route(1, tree, (edge[0], edge[1], "failed"), fails, tree_list, d, d_neighbor, route_list)
            else:
                # go through the arc (edge[1],d) and finish the routing
                route_list.append((edge[1],d,'forward'))
                return  
        elif edge[1] not in d_neighbor and H == 0:
            # keep going
            next_edge = getNextArc(tree[attr], edge, 0, route_list, fails)
            route(0, tree, next_edge, fails, tree_list, d, d_neighbor, route_list)
        else: # route on a tree and go backwards
            next_edge = hasOtherArcStartAt(tree[attr], edge[0], route_list, fails) # check if there are other not failed(not tried) edges starting from the current node
            if next_edge != None: # yes -> go forwards
                route(0, tree, next_edge, fails, tree_list, d, d_neighbor, route_list)
            else:
                # go back on the tree
                back_edge = getNextArc(tree[attr], edge, 1, route_list, fails) 
                if back_edge != None:
                    route(1, tree, back_edge, fails, tree_list, d, d_neighbor, route_list)
                else:
                    # this attribute path has been tried, then try to route on other attribute path.
                    next_attr_path = getNextAttrPath(tree, attr) 
                    if next_attr_path != None: # has next atrribute
                        next_edge = getNextArc(next_attr_path, None, 0, route_list, fails) # get the first edge of next attr_path
                        route(0, tree, next_edge, fails, tree_list, d, d_neighbor, route_list)
                    else: # all path has been tried and failed. -> go back on the main path
                        shortest_path = tree_list['main']
                        next_edge = (getNextNodeOnMainPath(edge[0], shortest_path, 1), edge[0], 'forward')
                        if next_edge[0] != None:
                            route_list.append((next_edge[1], next_edge[0], 'reverse'))
                        route(1, None, next_edge, fails, tree_list, d, d_neighbor, route_list)



print("##################### Test #############################")
def main(p, no_nodes, rep, failureModel, edgeFailFactorPar = None, failurePercentPar=None, failureDropPar=None, graphMLpath = None):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    sum_hops = 0
    sum_hops_tree = 0
    sum_hops_edp = 0
    success_times = 0
    success_times_tree = 0
    success_times_edp = 0

    rep_copy = rep
    if graphMLpath:
        loadedML = nx.Graph(nx.read_graphml(graphMLpath))
        no_nodes = loadedML.number_of_nodes()
        p = graphMLpath
        if failureModel == "clustered":
            csv_name = "output_{}_{}/trees-n{}-ff{}.csv".format(graphMLpath, failureModel, no_nodes, failurePercentPar)
        else:
            csv_name = "output_{}_{}/trees-n{}-ff{}.csv".format(graphMLpath, failureModel, no_nodes, edgeFailFactorPar)
    else:
        if failureModel == "clustered":
            csv_name = "output/trees-cluster-p{}-n{}-ff{}.csv".format(p, no_nodes, failurePercentPar)
        elif failureModel == "adversarial":
            csv_name = "output/trees-adversarial-p{}-n{}-ff{}.csv".format(p, no_nodes, failurePercentPar)
        else:
            csv_name = "output/trees-random-p{}-n{}-ff{}.csv".format(p, no_nodes, edgeFailFactorPar) # default name is "random"

    writemode = 'w' if no_nodes == 25 else 'a'
    with open(csv_name, mode='w') as csv_file, open('runtime.csv', mode=writemode) as runtime_file:
        fieldnames = ["rep", "nrNodes", "nrEdges", "edgeProb", "nrPaths", "failure rate", "noFailed", "failureModel", "failurePercent", "hopsWithHeader", "hopsMultipleTree", "hopsOfEDPWithHeader", "Runtime of Tree with Header", "Runtime of EDP with Header", "Runtime of MutipleTree"]
        fieldnames_runtime = ['nodes', 'edges', 'edp', 'one', 'mult']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer2 = csv.DictWriter(runtime_file, fieldnames=fieldnames_runtime)
        writer.writeheader()
        writer2.writeheader()
        while rep != 0:
            if graphMLpath:
                g = loadedML
            else:
                # print("iteration", rep)
                g = nx.erdos_renyi_graph(no_nodes, p) 
                # g = nx.random_regular_graph(p, no_nodes, seed=None)
            # degrees = dict(g.degree())
            # # set average degree
            # avg_degree = sum(degrees.values()) / len(degrees)
            # if avg_degree > 2.5:
            #     continue
            csv_dict = {}
            edges = list(g.edges)
            csv_dict["rep"] = rep
            csv_dict["nrNodes"] = no_nodes
            csv_dict["nrEdges"] = len(edges)
            csv_dict["edgeProb"] = p
            # set graph
            nodes = list(g.nodes)
            s = random.choice(nodes)
            d = random.choice(nodes)
            while s == d or d in list(g.neighbors(s)):
                s = random.choice(nodes)
                d = random.choice(nodes)
            nx.set_edge_attributes(g, "0", "attr")
            nx.set_node_attributes(g, "0", "attr")
            g.nodes[s]["attr"] = "s"
            g.nodes[s]["label"] = "s"
            g.nodes[d]["attr"] = "d"
            g.nodes[d]["label"] = "d"
            
            try:
                shortest_path = nx.shortest_path(g.copy(), s, d)
                edge_disjoint_paths = list(nx.edge_disjoint_paths(g.copy(),s,d))
            except Exception as e:
                continue
            # to create the main path
            # to create failed links, according to the failureModel
            no_edge_disjoint_paths = len(edge_disjoint_paths)
            no_failed_edges = no_edge_disjoint_paths-1
            csv_dict["failureModel"] = failureModel
            csv_dict["failurePercent"] = -1
            csv_dict["nrPaths"] = no_edge_disjoint_paths
            if (edges):
                if failureModel == "random":           
                    if edgeFailFactorPar is None:
                        print(edgeFailFactorPar)
                        print(edgeFailFactorPar is None)
                        edgesToFail = no_failed_edges
                    else:
                        edgesToFail = int(round(no_failed_edges * edgeFailFactorPar))
                    try:
                        # randomly select edgesToFail faulty edges among all edges
                        failed_edges = random.sample(edges, edgesToFail)
                    except ValueError:
                        continue
                        # 并将failurePercent设置为-1，failurePercent在随机模式中不起作用，edgeFailFactorPar在其他模式中不起作用
                    csv_dict["failure rate"] = edgeFailFactorPar
                    csv_dict["failurePercent"] = -1
                elif failureModel == "adversarial":
                    # adversarial, faults occur only around the destination node
                    # First get all edges connected to d
                    d_incidents = list(g.edges(d))
                    # If failurePercentPar is not given, then randomize failurePercentPar to a number between 0.3 and 0.9.
                    if failurePercentPar is None:
                        #failurePercent = 0.8
                        failurePercent = random.randint(30, 90) / 100
                    else:
                        failurePercent = failurePercentPar
                    sampleNr = math.floor(failurePercent * len(d_incidents))
                    # Randomly select sampleNr edges among the edges directly connected to d as faulty edges.
                    failed_edges = random.sample(d_incidents, sampleNr)
                    csv_dict["failure rate"] = failurePercent
                elif failureModel == "clustered":
                    # clustered, it means that the destination node d is the center and the surrounding edges are affected, 
                    # and the farther away from d, the smaller the effect is.
                    failed_edges = []
                    # First get all the edges connected to d (destination node)
                    incidents = list(g.edges(d))
                    if failurePercentPar is None:
                        failureStart = 0.6
                    else:
                        failureStart = failurePercentPar
                    
                    if failureDropPar is None:
                        failureDrop = 0.3
                    else:
                        failureDrop = failureDropPar
                    csv_dict["failurePercent"] = str(failureStart) + ";-" + str(failureDrop)
                    failurePercent = failureStart
                    # Decrement failurePercent, starting at failureStart and decrementing failureDrop each time
                    # Starting from the destination node, with failurePercent probability, select the faulty edge that is directly connected to it, step by step.
                    # After updating failurePercent, with new failurePercent probability, select the edge directly connected to the neighbor of d, step by step to expand the influence range.
                    # But the farther away from d, the smaller the failure probability is
                    while failurePercent > 0.0:
                        # start selecting the edge directly connected to d with failureStart probability as the fault edge, 
                        # and store it in failed_edges (removing duplicate edges)
                        sampleNr = math.floor(failurePercent * len(incidents))
                        failed_edges.extend(random.sample(incidents, sampleNr))
                        failed_edges = list(dict.fromkeys(failed_edges)) #remove duplicates
                        # save all nodes connected to d in next_incidents (remove duplicates), and expand the scope of influence step by step,
                        next_incidents = []
                        for edge in incidents:
                            next_incidents.extend(list(g.edges(edge[1])))
                        
                        next_incidents = list(dict.fromkeys(next_incidents)) #remove duplicates
                        incidents = next_incidents
                        failurePercent -= failureDrop
                
                no_failed_edges = len(failed_edges)
                csv_dict["noFailed"] = no_failed_edges
                # print(f"failed_edges:{failed_edges}")
                # Create a tree with each node as the root, save it in a dictionary
                # key = the root node and 
                # value is sub-dictionary, save the path of each attr
                # try: 
                startMakeTree = timer()
                tree_list_tree = makeTree(g.copy(), shortest_path, d)
                endMakeTree = timer()
                startMakeEDP = timer()
                tree_list_edp = makeEDP(g.copy(), shortest_path, d)
                endMakeEDP = timer()
                timeMakeEDP = (endMakeEDP - startMakeEDP) * 1000
                timeMakeTree = (endMakeTree - startMakeTree) * 1000
                csv_dict["Runtime of Tree with Header"] = timeMakeTree
                csv_dict["Runtime of EDP with Header"] = timeMakeEDP
                # except Exception as e:
                #     continue
                first_edge = (shortest_path[0], shortest_path[1], 'forward')
                # print(f"main_path:{main_path}")
                # print(f"first_edge:{first_edge}")

                # whether the first edge is broken
                route_list_tree = [first_edge]
                route_list_edp = [first_edge]
                # Neighbor nodes of d
                d_neighbor = []
                for d_edge in g.edges(d):
                    d_neighbor.append(d_edge[1])
                # print(f"d_neighbor:{d_neighbor}")
                try:
                    route(0, None, first_edge, failed_edges, tree_list_tree, d, d_neighbor, route_list_tree)
                    routeEDP(0, None, first_edge, failed_edges, tree_list_edp, d, route_list_edp)
                except Exception as e:
                    print(e)
                hops, sp_length, overallNodeAdditions, hops_unranked, g_res_mult, routed_paths_mult, rankings_mult, tree_order_ranked_mult, timeEDP1, timeMult \
                            = kResilienceTrees(s, d, failed_edges, g.copy(), version="multiple", file_name=dt_string+"/run"+str(rep), unranked=True, draw=False, treeChoice="shortest")
                timeMutipleTree = timeMult
                csv_dict["Runtime of MutipleTree"] = timeMutipleTree
                # print("route_list")
                success = 0
                if len(route_list_tree) == 0 or route_list_tree[-1][1] != d:
                    success = -1
                    csv_dict["hopsWithHeader"] = -1
                else:
                    csv_dict["hopsWithHeader"] = len(route_list_tree)
                    sum_hops += len(route_list_tree)
                    success_times += 1

                if len(route_list_edp) == 0 or route_list_edp[-1][1] != d:
                    success_edp = -1
                    csv_dict["hopsOfEDPWithHeader"] = -1
                else:
                    csv_dict["hopsOfEDPWithHeader"] = len(route_list_edp)
                    sum_hops_edp += len(route_list_edp)
                    success_times_edp += 1
                
                if hops > 0:
                    sum_hops_tree += hops
                    success_times_tree += 1
                csv_dict["hopsMultipleTree"] = hops
                writer.writerow(csv_dict)
                # writer2.writerow(csv_dict_runtime)
            # print(rep)
            rep -= 1
    avg_hops = sum_hops/success_times
    avg_hops_tree = sum_hops_tree/success_times_tree
    avg_hops_edp = sum_hops_edp/success_times_edp
    resilience = success_times/rep_copy
    resilience_tree = success_times_tree/rep_copy
    resilience_edp = success_times_edp/rep_copy
    print(f"avg_hops:{avg_hops}, avg_hops_edp:{avg_hops_edp}, avg_hops_tree;{avg_hops_tree}")
    print(f"resilience:{resilience}, resilience_EDP:{resilience_edp}, resilience_tree:{resilience_tree}")
    return avg_hops, avg_hops_tree, avg_hops_edp, resilience, resilience_tree, resilience_edp
DEBUG = False

if __name__ == "__main__":
    graphMLpaths = ["graphml/AttMpls.graphml", "graphml/Cogentco.graphml", "graphml/Deltacom.graphml", "graphml/GtsCe.graphml", "graphml/Interoute.graphml", "graphml/Oteglobe.graphml"]
    graphMLpath = None
    csv_name = "output_graphml/Cogentco.graphml_clustered/summary.csv"
    # csv_name = "output/summary.csv"
    p = 0.15
    no_nodes = 10
    writemode = 'w' if no_nodes == 25 else 'a'
    repeats = 200
    # failpercent = 0.1
    edgeFailFactorPar = 1
    with open(csv_name, mode='w') as csv_file:
        fieldnames = ["Avg. Hops of Tree with Header", "Avg. Hops of EDP with Header", "Avg. Hops of MutipleTree", "Resilience of Tree with Header", "Resilience of EDP with Header", "Resilience of Tree"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        while failpercent <= 0.9:
        # while edgeFailFactorPar <= 25:
            print(failpercent)
            # print(edgeFailFactorPar)
            csv_dict = {}
            avg_hops, avg_hops_tree, avg_hops_edp, resilience, resilience_tree, resilience_edp = main(p, no_nodes, repeats, "clustered", failurePercentPar = failpercent, failureDropPar = 0.3, graphMLpath = graphMLpaths[1])
            # avg_hops, avg_hops_tree, avg_hops_edp, resilience, resilience_tree, resilience_edp = main(p, no_nodes, repeats, "random", edgeFailFactorPar = edgeFailFactorPar, graphMLpath = graphMLpaths[1])
            csv_dict["Avg. Hops of Tree with Header"] = avg_hops
            csv_dict["Avg. Hops of EDP with Header"] = avg_hops_edp
            csv_dict["Avg. Hops of MutipleTree"] = avg_hops_tree
            csv_dict["Resilience of Tree with Header"] = resilience
            csv_dict["Resilience of Tree"] = resilience_tree
            csv_dict["Resilience of EDP with Header"] = resilience_edp
            failpercent += 0.1
            failpercent = round(failpercent, 1)
            # edgeFailFactorPar += 1
            writer.writerow(csv_dict)


# def errorFinder():
#     G = nx.Graph()
#     G.add_nodes_from([(i, {"attr": "0"}) for i in range(0, 35)])
#     # G.nodes[16]["attr"] = "s"
#     # G.nodes[15]["attr"] = "d"
#     s = 16
#     d = 15
#     # G.add_edges_from([(1, 9, {"attr": "0"}),
#     #                 (9, 4, {"attr": "0"}),
#     #                 (9, 6, {"attr": "0"}),
#     #                 (1, 7, {"attr": "0"}),
#     #                 (2, 1, {"attr": "0"}),
#     #                 (4, 8, {"attr": "0"}),
#     #                 (6, 5, {"attr": "0"}),
#     #                 (4, 6, {"attr": "0"}),
#     #                 (2, 7, {"attr": "0"}),
#     #                 (7, 8, {"attr": "0"}),
#     #                 (2, 0, {"attr": "0"}),
#     #                 (8, 3, {"attr": "0"}),
#     #                 (3, 5, {"attr": "0"}),
#     #                 (0, 5, {"attr": "0"}),
#     #                 (0, 3, {"attr": "0"})])
#     G.add_edges_from([(0, 3, {'attr': '0'}),
#     (0, 19, {'attr': '0'}),
#     (0, 24, {'attr': '0'}),
#     (1, 6, {'attr': '0'}),
#     (1, 8, {'attr': '0'}),
#     (1, 10, {'attr': '0'}),
#     (1, 12, {'attr': '0'}),
#     (1, 15, {'attr': '0'}),
#     (1, 22, {'attr': '0'}),
#     (2, 3, {'attr': '0'}),
#     (2, 15, {'attr': '0'}),
#     (2, 16, {'attr': '0'}),
#     (2, 18, {'attr': '0'}),
#     (3, 5, {'attr': '0'}),
#     (3, 16, {'attr': '0'}),
#     (4, 10, {'attr': '0'}),
#     (4, 12, {'attr': '0'}),
#     (4, 22, {'attr': '0'}),
#     (5, 15, {'attr': '0'}),
#     (5, 22, {'attr': '0'}),
#     (5, 24, {'attr': '0'}),
#     (6, 8, {'attr': '0'}),
#     (6, 11, {'attr': '0'}),
#     (6, 12, {'attr': '0'}),
#     (8, 18, {'attr': '0'}),
#     (8, 23, {'attr': '0'}),
#     (9, 15, {'attr': '0'}),
#     (10, 11, {'attr': '0'}),
#     (10, 14, {'attr': '0'}),
#     (10, 23, {'attr': '0'}),
#     (11, 15, {'attr': '0'}),
#     (12, 16, {'attr': '0'}),
#     (12, 20, {'attr': '0'}),
#     (15, 17, {'attr': '0'}),
#     (15, 23, {'attr': '0'}),
#     (16, 23, {'attr': '0'}),
#     (16, 24, {'attr': '0'}),
#     (17, 21, {'attr': '0'}),
#     (17, 23, {'attr': '0'}),
#     (18, 21, {'attr': '0'}),
#     (20, 21, {'attr': '0'}),
#     (20, 23, {'attr': '0'})])
#     # G = nx.Graph(nx.read_graphml("src/graphml/AttMpls.graphml"))
#     # s = '6'
#     # d = '10'
#     # nx.set_edge_attributes(G, "0", "attr")
#     # nx.set_node_attributes(G, "0", "attr")
#     # # 创建 d 的邻居节点保存在 d_incidents (list) 中
#     G.nodes[s]["attr"] = "s"
#     G.nodes[s]["label"] = "s"
#     G.nodes[d]["attr"] = "d"
#     G.nodes[d]["label"] = "d"
#     d_incidents = []
#     for d_edge in G.edges(d):
#         d_incidents.append(d_edge[1])
#     print(f"d_incidents:{d_incidents}")
#     # 创建最短路径
#     shortest_path = nx.shortest_path(G.copy(), s, d)
#     failed_edges = [(9, 15), (1, 8), (10, 11), (1, 6), (10, 23), (1, 15), (17, 23), (6, 8), (0, 3), (11, 15), (0, 24), (6, 12), (16, 23), (4, 22), (0, 19), (5, 22), (20, 23), (2, 15), (15, 23), (5, 24), (3, 16), (8, 23), (5, 15), (4, 12), (12, 20), (1, 10), (20, 21), (16, 24)]
#     tree_list = makeEDP(G, shortest_path, d)
#     first_edge = (shortest_path[0], shortest_path[1], 'forward')
#     print(f"first_edge:{first_edge}")
#     if (first_edge[0], first_edge[1]) not in failed_edges and (first_edge[1], first_edge[0]) not in failed_edges:
#         route_list = [first_edge]
#     else:
#         route_list = list()
#     # try:
#     routeEDP(0, None, first_edge, failed_edges, tree_list, d, route_list)
#     # except Exception as e:
#     #     print("route_list")
#     #     print(route_list)
#     #     print(f"s:{s},d:{d}")
#     print(route_list)
#     hops, sp_length, overallNodeAdditions, hops_unranked, g_res_mult, routed_paths_mult, rankings_mult, tree_order_ranked_mult, timeEDP1, timeMult \
#                             = kResilienceTrees(s, d, failed_edges, G.copy(), version="multiple", unranked=True, draw=False, treeChoice="shortest")
#     print(f"hops:{hops}")
#     for node, data in g_res_mult.nodes(data=True):
#         print(f"Node {node}: {data}")
#     for edge in g_res_mult.edges(data=True):
#         print(f"Edge {edge}")
    
# errorFinder()
# def finaltest(repeat):
#     for i in range(1, repeat):
#         test(3, 10, 3, "clustered")
#         print(f"{i}+1")
#         test(3, 10, 3, "adversarial")
#         print(f"{i}+2")
#         test(3, 10, 3, "random")
#         print(f"{i}+3")




    # csv_name = "output_graphml/Oteglobe.graphml_random/summary.csv"
    # p = 0.15
    # no_nodes = 50
    # writemode = 'w' if no_nodes == 25 else 'a'
    # repeats = 200
    # edgeFailFactorPar = 1
    # with open(csv_name, mode='w') as csv_file:
    #     fieldnames = ["avg_hops", "avg_hops_tree", "resilience", "resilience_tree"]
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #     writer.writeheader()
    #     while edgeFailFactorPar <= 25:
    #         print(edgeFailFactorPar)
    #         csv_dict = {}
    #         avg_hops, avg_hops_tree, resilience, resilience_tree = main(p, no_nodes, repeats, "random", edgeFailFactorPar = edgeFailFactorPar, graphMLpath = graphMLpaths[5])
    #         csv_dict["avg_hops"] = avg_hops
    #         csv_dict["avg_hops_tree"] = avg_hops_tree
    #         csv_dict["resilience"] = resilience
    #         csv_dict["resilience_tree"] = resilience_tree
    #         writer.writerow(csv_dict)
    #         edgeFailFactorPar += 1
    # csv_name = "output/summary.csv"
    # p = 0.15
    # no_nodes = 50
    # writemode = 'w' if no_nodes == 25 else 'a'
    # repeats = 200
    # failpercent = 0.1
    # with open(csv_name, mode='w') as csv_file:
    #     fieldnames = ["avg_hops", "avg_hops_tree", "resilience", "resilience_tree"]
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #     writer.writeheader()
    #     while failpercent <= 0.9:
    #         print(failpercent)
    #         csv_dict = {}
    #         avg_hops, avg_hops_tree, resilience, resilience_tree = main(p, no_nodes, repeats, "clustered", failurePercentPar = failpercent, failureDropPar = 0.3)
    #         csv_dict["avg_hops"] = avg_hops
    #         csv_dict["avg_hops_tree"] = avg_hops_tree
    #         csv_dict["resilience"] = resilience
    #         csv_dict["resilience_tree"] = resilience_tree
    #         writer.writerow(csv_dict)
    #         failpercent += 0.1
    #         failpercent = round(failpercent, 1)
    # csv_name = "output_graphml/AttMpls.graphml_clustered/summary.csv"
    # p = 0.15
    # no_nodes = 50
    # writemode = 'w' if no_nodes == 25 else 'a'
    # repeats = 200
    # failpercent = 0.1
    # with open(csv_name, mode='w') as csv_file:
    #     fieldnames = ["avg_hops", "avg_hops_tree", "resilience", "resilience_tree"]
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #     writer.writeheader()
    #     while failpercent <= 1.0:
    #         print(failpercent)
    #         csv_dict = {}
    #         avg_hops, avg_hops_tree, resilience, resilience_tree = main(p, no_nodes, repeats, "clustered", failurePercentPar = failpercent, failureDropPar = 0.2, graphMLpath = graphMLpaths[0])
    #         csv_dict["avg_hops"] = avg_hops
    #         csv_dict["avg_hops_tree"] = avg_hops_tree
    #         csv_dict["resilience"] = resilience
    #         csv_dict["resilience_tree"] = resilience_tree
    #         writer.writerow(csv_dict)
    #         failpercent += 0.1
    #         failpercent = round(failpercent, 1)
    # csv_name = "output_graphml/Cogentco.graphml_clustered/summary.csv"
    # p = 0.15
    # no_nodes = 50
    # writemode = 'w' if no_nodes == 25 else 'a'
    # repeats = 200
    # failpercent = 0.1
    # with open(csv_name, mode='w') as csv_file:
    #     fieldnames = ["avg_hops", "avg_hops_tree", "resilience", "resilience_tree"]
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #     writer.writeheader()
    #     while failpercent <= 1.0:
    #         print(failpercent)
    #         csv_dict = {}
    #         avg_hops, avg_hops_tree, resilience, resilience_tree = main(p, no_nodes, repeats, "clustered", failurePercentPar = failpercent, failureDropPar = 0.2, graphMLpath = graphMLpaths[1])
    #         csv_dict["avg_hops"] = avg_hops
    #         csv_dict["avg_hops_tree"] = avg_hops_tree
    #         csv_dict["resilience"] = resilience
    #         csv_dict["resilience_tree"] = resilience_tree
    #         writer.writerow(csv_dict)
    #         failpercent += 0.1
    #         failpercent = round(failpercent, 1)
    # csv_name = "output_graphml/Oteglobe.graphml_clustered/summary.csv"
    # p = 0.15
    # no_nodes = 50
    # writemode = 'w' if no_nodes == 25 else 'a'
    # repeats = 200
    # failpercent = 0.1
    # with open(csv_name, mode='w') as csv_file:
    #     fieldnames = ["avg_hops", "avg_hops_tree", "resilience", "resilience_tree"]
    #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #     writer.writeheader()
    #     while failpercent <= 1.0:
    #         print(failpercent)
    #         csv_dict = {}
    #         avg_hops, avg_hops_tree, resilience, resilience_tree = main(p, no_nodes, repeats, "clustered", failurePercentPar = failpercent, failureDropPar = 0.3, graphMLpath = graphMLpaths[5])
    #         csv_dict["avg_hops"] = avg_hops
    #         csv_dict["avg_hops_tree"] = avg_hops_tree
    #         csv_dict["resilience"] = resilience
    #         csv_dict["resilience_tree"] = resilience_tree
    #         writer.writerow(csv_dict)
    #         failpercent += 0.1
    #         failpercent = round(failpercent, 1)
    # failpercent = 0.1
    # while failpercent <= 0.8:
    #     print(failpercent)
    #     main(p, no_nodes, repeats, "clustered", failurePercentPar = failpercent, failureDropPar = 0.2)
    #     failpercent += 0.1
    #     failpercent = round(failpercent, 1)
    
    # failpercent = 0.1
    # while failpercent <= 0.9:
    #     print(failpercent)
    #     main(p, no_nodes, repeats, "adversarial", failurePercentPar = failpercent, failureDropPar = 0.3)
    #     failpercent += 0.1
    #     failpercent = round(failpercent, 1)

    # main(p, no_nodes, rep, failureModel, edgeFailFactorPar = None, failurePercentPar=None, failureDropPar=None, graphMLpath = None)
    # while i > 0:
    #     edgeFailFactorPar = 8
    #     while edgeFailFactorPar <= 15:
    #         main(p, no_nodes, repeats, "cluster", failurePercentPar = edgeFailFactorPar)
    #         edgeFailFactorPar += 1
    #     i -= 1
        # edgeFailFactorPar = round(edgeFailFactorPar, 1)

    # graphMLpaths = ["graphml/AttMpls.graphml", "graphml/Cogentco.graphml", "graphml/Deltacom.graphml", "graphml/GtsCe.graphml", "graphml/Interoute.graphml", "graphml/Oteglobe.graphml"]
    #graphMLpath = graphMLpaths[5]
    #graphMLpath = None
    #treeChoices = ["shortest", "average", "edgeCount"]

    # """main(repeats, 8, no_nodes, failureModel="random", treeChoice="edgeCount", edgeFailFactorPar=None, failurePercentPar=0,
    #      failureDropPar=0, draw=False)"""

    # """while no_nodes<=105:
    #     main(repeats, p, no_nodes, failureModel="random", treeChoice="edgeCount", edgeFailFactorPar=0, failurePercentPar=0,
    #          failureDropPar=0, draw=False)
    #     no_nodes += 10"""

    # failpercent = 0.1
    # while failpercent <= 0.9:
    #     print(failpercent)
    #     main(3, no_nodes, 10, "clustered")
    #     failpercent += 0.1
    #     failpercent = round(failpercent, 1)