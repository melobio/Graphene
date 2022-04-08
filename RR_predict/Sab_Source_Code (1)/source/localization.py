#! /usr/bin/env python

"""
# -----------------------------------------------------------------------
#
# localization.py
#
# by Joerg Menche
# Last Modified: 2014-12-06
#
# This code determines the network-based distance and sepration for
# two given sets of nodes on given network as described in 
# 
# Uncovering Disease-Disease Relationships Through The Human
# Interactome
#
# by Joerg Menche, Amitabh Sharma, Maksim Kitsak, Susan Dina
#    Ghiassian, Marc Vidal, Joseph Loscalzo & Albert-Laszlo Barabasi
# 
# 
# -----------------------------------------------------------------------
# 
# 
# This program will calculate the size of the largest connected
# component S and mean shortest distance <d_s> for a given gene
# set. It will also compute the expected lcc size for the same number
# of randomly distributed genes.
# 
# * Required input:
# 
#   a file containing a gene set. The file must be in form of a table,
#   one gene per line. If the table contains several columns, they
#   must be tab-separated, only the first column will be used. See the
#   two files MS.txt and PD.txt for valid examples (they contain genes
#   for multiple sclerosis and peroxisomal disorders, respectively).
# 
# * Optional input:  
# 
#   - file containing an interaction network. If now file is given, the
#     default network \"interactome.tsv\" will be used instead. The file
#     must contain an edgelist provided as a tab-separated table. The
#     first two columns of the table will be interpreted as an
#     interaction gene1 <==> gene2
# 
#  - filename for the output. If none is given,
#    \"localiztion_results.txt\" will be used
#
#  - the number or random simulations can be chosen. Default is 1000,
#    which should run fast even for large gene sets and typically
#    gives good result. 
# 
# Here's an example that should work, provided the files are in the same
# directory as this python script:
# 
# ./localization.py -n interactome.tsv -g PD.txt -o output.txt
# 
#
# -----------------------------------------------------------------------
"""


import networkx as nx
import random 
import numpy as np
import optparse
import sys

import separation as tools


"""
# =============================================================================

           S T A R T   D E F I N I T I O N S 

# =============================================================================
"""

# =============================================================================
def print_usage(option, opt, value, parser):


    usage_message = """

# ----------------------------------------------------------------------

This program will calculate the network-based localization for a given
gene set

* Required input:

  one files containing a gene set. The file must be in form of a
  table, one gene per line. If the table contains several columns,
  they must be tab-separated, only the first column will be used. See
  the two files MS.txt and PD.txt for valid examples (the contain
  genes for multiple sclerosis and peroxisomal disorders).

* Optional input:  

  - file containing an interaction network. If now file is given, the
    default network \"interactome.tsv\" will be used instead. The file
    must contain an edgelist provided as a tab-separated table. The
    first two columns of the table will be interpreted as an
    interaction gene1 <==> gene2

  - filename for the output. If none is given,
    \"localiztion_results.txt\" will be used

  - the number or random simulations can be chosen. Default is 1000,
    which should run fast even for large gene sets and typically gives
    good result.


Here's an example that should work, provided the files are in the same
directory as this python script:

./localization.py -n interactome.tsv -g PD.txt -o output.txt

# ----------------------------------------------------------------------

    """

    print usage_message

    sys.exit()



# =================================================================================
def get_lcc_size(G,seed_nodes):
    """
    return the lcc size
    """

    # getting subgraph that only consists of the black_nodes
    g = nx.subgraph(G,seed_nodes)

    if g.number_of_nodes() != 0:
        # get all components 
        components = nx.connected_components(g)
        
        return len(components[0])

    else:
        return 0


# =============================================================================
def get_random_comparison(G,gene_set,sims):

    """
    gets the random expectation for the lcc size for a given gene set
    by drawing the same number of genes at random from the network

    PARAMETERS:
    -----------
        - G       : network
        - gene_set: dito
        - sims    : number of random simulations 

    RETURNS:
    --------
        - a string containing the results

    """

    # getting all genes in the network  
    all_genes = G.nodes()

    number_of_seed_genes = len(gene_set & set(all_genes))
    
    l_list  = []

    # simulations with randomly distributed seed nodes
    print ""
    for i in range(1,sims+1):
        # print out status
        if i % 100 == 0:
            sys.stdout.write("> random simulation [%s of %s]\r" % (i,sims))
            sys.stdout.flush()

        # get random seeds
        rand_seeds = set(random.sample(all_genes,number_of_seed_genes))

        # get rand lcc
        lcc = get_lcc_size(G,rand_seeds)
        l_list.append(lcc) 
        

    # get the actual value
    lcc_observed = get_lcc_size(G,gene_set)

    # get the lcc z-score:
    l_mean = np.mean(l_list)
    l_std  = np.std(l_list)

    if l_std == 0:
        z_score = 'not available'
    else:
        z_score = (1.*lcc_observed - l_mean)/l_std

    results_message = """
> Random expectation:
> lcc [rand] = %s
> => z-score of observed lcc = %s
"""%(l_mean,z_score)

    return results_message


"""
# =============================================================================

           E N D    O F    D E F I N I T I O N S 

# =============================================================================
"""


if __name__ == '__main__':

    # "Hey Ho, Let's go!" -- The Ramones (1976)

    # --------------------------------------------------------
    # 
    # PARSING THE COMMAND LINE
    # 
    # --------------------------------------------------------

    parser = optparse.OptionParser()

    parser.add_option('-u', '--usage',
                      help    ='print more info on how to use this script',
                      action="callback", callback=print_usage)

    parser.add_option('-n',
                      help    ='file containing the network edgelist [interactome.tsv]',
                      dest    ='network_file',
                      default ='interactome.tsv',
                      type    = "string")

    parser.add_option('-g',
                      help    ='file containing gene set',
                      dest    ='gene_file',
                      default ='none',
                      type    = "string")

    parser.add_option('-s',
                      help    ='number of random simulations [1000]',
                      dest    ='sims',
                      default ='1000',
                      type    = "int")

    parser.add_option('-o',
                      help    ='file for results [separation_results.txt]',
                      dest    ='results_file',
                      default ='localization_results.txt',
                      type    = "string")


    (opts, args) = parser.parse_args()

    network_file = opts.network_file
    gene_file    = opts.gene_file
    results_file = opts.results_file
    sims         = opts.sims

    # checking for input:
    if gene_file == 'none':
        error_message = """
        ERROR: you must specify an input file with a gene set, for example:
        ./localization.py -g MS.txt

        For more information, type
        ./localization.py --usage
        
        """
        print error_message
        sys.exit(0)

    if network_file == 'interactome.tsv':
        print '> default network from "interactome.tsv" will be used'


    # --------------------------------------------------------
    #
    # LOADING NETWORK and DISEASE GENES
    #
    # --------------------------------------------------------

    # read network
    G  = tools.read_network(network_file)
    # get all genes ad remove self links
    all_genes_in_network = set(G.nodes())
    tools.remove_self_links(G)

    # read gene set
    gene_set_full = tools.read_gene_list(gene_file)
    # removing genes that are not in the network:
    gene_set = gene_set_full & all_genes_in_network
    if len(gene_set_full) != len(gene_set):
        print "> ignoring %s genes that are not in the network" %(
            len(gene_set_full - all_genes_in_network))
        print "> remaining number of genes: %s" %(len(gene_set))


    # --------------------------------------------------------
    #
    # CALCULATE NETWORK QUANTITIES
    #
    # --------------------------------------------------------

    # get lcc size S
    lcc = get_lcc_size(G,gene_set)
    print "\n> lcc size = %s" %(lcc)

    # get mean shortest distance
    d_s = tools.calc_single_set_distance(G,gene_set)
    print "> mean shortest distance = %s" %(d_s)

    results_message = """
> gene set from \"%s\": %s genes
> lcc size   S = %s
> diameter d_s = %s
"""%(gene_file,len(gene_set),lcc,d_s)

    # --------------------------------------------------------
    #
    # CALCULATE RANDOM COMPARISON
    #
    # --------------------------------------------------------

    results_message += get_random_comparison(G,gene_set,sims)

    print results_message
    
    fp = open(results_file,'w')
    fp.write(results_message)
    fp.close()

    print "> results have been saved to %s" % (results_file)






    
    


