#This snippet of code does multiple sequence alignment and builds a phylogenetic tree.

library(msa)
library(ape)
mySequences <- readAAStringSet("simple.fasta") 
myClustalWAlignment <- msa(mySequences, "ClustalW")
clustal = msaConvert(myClustalWAlignment, "seqinr::alignment")

unrootedNJtree <- function(alignment,type)
{
  # this function requires the ape and seqinR packages:
  require("ape")
  require("seqinr")
  # define a function for making a tree:
  makemytree <- function(alignmentmat)
  {
    alignment <- ape::as.alignment(alignmentmat)
    if      (type == "protein")
    {
      mydist <- dist.alignment(alignment)
    }
    else if (type == "DNA")
    {
      alignmentbin <- as.DNAbin(alignment) #transform a set of DNA sequences among 
      # various internal formats
      mydist <- dist.dna(alignmentbin) #computes a matrix of pairwise distances from 
      # DNA sequences using a model of DNA evolution. 
      #Eleven substitution models (and the raw distance) are currently available.
    }
    print(mydist)
    mytree <- njs(mydist) #performs the neighbor-joining tree estimation of Saitou and Nei (1987).
    mytree <- makeLabel(mytree, space="") # get rid of spaces in tip names.
    return(mytree)
  }
  # infer a tree
  mymat  <- as.matrix.alignment(alignment)
  mytree <- makemytree(mymat)
  # bootstrap the tree
  # myboot <- boot.phylo(mytree, mymat, makemytree)
  # plot the tree:
  plot.phylo(mytree,type="p")   # plot the unrooted phylogenetic tree
  # nodelabels(myboot,cex=0.7)    # plot the bootstrap values
  mytree$node.label <- myboot   # make the bootstrap values be the node labels
  return(mytree)
}

tree <- unrootedNJtree(clustal,type="DNA")

