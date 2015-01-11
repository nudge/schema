# schema
A Python implementation of SCHEMA - An Algorithm for Automated Product Taxonomy Mapping in E-commerce.

Based of the SCHEMA algorithm proposed by Aanen, et al.

Ref: http://disi.unitn.it/~p2p/RelatedWork/Matching/Aanen_eswc_2012.pdf

Author: David Ng <david@theopenlabel.com>, <nudgeee@gmail.com>


### Required modules
* nltk with wordnet download (see http://www.nltk.org/data.html)
* Levenshtein
* pyxdameraulevenshtein
 

### Example usage
    import schema
 
    # create source and candidate paths
    source_path     =  schema.Path().add_node(..)
    candidate_paths = [schema.Path().add_node(..), ..]

    # generate key paths and match
    keypathgen = schema.KeyPathGenerator(source_path, candidate_paths)
    source_key_path                  = keypathgen.source_key_path()
    matched_key_paths, matched_paths = keypathgen.matched_candidate_key_paths()

    # rank and print results
    ranker = schema.KeyPathRanker()
    for i,candidate_key_path in enumerate(matched_key_paths):
        rank = ranker.rank(source_key_path, candidate_key_path)
        print rank, matched_paths[i]
        
