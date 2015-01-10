# schema
A Python implementation of SCHEMA - An Algorithm for Automated Product Taxonomy Mapping in E-commerce.

Based of the SCHEMA algorithm proposed by Aanen, et al.

Ref: http://disi.unitn.it/~p2p/RelatedWork/Matching/Aanen_eswc_2012.pdf

Author: David Ng <david@theopenlabel.com>, <nudgeee@gmail.com>


### Required modules
* nltk with wordnet download
* Levenshtein
 
### Example usage
    import schema
    tnode = 0.8
    wparent = "Cheese & Cheese Alternatives"
    wcategory = "Cottage Cheese"
    Wchildren = []
    wtarget = ...
    
    matcher = schema.SemanticMatcher()
    e = schema.ExtendedSplitTermSet(wcategory, wparent, Wchildren)
    E = e.split_terms()
    m = matcher.match(E, wtarget, tnode)
