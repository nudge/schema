from __future__ import division

import itertools, re
import Levenshtein
import nltk
from nltk.corpus import wordnet as wn

#
# Python implementation of SCHEMA - An Algorithm for Automated Product Taxonomy
# Mapping in E-commerce.
#
# Ref: http://disi.unitn.it/~p2p/RelatedWork/Matching/Aanen_eswc_2012.pdf
#
# Copyright 2015 David Ng <david@theopenlabel.com>, <nudgeee@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" A Python implementation of SCHEMA - An Algorithm for Automated Product
    Taxonomy Mapping in E-commerce.

    Based of the SCHEMA algorithm proposed by Aanen, et al.
    Ref: http://disi.unitn.it/~p2p/RelatedWork/Matching/Aanen_eswc_2012.pdf

    Author: David Ng <david@theopenlabel.com>, <nudgeee@gmail.com>


    Required modules:
    
      * nltk with wordnet download
      * Levenshtein


    Example usage:

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

    """

# -----------------------------------------------------------------------------
# Utility functions                                                            
# -----------------------------------------------------------------------------

def _split_composite(w):
    ''' Splits composite category name w into a set of individual classes:
    a split term set W. '''
    m = re.split(', | & | and | ', w)
    return set([s.lower() for s in m])

def _longest_common_substring(wa, wb):
    ''' Which computes the length of the longest common sequence of consecutive
    characters between two strings, corrected for length of the longest string,
    resulting in an index in the range [0; 1]. '''
    (s1, s2) = (wa, wb) if len(wa) > len(wb) else (wb, wa)
    m = [[0] * (1 + len(s2)) for i in xrange(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in xrange(1, 1 + len(s1)):
        for y in xrange(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    lcs = s1[x_longest - longest: x_longest]
    return len(lcs)/len(s1)

def _contains_as_separate_component(wa, wb):
    ''' Indicates whether string wa contains string wb as separate part (middle
    of another word is not suffcient) '''
    if wb in wa:
        return True
    return False

# -----------------------------------------------------------------------------
# Find Source Category's Extended Split Term Set                               
# -----------------------------------------------------------------------------

class ExtendedSplitTermSet(object):
    ''' Generates a split term set for the given category, using parent
        and children as context. '''

    def __init__(self, wcategory, wparent, Wchildren):
        self.wcategory = wcategory
        self.wparent = wparent
        self.Wchildren = Wchildren

    def split_terms(self):
        ''' Returns the extended split term set. '''
        Wcategory=_split_composite(self.wcategory)
        Wparent=_split_composite(self.wparent)
        Wchild=set()
        for w in self.Wchildren:
            Wchild=Wchild | _split_composite(w)
        Wcontext=Wchild | Wparent
        extendedSplitTermSet=set()
        for wsrcSplit in Wcategory:
            extendedTermSet=self.disambiguate(wsrcSplit, Wcontext)
            if extendedTermSet:
                extendedTermSet=set([l.name() for l in extendedTermSet.lemmas()])
                extendedTermSet=extendedTermSet | set([wsrcSplit])
                extendedSplitTermSet=extendedSplitTermSet | extendedTermSet
        return extendedSplitTermSet

    def disambiguate(self, w, Wcontext):
        ''' Disambiguates a word using a set of context words, resulting in a set of
        correct synonyms. '''
        z=self.get_synsets(w)
        bestscore=0
        bestsynset=None
        for s in z:
            sensescore=0
            r=set(self.get_related(s))
            p=itertools.product(r, Wcontext)
            for (sr, w) in p:
                gloss=self.get_gloss(sr)
                sensescore+=_longest_common_substring(gloss, w)
            if sensescore>bestscore:
                bestscore=sensescore
                bestsynset=s
        return bestsynset

    def get_synsets(self, w):
        ''' Gives all synonym sets (representing one sense in WordNet), of which word
        w is a member. '''
        return wn.synsets(w)

    def get_related(self, S):
        ''' Gives synonym sets directly related to synset S in WordNet, based on
        hypernymy, hyponymy, meronymy and holonymy. Result includes synset S as well.
        '''
        related=[]
        related.extend(S.hypernyms())
        related.extend(S.hyponyms())
        related.extend(S.part_meronyms())
        related.extend(S.part_holonyms())
        return related

    def get_gloss(self, S):
        ''' Returns the gloss associated to a synset S in WordNet. '''
        return S.definition()

# -----------------------------------------------------------------------------
# Semantic Match                                                               
# -----------------------------------------------------------------------------

class SemanticMatcher(object):
    ''' Semantic matcher class. '''

    def match(self, E, wtarget, tnode):
        ''' Returns true if a semantic match exists between the ExtendedSplitTermSet (E)
        and wtarget, with a node matching threshold specified by tnode. '''
        Wtarget = _split_composite(wtarget)
        subSetOf = True
        for SsrcSplit in E:
            matchFound = False
            p=itertools.product([SsrcSplit], Wtarget)
            for (wsrcSplitSyn, wtargetSplit) in p:
                edit_dist = Levenshtein.distance(unicode(wsrcSplitSyn), unicode(wtargetSplit))
                similarity = 1 - edit_dist / max(len(wsrcSplitSyn), len(wtargetSplit))
                if _contains_as_separate_component(wsrcSplitSyn, wtargetSplit):
                    matchFound = True
                elif similarity >= tnode:
                    matchFound = True
            if matchFound == False:
                subSetOf = False
        return subSetOf


