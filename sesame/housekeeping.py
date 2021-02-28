# coding=utf-8
# Copyright 2018 Swabha Swayamdipta. All rights reserved.
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
from __future__ import division

import random
import sys

from .globalconfig import UNK

class FspDict:
    def __init__(self):
        self._strtoint = {}
        self._inttostr = {}
        self._locked = False
        self._posttrainlocked = False
        self._singletons = set([])
        self._unseens = set([])
        self._unks = set([])  # these are vocabulary items which were not in train, so we don't know parameters for.

    def addstr(self, itemstr):
        if self._posttrainlocked and itemstr not in self._strtoint:
            self._unks.add(itemstr)
        if self._locked:
            if itemstr in self._strtoint:
                return self.getid(itemstr)
            self._unseens.add(itemstr)  # rpt handles the repeated, discontinuous FEs(R-FEs, under Propbank style)
            return self._strtoint[UNK]
        else:
            if itemstr not in self._strtoint:
                idforstr = len(self._strtoint)
                self._strtoint[itemstr] = idforstr
                self._inttostr[idforstr] = itemstr
                self._singletons.add(idforstr)
                return idforstr
            else:
                idforstr = self.getid(itemstr)
                if self.is_singleton(idforstr):
                    self._singletons.remove(idforstr)
                return idforstr

    def remove_extras(self, extras):
        for e in extras:
            eid = self._strtoint[e]
            del self._strtoint[e]
            del self._inttostr[eid]
            if eid in self._singletons:
                self._singletons.remove(eid)
                # no need to remove from unks because the repeat extras were never added to it

    def getid(self, itemstr):
        if itemstr in self._strtoint:
            return self._strtoint[itemstr]
        elif self._locked:
            return self._strtoint[UNK]
        else:
            raise Exception("not in dictionary, but can be added", id)

    def getstr(self, itemid):
        if itemid in self._inttostr:
            return self._inttostr[itemid]
        else:
            raise Exception("not in dictionary", itemid)

    def printdict(self):
        print(sorted(self._strtoint.keys()))

    def size(self):
        if not self._locked:
            raise Exception("dictionary still modifiable")
        return len(self._strtoint)

    def lock(self):
        if self._locked:
            raise Exception("dictionary already locked!")
        self.addstr(UNK)
        self._locked = True
        self._unseens = set([])

    def post_train_lock(self):
        if self._posttrainlocked:
            raise Exception("dictionary already post-train-locked!")
        self._posttrainlocked = True
        self._unks = set([])

    def islocked(self):
        return self._locked

    def is_singleton(self, idforstr):
        if idforstr in self._singletons:
            return True
        return False

    def num_unks(self):
        """
        :return: Number of unknowns attempted to be added to dictionary
        """
        return len(self._unseens), len(self._unks)

    def getidset(self):
        unkset = {self._strtoint[UNK]}
        fullset = set(self._inttostr.keys())
        return list(fullset - unkset)


def unk_replace_tokens(tokens, replaced, vocdict, unkprob, unktoken):
    """
    replaces singleton tokens in the train set with UNK with a probability UNK_PROB
    :param tokens: original token IDs
    :param replaced: replaced token IDs
    :return:
    """
    for t in tokens:
        if vocdict.is_singleton(t) and random.random() < unkprob:
            replaced.append(unktoken)
        else:
            replaced.append(t)


def extract_spans(indices):
    """
    Handles discontinuous, repeated FEs.
    In PropBank, the equivalent is reference-style arguments, like R-A0
    :param indices: list of array indices with the same FE
    :return: list of tuples containing argument spans
    """
    indices.sort()
    spans = [(indices[0], indices[0])]
    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            o = spans.pop()
            spans.append((o[0], indices[i]))
        else:
            spans.append((indices[i], indices[i]))
    return spans


def merge_span_ex(dataset, notanfeid):
    n_total = 0
    n_combined = 0
    for ex in dataset:
        for feid in ex.invertedfes:
            spans = sorted(ex.invertedfes[feid])
            spans_new = [spans[0]]
            n_total += 1
            for span in spans[1:]:
                n_total += 1
                last_span = spans_new[-1]
                if last_span[1] + 1 == span[0]:
                    print ("Combine {} and {}".format(last_span, span))
                    n_combined += 1
                    spans_new[-1] = (last_span[0], span[1])
                else:
                    spans_new.append(span)
            ex.invertedfes[feid] = spans_new
    sys.stderr.write("total spans: " + str(n_total) + "\n")
    sys.stderr.write("spans combined: " + str(n_combined) + "\n")
    return dataset


def filter_long_ex(dataset, use_span_clip, allowed_spanlen, notanfeid):
    if not use_span_clip:
        sys.stderr.write("\nfiltering out training examples with spans longer than " + str(allowed_spanlen) + "...\n")
    else:
        sys.stderr.write("\nclipping spans longer than " + str(allowed_spanlen) + "...\n")
    longestspan = 0
    longestfespan = 0
    tmpdataset = []
    for ex in dataset:
        haslongfe = False
        for feid in ex.invertedfes:
            haslongspans = False

            for span in ex.invertedfes[feid]:
                spanlen = span[1] - span[0] + 1
                if spanlen > allowed_spanlen:
                    haslongspans = True
                    haslongfe = True
                    if spanlen > longestspan:
                        longestspan = spanlen
                    if feid != notanfeid and spanlen > longestfespan:
                        longestfespan = spanlen

            if haslongspans and use_span_clip:
                clip_long_spans(ex.invertedfes[feid], allowed_spanlen)

        if haslongfe and not use_span_clip:
            continue  # filter out long examples
        tmpdataset.append(ex)
    sys.stderr.write("longest span size: " + str(longestspan) + "\n")
    sys.stderr.write("longest FE span size: " + str(longestfespan) + "\n")
    sys.stderr.write("# train examples before filter: " + str(len(dataset)) + "\n")
    sys.stderr.write("# train examples after filter: " + str(len(tmpdataset)) + "\n\n")
    return tmpdataset


def clip_long_spans(spans, maxspanlen):
    faultyspans = []
    for i in range(len(spans)):
        span = spans[i]
        spanlen = span[1] - span[0] + 1
        if spanlen <= maxspanlen:
            continue
        faultyspans.append(span)

    if len(faultyspans) == 0:
        return spans
    for span in faultyspans:
        spanlen = span[1] - span[0] + 1
        numbreaks = spanlen // maxspanlen
        newspans = []
        spanbeg = span[0]
        for x in range(numbreaks):
            newspans.append((spanbeg, spanbeg + maxspanlen - 1))
            spanbeg += maxspanlen
        if spanlen % maxspanlen != 0:
            newspans.append((span[0] + numbreaks * maxspanlen, span[1]))
        spans.remove(span)
        spans.extend(newspans)
    spans.sort()


class Factor(object):
    def __init__(self, beg, end, label):
        self.begin = beg
        self.end = end
        self.label = label

    def to_str(self, fedict):
        return str(self.begin) + "\t" + str(self.end) + "\t" + fedict.getstr(self.label)

    def unlabeled_eq(self, other):
        return self.begin == other.begin and self.end == other.end

    def __hash__(self):
        return hash((self.begin, self.end, self.label))

    def __eq__(self, other):
        return self.begin == other.begin and self.end == other.end and self.label == other.label

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)
