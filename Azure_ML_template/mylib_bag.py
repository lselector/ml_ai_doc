"""
# mylib_bag.py - MyBunch class
# by Lev Selector, 2012-2021
# offers generic dict-like class MyBunch.
# Convenient to use from iPython prompt.
# Helps to verify data under debugger.
# Helps to decouple parts of software system
# (works like internal messaging system).
# 
# note - needs python version 3.3 or higher
"""

import os, sys, re
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
import pandas as pd
import importlib

# --------------------------------------------------------------
class MyBunch(dict):
    """
    # class for a simple container, inherits from a dict(),
    # but points its __dict__ to itself (1 dict instedad of 2).
    # Thus you can access members as obj['kk'] and obj.kk
    # auto-completion works correctly under iPython
    #     bag = MyBunch()
    #     bag = MyBunch(apples = 1, pears = 2)
    # --------------
    # Note: 
    # This simple __init__() is from this old (March 2001) discussion started by Alex Martelli:
    #     http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/
    # --------------
    # Note:
    # there is a more advanced implementation which we don't use 
    # because it doesn't work correctly under iPython:
    #     http://pypi.python.org/pypi/bunch/
    # --------------
    # Note:
    # methods __getstate__ and __setstate__ are added 
    # because they are used by the pickle protocol
    # to maintain the equivalence between self and self.__dict__ 
    # --------------
    # Note:
    # you can attach methods to a bag like this:
    # def func_name(bag,message):
    #     print("message")
    # bag.method_name = types.MethodType(func_name, bag)
    """
    # -----------------------------------
    def __init__(self,**kw):
        dict.__init__(self,kw)
        self.__dict__ = self

    # -----------------------------------
    def __getstate__(self):      # 
        return self 
 
    # -----------------------------------
    def __setstate__(self, state): 
        self.update(state) 
        self.__dict__ = self   

    # -----------------------------------
    def __repr__(self):
        """
        # string representation for interactive use
        """
        ks = sorted([ x for x in self.keys() ])
        if len(ks) <= 0:
            return ""
        ss = "\n"
        len_n_elems = 100
        len_list_str = 50
        mytab = max(len(w) for w in ks)
        for kk in ks:
            aa = self[kk]
            # -----------------------------------
            if callable(aa):
                ss += "%s()\n" % kk
                continue
            # -----------------------------------
            format_str = "%-"+str(mytab)+"s = "
            ss += format_str % kk
            # -----------------------------------
            if type(aa) == pd.DataFrame:
                ss += "(df - %d rows)\n" % (len(aa))
            # -----------------------------------
            elif type(aa).__name__ == 'MyBunch':
                ss += "(MyBunch - %d elems)\n" % (len(aa))
            # -----------------------------------
            elif type(aa) in [set, list, dict, tuple]:
                mylen = len(aa)
                if (type(aa) == list):
                    mystr = str(aa[:len_n_elems])           # take no more than few elements
                    word = 'list'
                elif (type(aa) == tuple):
                    mystr = str(aa[:len_n_elems])           # take no more than few elements
                    word = 'tuple'
                elif (type(aa) == dict):
                    nn=0
                    mydict = dict()
                    for k1,v1 in aa.items():
                        mydict[k1] = v1
                        nn += 1
                        if nn >= len_n_elems:
                            break
                    mystr = str(mydict)                     # string repr of a dictionary
                    word = 'dict'
                elif (type(aa) == set):
                    mystr = str(list(aa)[:len_n_elems])     # take no more than few elements
                    word = 'set'
                if len(mystr) >= (len_list_str + 4):        # if it takes too much space
                    mystr = mystr[:len_list_str] + " ..."   #   cut it and add ...
                ss += "%s %d elems: %s\n" % (word, mylen, mystr)
            # -----------------------------------
            elif type(aa) in [str]:
                tmp = aa
                nnn = len(tmp)
                if nnn > 75:
                    tmp = tmp[:75] + " ... " + " (len(str) = %d)" % nnn
                ss += "%r\n" % (tmp)
            # -----------------------------------
            elif ("__str4bag__" in dir(aa)) and callable(getattr(aa, "__str4bag__")):
                ss += aa.__str4bag__() + "\n"
            # -----------------------------------
            else:
                ss += str(aa) + "\n"
            # -----------------------------------

        return ss

    # -----------------------------------
    def __str__(self):
        """
        # used in provide a list of members
        """
        return self.__repr__()

# --------------------------------------------------------------
def test_avail(container, mem_str):
    """
    # recursive procedure to test if bunch members exist 
    """
    mem_str = mem_str.strip()
    if not len(mem_str):
        return False
    elems = mem_str.split('.')
    first_elem = elems.pop(0)
    if first_elem not in container:
        return False
    elif len(elems) >=1:
        return test_avail(container[first_elem], '.'.join(elems))
    else:
        return True
        # thing = container[first_elem]
        # mytype = str(type(thing))
        # if re.search('DataFrame', mytype): # even empty dataframe should return True
        #    return True
        # return bool(thing)
