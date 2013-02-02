__author__ = 'me'



#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py

import pstats, cProfile

import pyximport
pyximport.install()

import sobol_seq
import csobol_seq

cProfile.runctx("sobol_seq.i4_sobol_generate(20,100000,1000)", globals(), locals(), "Profiler.prof")

s = pstats.Stats("Profiler.prof")
s.strip_dirs().sort_stats("time").print_stats()


cProfile.runctx("csobol_seq.i4_sobol_generate(20,100000,1000)", globals(), locals(), "CProfiler.prof")

s = pstats.Stats("CProfiler.prof")
s.strip_dirs().sort_stats("time").print_stats()

