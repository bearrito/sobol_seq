
from __future__ import division
import cython
import numpy as numpy
import multiprocessing
cimport numpy as numpy
from cython.parallel cimport *
import math
import threading

from I4_State import I4_State

__all__ = ['i4_bit_hi1', 'i4_bit_lo0', 'i4_sobol_generate', 'i4_sobol', 'i4_uniform', 'prime_ge', 'is_prime']


def i4_bit_hi1 ( n ):
    """
    i4_bit_hi1 returns the position of the high 1 bit base 2 in an integer.

    Example:
      +------+-------------+-----
      |    N |      Binary | BIT
      +------|-------------+-----
      |    0 |           0 |   0
      |    1 |           1 |   1
      |    2 |          10 |   2
      |    3 |          11 |   2
      |    4 |         100 |   3
      |    5 |         101 |   3
      |    6 |         110 |   3
      |    7 |         111 |   3
      |    8 |        1000 |   4
      |    9 |        1001 |   4
      |   10 |        1010 |   4
      |   11 |        1011 |   4
      |   12 |        1100 |   4
      |   13 |        1101 |   4
      |   14 |        1110 |   4
      |   15 |        1111 |   4
      |   16 |       10000 |   5
      |   17 |       10001 |   5
      | 1023 |  1111111111 |  10
      | 1024 | 10000000000 |  11
      | 1025 | 10000000001 |  11

    Parameters:
      Input, integer N, the integer to be measured.
      N should be nonnegative.  If N is nonpositive, the value will always be 0.

      Output, integer BIT, the number of bits base 2.
    """
    i = math.floor ( n )
    bit = 0
    while 1:
        if i <= 0:
            break
        bit += 1
        i = math.floor ( i / 2. )
    return bit

#clear of globals
def i4_bit_lo0 ( n ):
    """
    I4_BIT_LO0 returns the position of the low 0 bit base 2 in an integer.

    Example:
      +------+------------+----
      |    N |     Binary | BIT
      +------+------------+----
      |    0 |          0 |   1
      |    1 |          1 |   2
      |    2 |         10 |   1
      |    3 |         11 |   3
      |    4 |        100 |   1
      |    5 |        101 |   2
      |    6 |        110 |   1
      |    7 |        111 |   4
      |    8 |       1000 |   1
      |    9 |       1001 |   2
      |   10 |       1010 |   1
      |   11 |       1011 |   3
      |   12 |       1100 |   1
      |   13 |       1101 |   2
      |   14 |       1110 |   1
      |   15 |       1111 |   5
      |   16 |      10000 |   1
      |   17 |      10001 |   2
      | 1023 | 1111111111 |   1
      | 1024 | 0000000000 |   1
      | 1025 | 0000000001 |   1

    Parameters:
      Input, integer N, the integer to be measured.
      N should be nonnegative.

      Output, integer BIT, the position of the low 1 bit.
    """
    bit = 0
    i = math.floor ( n )
    while 1:
        bit +=  1
        i2 = math.floor ( i / 2. )
        if i == 2 * i2:
            break

        i = i2
    return bit


@cython.profile(True)
def i4_sobol_generate ( m, n, skip ):
    """
i4_sobol_generate generates a Sobol dataset.

Parameters:
  Input, integer M, the spatial dimension.
  Input, integer N, the number of points to generate.
  Input, integer SKIP, the number of initial points to skip.

  Output, real R(M,N), the pointstate.
"""
    local_seed = 0
    r=numpy.zeros((m,n))
    state = i4_initialize()
    for j in xrange (1, n+1):
        local_seed = skip + j - 2
        [ r[0:m,j-1], local_seed ,state] = i4_sobol ( m, local_seed,state )
    return r


def i4_initialize() :

    state = I4_State()
    state.maxcol = i4_bit_hi1 ( state.atmost )
    state.poly = i4_getPoly()
    v = i4_initializeV(state.dim_max,state.log_max)
    v[0,0:state.maxcol] = 1
    state.v = v

    return state


@cython.profile(True)
def i4_sobol0(dim_num, l, lastq, v, recipd) :
    cdef numpy.ndarray[numpy.float64_t, ndim=1] cquasi= numpy.zeros(dim_num)
    cdef numpy.ndarray[numpy.float64_t, ndim=1] clastq = lastq
    cdef numpy.ndarray[numpy.long_t, ndim=2] cv = v
    cdef int cdim_num,lowbit,index
    cdef numpy.float64_t crecipd
    lowbit = l
    crecipd = recipd
    cdim_num = dim_num + 1


    for index in xrange( 1, cdim_num):
        cquasi[index -1] = clastq[index-1] * crecipd
        clastq[index-1] =  numpy.bitwise_xor ( int(clastq[index-1]), int(cv[index-1,lowbit-1]) )


    return [cquasi,clastq]
def i4_sobol_quasi(numpy.ndarray[numpy.float64_t, ndim=1] cquasi, numpy.ndarray[numpy.float64_t, ndim=1] clastq,numpy.float64_t crecipd,int cdim_num,result_queue) :

        for index in xrange( 1, cdim_num):
            cquasi[index -1] = clastq[index-1] * crecipd

        result_queue.put(cquasi)

def i4_sobol_lastq(int lowbit,int cdim_num, numpy.ndarray[numpy.float64_t, ndim=1] clastq,numpy.ndarray[numpy.long_t, ndim=2] cv,result_queue) :
        for index in xrange( 1, cdim_num):
            clastq[index-1] = numpy.bitwise_xor ( int(clastq[index-1]), int(cv[index-1,lowbit-1]) )

        result_queue.put(clastq)

def i4_initializeV(dim_max,log_max) :
    cdef numpy.ndarray[numpy.int_t, ndim=2] v = numpy.zeros((dim_max,log_max),dtype=numpy.int)
    v[0:40,0] = numpy.transpose([
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ])

    v[2:40,1] = numpy.transpose([
        1, 3, 1, 3, 1, 3, 3, 1,
        3, 1, 3, 1, 3, 1, 1, 3, 1, 3,
        1, 3, 1, 3, 3, 1, 3, 1, 3, 1,
        3, 1, 1, 3, 1, 3, 1, 3, 1, 3 ])

    v[3:40,2] = numpy.transpose([
        7, 5, 1, 3, 3, 7, 5,
        5, 7, 7, 1, 3, 3, 7, 5, 1, 1,
        5, 3, 3, 1, 7, 5, 1, 3, 3, 7,
        5, 1, 1, 5, 7, 7, 5, 1, 3, 3 ])

    v[5:40,3] = numpy.transpose([
        1, 7, 9,13,11,
        1, 3, 7, 9, 5,13,13,11, 3,15,
        5, 3,15, 7, 9,13, 9, 1,11, 7,
        5,15, 1,15,11, 5, 3, 1, 7, 9 ])

    v[7:40,4] = numpy.transpose([
        9, 3,27,
        15,29,21,23,19,11,25, 7,13,17,
        1,25,29, 3,31,11, 5,23,27,19,
        21, 5, 1,17,13, 7,15, 9,31, 9 ])

    v[13:40,5] = numpy.transpose([
        37,33, 7, 5,11,39,63,
        27,17,15,23,29, 3,21,13,31,25,
        9,49,33,19,29,11,19,27,15,25 ])

    v[19:40,6] = numpy.transpose([
        13,
        33,115, 41, 79, 17, 29,119, 75, 73,105,
        7, 59, 65, 21,  3,113, 61, 89, 45,107 ])

    v[37:40,7] = numpy.transpose([7, 23, 39 ])
    return v

def i4_getPoly() :
    poly= [
        1,   3,   7,  11,  13,  19,  25,  37,  59,  47,
        61,  55,  41,  67,  97,  91, 109, 103, 115, 131,
        193, 137, 145, 143, 241, 157, 185, 167, 229, 171,
        213, 191, 253, 203, 211, 239, 247, 285, 369, 299 ]
    return poly


@cython.profile(True)
def i4_sobol ( dim_num, seed ,state):
    """
    i4_sobol generates a new quasirandom Sobol vector with each call.
    Parameters:
      Input, integer DIM_NUM, the number of spatial dimensionstate.
      DIM_NUM must satisfy 1 <= DIM_NUM <= 40.

      Input/output, integer SEED, the "seed" for the sequence.
      This is essentially the index in the sequence of the quasirandom
      value to be generated.  On output, SEED has been set to the
      appropriate next value, usually simply SEED+1.
      If SEED is less than 0 on input, it is treated as though it were 0.
      An input value of 0 requests the first (0-th) element of the sequence.

      Output, real QUASI(DIM_NUM), the next quasirandom vector.
    """
    #atmost # is a pure function of logmax
    #dim_max # assigned once
    #dim_num_save # re-assigned
    #initialized Set once and never reset
    #lastq
    #log_max CONSTANT 30
    #maxcol #assigned but pure function of atmost
    #poly assigned once
    #recipd #pure
    #seed_save #reassigned
    #v assigned once




    #  Things to do only if the dimension changed.
    if dim_num != state.dim_num_save:

    #  Check parameterstate.
        if dim_num < 1 or state.dim_max < dim_num:
            print 'I4_SOBOL - Fatal error!'
            print '  The spatial dimension DIM_NUM should satisfy:'
            print '    1 <= DIM_NUM <= %d'%state.dim_max
            print '  But this input value is DIM_NUM = %d'%dim_num
            return

        state.dim_num_save = dim_num

        #  Initialize the remaining rows of V.
        for i in xrange(2 , dim_num+1):

        #  The bits of the integer POLY(I) gives the form of polynomial I.
        #  Find the degree of polynomial I from binary encoding.
            j = state.poly[i-1]
            m = 0
            while 1:
                j = math.floor ( j / 2. )
                if j <= 0:
                    break
                m +=  1

            #  Expand this bit pattern to separate components of the logical array INCLUD.
            j = state.poly[i-1]
            includ=numpy.zeros(m)
            for k in xrange(m, 0, -1):
                j2 = math.floor ( j / 2. )
                includ[k-1] =  (j != 2 * j2 )
                j = j2

            #  Calculate the remaining elements of row I as explained
            #  in Bratley and Fox, section 2.
            for j in xrange( m+1, state.maxcol+1 ):
                newv = state.v[i-1,j-m-1]
                l = 1
                for k in xrange(1, m+1):
                    l *= 2
                    if includ[k-1]:
                        newv = numpy.bitwise_xor ( int(newv), int(l * state.v[i-1,j-k-1]) )
                state.v[i-1,j-1] = newv

            #  Multiply columns of V by appropriate power of 2.
        l = 1
        for j in xrange( state.maxcol-1, 0, -1):
            l *= 2
            state.v[0:dim_num,j-1] = state.v[0:dim_num,j-1] * l

        #  RECIPD is 1/(common denominator of the elements in V).
        state.recipd = 1.0 / ( 2 * l )
        state.lastq=numpy.zeros(dim_num)

    seed = int(math.floor ( seed ))

    if seed < 0 or seed == 0:
        seed = 0
        l = 1
        state.lastq=numpy.zeros(dim_num)



    elif seed == state.seed_save + 1:

    #  Find the position of the right-hand zero in SEED.
        l = i4_bit_lo0 ( seed )

    elif seed <= state.seed_save:

        state.seed_save = 0
        l = 1
        state.lastq=numpy.zeros(dim_num)

        for seed_temp in xrange( int(state.seed_save), int(seed)):
            l = i4_bit_lo0 ( seed_temp )
            for i in xrange(1 , dim_num+1):
                state.lastq[i-1] = numpy.bitwise_xor ( int(state.lastq[i-1]), int(state.v[i-1,l-1]) )

        l = i4_bit_lo0 ( seed )

    elif state.seed_save + 1 < seed:

        for seed_temp in xrange( int(state.seed_save + 1), int(seed) ):
            l = i4_bit_lo0 ( seed_temp )
            for i in xrange(1, dim_num+1):
                state.lastq[i-1] = numpy.bitwise_xor ( int(state.lastq[i-1]), int(state.v[i-1,l-1]) )

        l = i4_bit_lo0 ( seed )

    #  Check that the user is not calling too many times!
    if state.maxcol < l:
        print 'I4_SOBOL - Fatal error!'
        print '  Too many calls!'
        print '  MAXCOL = %d\n'%state.maxcol
        print '  L =      %d\n'%l
        return

    #  Calculate the new components of QUASI.


    [quasi,q] = i4_sobol0(dim_num,l,state.lastq,state.v,state.recipd)
    state.lastq = q
    state.seed_save = seed
    seed += 1

    return [ quasi, seed,state ]

def i4_uniform ( a, b, seed ):
    """
    i4_uniform returns a scaled pseudorandom I4.

    Discussion:
      The pseudorandom number will be scaled to be uniformly distributed
      between A and B.

    Reference:
      Paul Bratley, Bennett Fox, Linus Schrage,
      A Guide to Simulation,
      Springer Verlag, pages 201-202, 1983.

      Pierre L'Ecuyer,
      Random Number Generation,
      in Handbook of Simulation,
      edited by Jerry Banks,
      Wiley Interscience, page 95, 1998.

      Bennett Fox,
      Algorithm 647:
      Implementation and Relative Efficiency of Quasirandom
      Sequence Generators,
      ACM Transactions on Mathematical Software,
      Volume 12, Number 4, pages 362-376, 1986.

      Peter Lewis, Allen Goodman, James Miller
      A Pseudo-Random Number Generator for the System/360,
      IBM Systems Journal,
      Volume 8, pages 136-143, 1969.

    Parameters:
      Input, integer A, B, the minimum and maximum acceptable valuestate.
      Input, integer SEED, a seed for the random number generator.

      Output, integer C, the randomly chosen integer.
      Output, integer SEED, the updated seed.
    """
    if seed == 0:
        print 'I4_UNIFORM - Fatal error!'
        print '  Input SEED = 0!'

    seed = math.floor ( seed )
    a = round ( a )
    b = round ( b )

    seed = numpy.mod ( seed, 2147483647 )

    if seed < 0:
        seed += 2147483647

    k = math.floor ( seed / 127773 )

    seed = 16807 * ( seed - k * 127773 ) - k * 2836

    if seed < 0:
        seed += 2147483647

    r = seed * 4.656612875E-10

    #  Scale R to lie between A-0.5 and B+0.5.
    r = ( 1.0 - r ) * ( min ( a, b ) - 0.5 ) + r * ( max ( a, b ) + 0.5 )

    #  Use rounding to convert R to an integer between A and B.
    value = round ( r )

    value = max ( value, min ( a, b ) )
    value = min ( value, max ( a, b ) )

    c = value

    return [ int(c), int(seed) ]


def prime_ge ( n ):
    """
    PRIME_GE returns the smallest prime greater than or equal to N.

    Example:
      +-----+---------
      |   N | PRIME_GE
      +-----+---------
      | -10 |        2
      |   1 |        2
      |   2 |        2
      |   3 |        3
      |   4 |        5
      |   5 |        5
      |   6 |        7
      |   7 |        7
      |   8 |       11
      |   9 |       11
      |  10 |       11

    Parameters:
      Input, integer N, the number to be bounded.

      Output, integer P, the smallest prime number that is greater
      than or equal to N.
    """
    p = max ( math.ceil ( n ), 2 )
    while not is_prime ( p ):
        p += 1

    return p

def is_prime(n):
    """
    is_prime returns True if N is a prime number, False otherwise

    Parameters:
       Input, integer N, the number to be checked.

       Output, boolean value, True or False
    """
    if n!=int(n) or n<1:
        return False
    p=2
    while p<n:
        if n%p==0:
            return False
        p+=1
    return True

