ncalls
    for the number of calls,
tottime
    for the total time spent in the given function (and excluding time made in
    calls to sub-functions),
percall
    is the quotient of tottime divided by ncalls
cumtime
    is the total time spent in this and all subfunctions (from invocation till
    exit). This figure is accurate even for recursive functions.
percall
    is the quotient of cumtime divided by primitive calls
filename:lineno(function)
    provides the respective data of each function

When there are two numbers in the first column (for example, 43/3), then the
latter is the number of primitive calls, and the former is the actual number of
calls. Note that when the function does not recurse, these two values are the
same, and only the single figure is printed.
