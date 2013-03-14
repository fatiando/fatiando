         3309 function calls in 0.965 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      299    0.325    0.001    0.325    0.001 {fatiando.seismic._cwavefd._step_elastic_psv_x}
      299    0.316    0.001    0.316    0.001 {fatiando.seismic._cwavefd._step_elastic_psv_z}
      598    0.301    0.001    0.301    0.001 {fatiando.seismic._cwavefd._apply_damping}
      302    0.011    0.000    0.965    0.003 wavefd.py:502(elastic_psv)
      600    0.009    0.000    0.009    0.000 wavefd.py:229(__call__)
      598    0.001    0.000    0.001    0.000 {fatiando.seismic._cwavefd._boundary_conditions}
        2    0.001    0.001    0.001    0.001 wavefd.py:331(_add_pad)
        1    0.000    0.000    0.965    0.965 <ipython-input-13-887484d2d225>:1(run)
      600    0.000    0.000    0.000    0.000 wavefd.py:237(coords)
        4    0.000    0.000    0.000    0.000 {numpy.core.multiarray.zeros}
        3    0.000    0.000    0.000    0.000 wavefd.py:545(<genexpr>)
        1    0.000    0.000    0.965    0.965 <string>:1(<module>)
        1    0.000    0.000    0.000    0.000 {max}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}