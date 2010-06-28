import cProfile
import pstats

cProfile.run("import pgrav_example; pgrav_example.main()", 'pgrav.prof')

stats = pstats.Stats('pgrav.prof')
print "Cumulative time"
stats.strip_dirs().sort_stats('cumulative').print_stats(20)
print "\nFunction time"
stats.strip_dirs().sort_stats('time').print_stats(20)
